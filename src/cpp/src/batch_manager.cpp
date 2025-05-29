#include "batch_manager.hpp"
#include <thread>
#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>

namespace az73 {

BatchManager& BatchManager::instance() {
    static BatchManager instance;
    return instance;
}

BatchManager::~BatchManager() {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        running_ = false;
    }
    cv_.notify_all();
    if (runner_thread_.joinable())
        runner_thread_.join();
}

void BatchManager::init(const std::string& model_path, size_t batch_size) {
    std::unique_lock<std::mutex> lk(mtx_);
    if (running_ && current_path_ == model_path) {
        batch_size_ = batch_size;
        return;
    }
    if (running_) {
        running_ = false;
        cv_.notify_all();
        lk.unlock();
        runner_thread_.join();
        lk.lock();
        queue_.clear();
    }
    module_ = torch::jit::load(model_path);
    module_.eval();
    module_ = torch::jit::freeze(module_);
    module_ = torch::jit::optimize_for_inference(module_);
    device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    module_.to(device_);
    batch_size_ = batch_size;
    current_path_ = model_path;
    running_ = true;
    runner_thread_ = std::thread(&BatchManager::run_loop, this);
}

std::future<std::pair<std::vector<float>, float>> BatchManager::enqueue(const std::vector<float>& flat_tensor) {
    auto req = std::make_shared<EvalRequest>();
    req->tensor = flat_tensor;
    auto fut = req->promise.get_future();
    {
        std::lock_guard<std::mutex> lk(mtx_);
        queue_.push_back(req);
    }
    cv_.notify_one();
    return fut;
}

void BatchManager::run_loop() {
    using namespace std::chrono;
    while (true) {
        std::vector<std::shared_ptr<EvalRequest>> batch;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait_for(lk, std::chrono::milliseconds(2), [&] {
                return !queue_.empty() || !running_;
            });
            if (!running_ && queue_.empty())
                break;
            while (!queue_.empty() && batch.size() < batch_size_) {
                batch.push_back(queue_.front());
                queue_.pop_front();
            }
        }
        if (batch.empty()) continue;
        size_t B = batch.size();
        //std::cout << "batch size: " << B << "\n";
        std::vector<int64_t> dims = { (int64_t)B, 8, 8, 119 };
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        at::Tensor input = torch::empty(dims, options);
        float* ptr = input.data_ptr<float>();
        for (size_t i = 0; i < B; ++i) {
            std::memcpy(ptr + i * 8 * 8 * 119, batch[i]->tensor.data(), 8 * 8 * 119 * sizeof(float));
        }
        input = input.to(device_);
        torch::InferenceMode guard;
        auto outputs = module_.forward({input}).toTuple();
        at::Tensor pol_t = outputs->elements()[0].toTensor().to(torch::kCPU);
        at::Tensor val_t = outputs->elements()[1].toTensor().to(torch::kCPU);
        auto pol_acc = pol_t.accessor<float,2>();
        auto val_acc = val_t.accessor<float,2>();
        for (size_t i = 0; i < B; ++i) {
            size_t A = pol_acc.size(1);
            std::vector<float> policy(A);
            for (size_t a = 0; a < A; ++a)
                policy[a] = pol_acc[i][a];
            float v = val_acc[i][0];
            batch[i]->promise.set_value({std::move(policy), v});
        }
    }
}

} // namespace az73
