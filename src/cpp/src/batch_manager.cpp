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
    }

    void BatchManager::init(const std::string& model_path, size_t batch_size) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            // Load the serialized TorchScript module
            module_ = torch::jit::load(model_path);
            module_.eval();
            module_ = torch::jit::freeze(module_);
            module_ = torch::jit::optimize_for_inference(module_);
            // Choose device: use CUDA if available, else CPU
            if (torch::cuda::is_available()) {
                device_ = torch::kCUDA;
            } else {
                device_ = torch::kCPU;
            }
            module_.to(device_);
            batch_size_ = batch_size;
            running_    = true;
        }
        // Start the runner thread
        std::thread(&BatchManager::run_loop, this).detach();
    }

    std::future<std::pair<std::vector<float>, float>>
        BatchManager::enqueue(const std::vector<float>& flat_tensor) {
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
                // wait until we have at least one request or shutting down
                //
                cv_.wait_for(lk, std::chrono::milliseconds(1), [&](){
                        return !queue_.empty() || !running_;
                        });

                if (!running_ && queue_.empty())
                    break;
                // gather up to batch_size_ requests
                while (!queue_.empty() && batch.size() < batch_size_) {
                    batch.push_back(queue_.front());
                    queue_.pop_front();
                }
            }
            if (batch.empty()) continue;

            // Build a [B,8,8,119] CPU tensor then permute -> [B,119,8,8]
            size_t B = batch.size();
            std::vector<int64_t> dims = { (int64_t)B, 8, 8, 119 };
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            at::Tensor input = torch::empty(dims, options);
            float* ptr = input.data_ptr<float>();
            for (size_t i = 0; i < B; ++i) {
                std::memcpy(ptr + i * 8 * 8 * 119,
                        batch[i]->tensor.data(),
                        8 * 8 * 119 * sizeof(float));
            }

            input = input.to(device_);
            // Forward through TorchScript
            auto start = high_resolution_clock::now();
            torch::InferenceMode guard;
            auto outputs = module_.forward({input}).toTuple();
            //auto end = high_resolution_clock::now();
        
            //auto duration = duration_cast<nanoseconds>(end - start).count();
            //std::cout << "Time taken: " << duration << " nanoseconds" << std::endl;

            at::Tensor pol_t = outputs->elements()[0].toTensor().to(torch::kCPU);
            at::Tensor val_t = outputs->elements()[1].toTensor().to(torch::kCPU);
            auto pol_acc = pol_t.accessor<float,2>();
            auto val_acc = val_t.accessor<float,2>();

            // Dispatch results to promises
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
