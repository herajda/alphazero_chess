#include "batch_manager.hpp"
#include <thread>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <exception>
#include <torch/torch.h>
#include <torch/script.h>

namespace az73 {

namespace {

bool perf_debug_enabled() {
    static const bool enabled = []() {
        if (const char* env = std::getenv("AZ_PERF_DEBUG")) {
            return env[0] != '0';
        }
        return false;
    }();
    return enabled;
}

int perf_log_interval_ms() {
    static const int interval = []() {
        if (const char* env = std::getenv("AZ_PERF_LOG_MS")) {
            int v = std::atoi(env);
            return v > 0 ? v : 1000;
        }
        return 1000;
    }();
    return interval;
}

} // namespace

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

void BatchManager::init(const std::string& model_path, size_t batch_size, int wait_ms) {
    std::unique_lock<std::mutex> lk(mtx_);
    if (running_ && current_path_ == model_path) {
        batch_size_ = batch_size;
        flush_wait_ms_ = wait_ms;
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
    module_dtype_ = torch::kFloat32;
    for (const auto& named_param : module_.named_parameters()) {
        module_dtype_ = named_param.value.scalar_type();
        break;
    }
    module_.to(device_, module_dtype_);
    batch_size_ = std::max<size_t>(1, batch_size);
    flush_wait_ms_ = wait_ms > 0 ? wait_ms : 2;
    if (perf_debug_enabled()) {
        std::cerr << "[AZ][perf] BatchManager init path=" << model_path
                  << " device=" << (device_.is_cuda() ? "cuda" : "cpu")
                  << " dtype=" << c10::toString(module_dtype_)
                  << " batch_size=" << batch_size_
                  << " flush_wait_ms=" << flush_wait_ms_
                  << std::endl;
    }
    current_path_ = model_path;
    running_ = true;
    runner_thread_ = std::thread(&BatchManager::run_loop, this);
}

std::future<std::pair<std::vector<float>, float>> BatchManager::enqueue(const std::vector<float>& flat_tensor) {
    auto req = std::make_shared<EvalRequest>();
    req->tensor = flat_tensor;
    req->enqueue_time = std::chrono::steady_clock::now();
    auto fut = req->promise.get_future();
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!running_ && perf_debug_enabled()) {
            std::cerr << "[AZ][perf][batch] enqueue while not running; pending requests will stall\n";
        }
        queue_.push_back(req);
    }
    cv_.notify_one();
    return fut;
}

void BatchManager::run_loop() {
    using namespace std::chrono;
    using clock = std::chrono::steady_clock;
    auto last_log = clock::now();
    if (perf_debug_enabled()) {
        std::cerr << "[AZ][perf][batch] runner thread started\n";
    }
    while (true) {
        std::vector<std::shared_ptr<EvalRequest>> batch;
        size_t backlog_after = 0;
        auto collect_start = clock::now();
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [&] { return !queue_.empty() || !running_; });
            if (!running_ && queue_.empty())
                break;
            auto deadline = clock::now() + std::chrono::milliseconds(flush_wait_ms_);
            while (queue_.size() < batch_size_ && running_) {
                if (cv_.wait_until(lk, deadline, [&] {
                        return queue_.size() >= batch_size_ || !running_;
                    })) {
                    break;
                }
                if (clock::now() >= deadline) {
                    break;
                }
            }
            while (!queue_.empty() && batch.size() < batch_size_) {
                batch.push_back(queue_.front());
                queue_.pop_front();
            }
            backlog_after = queue_.size();
        }
        if (batch.empty()) continue;
        auto after_collect = clock::now();
        auto wait_ms = duration_cast<milliseconds>(after_collect - collect_start).count();
        auto oldest_wait_ms = duration_cast<milliseconds>(after_collect - batch.front()->enqueue_time).count();
        size_t B = batch.size();
        std::vector<int64_t> dims = { (int64_t)B, 8, 8, 119 };
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        at::Tensor input = torch::empty(dims, options);
        float* ptr = input.data_ptr<float>();
        for (size_t i = 0; i < B; ++i) {
            std::memcpy(ptr + i * 8 * 8 * 119, batch[i]->tensor.data(), 8 * 8 * 119 * sizeof(float));
        }
        input = input.to(device_, module_dtype_);
        auto forward_start = clock::now();
        try {
            torch::InferenceMode guard;
            auto outputs = module_.forward({input}).toTuple();
            auto forward_end = clock::now();
            at::Tensor pol_t = outputs->elements()[0].toTensor().to(torch::kCPU, torch::kFloat32);
            at::Tensor val_t = outputs->elements()[1].toTensor().to(torch::kCPU, torch::kFloat32);
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

            if (perf_debug_enabled()) {
                auto now = clock::now();
                if (now - last_log >= milliseconds(perf_log_interval_ms())) {
                    last_log = now;
                    auto forward_ms = duration_cast<milliseconds>(forward_end - forward_start).count();
                    auto total_ms = duration_cast<milliseconds>(forward_end - collect_start).count();
                    double samples_per_s = forward_ms > 0 ? (1000.0 * static_cast<double>(B) / forward_ms) : 0.0;
                    std::cerr << "[AZ][perf][batch] size=" << B
                              << " wait_ms=" << wait_ms
                              << " oldest_wait_ms=" << oldest_wait_ms
                              << " forward_ms=" << forward_ms
                              << " total_ms=" << total_ms
                              << " queue_after=" << backlog_after
                              << " flush_wait_ms=" << flush_wait_ms_
                              << " est_samples_per_s=" << samples_per_s
                              << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[AZ][perf][batch] forward exception: " << e.what() << std::endl;
            for (auto& r : batch) {
                r->promise.set_exception(std::make_exception_ptr(e));
            }
            // Stop processing further to surface error to callers.
            break;
        } catch (...) {
            std::cerr << "[AZ][perf][batch] forward unknown exception\n";
            for (auto& r : batch) {
                r->promise.set_exception(std::current_exception());
            }
            break;
        }
    }
}

} // namespace az73
