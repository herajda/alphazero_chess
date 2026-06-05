#include "batch_manager.hpp"
#include <thread>
#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/Context.h>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace az73 {

namespace {

std::size_t env_size_t(const char* name, std::size_t fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    char* end = nullptr;
    unsigned long long value = std::strtoull(raw, &end, 10);
    if (end == raw) return fallback;
    return static_cast<std::size_t>(value);
}

std::string basename_for_log(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

bool env_enabled(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return false;
    return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0 && std::strcmp(raw, "FALSE") != 0;
}

struct BatchLogStats {
    std::string model;
    std::string precision;
    std::size_t configured_batch = 0;
    std::size_t log_first = 20;
    std::size_t log_every = 1000;
    std::size_t gather_us = 0;
    std::uint64_t batches = 0;
    std::uint64_t requests = 0;
    std::uint64_t last_log_batches = 0;
    std::uint64_t last_log_requests = 0;
    std::size_t max_batch = 0;
    std::array<std::uint64_t, 8> buckets{};
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_log_time = started;

    BatchLogStats(std::string model_path, std::size_t batch_size, std::size_t gather_micros, std::string precision_mode)
        : model(basename_for_log(model_path)), precision(std::move(precision_mode)), configured_batch(batch_size), gather_us(gather_micros) {
        log_first = env_size_t("AZ_BATCH_LOG_FIRST", 20);
        log_every = env_size_t("AZ_BATCH_LOG_EVERY", 1000);
        if (log_first || log_every) {
            std::fprintf(
                stderr,
                "[AZ][batch] model=%s precision=%s configured=%zu gather_us=%zu log_first=%zu log_every=%zu\n",
                model.c_str(), precision.c_str(), configured_batch, gather_us, log_first, log_every);
            std::fflush(stderr);
        }
    }

    static std::size_t bucket_for(std::size_t size) {
        if (size <= 1) return 0;
        if (size == 2) return 1;
        if (size <= 4) return 2;
        if (size <= 8) return 3;
        if (size <= 16) return 4;
        if (size <= 32) return 5;
        if (size <= 64) return 6;
        return 7;
    }

    void observe(std::size_t batch_size, std::size_t queued_after) {
        ++batches;
        requests += batch_size;
        if (batch_size > max_batch) max_batch = batch_size;
        ++buckets[bucket_for(batch_size)];

        if (log_first && batches <= log_first) {
            std::fprintf(
                stderr,
                "[AZ][batch] model=%s sample=%llu batch=%zu queued_after=%zu configured=%zu gather_us=%zu\n",
                model.c_str(),
                static_cast<unsigned long long>(batches),
                batch_size,
                queued_after,
                configured_batch,
                gather_us);
            std::fflush(stderr);
        }
    }

    void maybe_log(bool force = false) {
        if (!log_every && !force) return;
        if (!force && batches - last_log_batches < log_every) return;
        if (batches == last_log_batches) return;

        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - started).count() / 1000.0;
        const auto delta_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_log_time).count() / 1000.0;
        const auto delta_batches = batches - last_log_batches;
        const auto delta_requests = requests - last_log_requests;
        const double avg = batches ? static_cast<double>(requests) / static_cast<double>(batches) : 0.0;
        const double recent_avg = delta_batches ? static_cast<double>(delta_requests) / static_cast<double>(delta_batches) : 0.0;
        const double req_s = elapsed > 0.0 ? static_cast<double>(requests) / elapsed : 0.0;
        const double recent_req_s = delta_elapsed > 0.0 ? static_cast<double>(delta_requests) / delta_elapsed : 0.0;

        std::fprintf(
            stderr,
            "[AZ][batch] model=%s precision=%s configured=%zu gather_us=%zu batches=%llu requests=%llu avg=%.3f recent_avg=%.3f "
            "req_s=%.1f recent_req_s=%.1f max=%zu "
            "hist{1=%llu,2=%llu,3-4=%llu,5-8=%llu,9-16=%llu,17-32=%llu,33-64=%llu,65+=%llu} elapsed_s=%.3f%s\n",
            model.c_str(),
            precision.c_str(),
            configured_batch,
            gather_us,
            static_cast<unsigned long long>(batches),
            static_cast<unsigned long long>(requests),
            avg,
            recent_avg,
            req_s,
            recent_req_s,
            max_batch,
            static_cast<unsigned long long>(buckets[0]),
            static_cast<unsigned long long>(buckets[1]),
            static_cast<unsigned long long>(buckets[2]),
            static_cast<unsigned long long>(buckets[3]),
            static_cast<unsigned long long>(buckets[4]),
            static_cast<unsigned long long>(buckets[5]),
            static_cast<unsigned long long>(buckets[6]),
            static_cast<unsigned long long>(buckets[7]),
            elapsed,
            force ? " final=1" : "");
        std::fflush(stderr);

        last_log_batches = batches;
        last_log_requests = requests;
        last_log_time = now;
    }
};

}  // namespace

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
    } else if (runner_thread_.joinable()) {
        lk.unlock();
        runner_thread_.join();
        lk.lock();
        queue_.clear();
    }

    batch_size_ = batch_size;
    current_path_ = model_path;
    running_ = true;

    auto ready = std::make_shared<std::promise<void>>();
    auto ready_future = ready->get_future();
    runner_thread_ = std::thread(&BatchManager::run_loop, this, model_path, ready);
    lk.unlock();
    ready_future.get();
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

void BatchManager::run_loop(std::string model_path, std::shared_ptr<std::promise<void>> ready) {
    using namespace std::chrono;
    bool use_fp16 = false;
    try {
        if (std::getenv("AZ_DISABLE_CUDNN")) {
            at::globalContext().setUserEnabledCuDNN(false);
        }
        device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        use_fp16 = device_.is_cuda() && env_enabled("AZ_INFERENCE_FP16");
        module_ = torch::jit::load(model_path);
        module_.eval();
        if (env_enabled("AZ_ENABLE_JIT_OPT")) {
            module_ = torch::jit::freeze(module_);
            module_ = torch::jit::optimize_for_inference(module_);
        }
        if (use_fp16) {
            module_.to(device_, torch::kFloat16);
        } else {
            module_.to(device_);
        }

        torch::InferenceMode guard;
        const auto inference_dtype = use_fp16 ? torch::kFloat16 : torch::kFloat32;
        auto warmup = torch::zeros({1, 8, 8, 119}, torch::TensorOptions().dtype(inference_dtype).device(device_));
        (void)module_.forward({warmup});
        ready->set_value();
    } catch (...) {
        ready->set_exception(std::current_exception());
        std::lock_guard<std::mutex> lk(mtx_);
        running_ = false;
        return;
    }

    const std::size_t gather_us = env_size_t("AZ_BATCH_GATHER_US", 500);
    BatchLogStats batch_log(model_path, batch_size_, gather_us, use_fp16 ? "fp16" : "fp32");

    while (true) {
        std::vector<std::shared_ptr<EvalRequest>> batch;
        std::size_t queued_after = 0;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait_for(lk, std::chrono::milliseconds(2), [&] {
                return !queue_.empty() || !running_;
            });
            if (!running_ && queue_.empty())
                break;

            auto drain_queue = [&] {
                while (!queue_.empty() && batch.size() < batch_size_) {
                    batch.push_back(queue_.front());
                    queue_.pop_front();
                }
            };

            drain_queue();
            if (running_ && gather_us > 0 && batch.size() < batch_size_) {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(gather_us);
                while (batch.size() < batch_size_) {
                    if (queue_.empty()) {
                        if (!cv_.wait_until(lk, deadline, [&] { return !queue_.empty() || !running_; })) {
                            break;
                        }
                    }
                    drain_queue();
                    if (!running_ || std::chrono::steady_clock::now() >= deadline) {
                        break;
                    }
                }
            }
            queued_after = queue_.size();
        }
        if (batch.empty()) continue;
        size_t B = batch.size();
        batch_log.observe(B, queued_after);
        batch_log.maybe_log();
        std::vector<int64_t> dims = { (int64_t)B, 8, 8, 119 };
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        at::Tensor input = torch::empty(dims, options);
        float* ptr = input.data_ptr<float>();
        for (size_t i = 0; i < B; ++i) {
            std::memcpy(ptr + i * 8 * 8 * 119, batch[i]->tensor.data(), 8 * 8 * 119 * sizeof(float));
        }
        input = use_fp16 ? input.to(device_, torch::kFloat16) : input.to(device_);
        torch::InferenceMode guard;
        auto outputs = module_.forward({input}).toTuple();
        at::Tensor pol_t = outputs->elements()[0].toTensor().to(torch::kFloat32).to(torch::kCPU).contiguous();
        at::Tensor val_t = outputs->elements()[1].toTensor().to(torch::kFloat32).to(torch::kCPU).contiguous();
        const auto A = static_cast<std::size_t>(pol_t.size(1));
        const float* pol_ptr = pol_t.data_ptr<float>();
        const float* val_ptr = val_t.data_ptr<float>();
        for (size_t i = 0; i < B; ++i) {
            std::vector<float> policy(A);
            std::memcpy(policy.data(), pol_ptr + i * A, A * sizeof(float));
            float v = val_ptr[i];
            batch[i]->promise.set_value({std::move(policy), v});
        }
    }
    batch_log.maybe_log(true);
}

} // namespace az73
