#pragma once

#include <vector>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <future>
#include <memory>
#include <chrono>
#include <torch/script.h>

namespace az73 {

// A single evaluation request: holds the flattened board tensor and a promise
struct EvalRequest {
    std::vector<float> tensor;
    std::promise<std::pair<std::vector<float>, float>> promise;
    std::chrono::steady_clock::time_point enqueue_time;
};

class __attribute__((visibility("hidden"))) BatchManager {
public:
    // Get the singleton instance
    static BatchManager& instance();

    // Initialize with the path to a TorchScript module and batch size
    // wait_ms controls how long we wait to gather a fuller batch.
    void init(const std::string& model_path, size_t batch_size, int wait_ms = 2);

    // Enqueue a flat tensor, return a future for {policy, value}
    std::future<std::pair<std::vector<float>, float>>
    enqueue(const std::vector<float>& flat_tensor);
    std::string current_path_;
private:
    BatchManager() = default;
    ~BatchManager();

    // Main loop that gathers requests and calls into Python
    void run_loop();


    std::mutex                               mtx_;
    std::condition_variable                  cv_;
    std::deque<std::shared_ptr<EvalRequest>> queue_;
    bool                                     running_{false};
    size_t                                   batch_size_{1};
    int                                      flush_wait_ms_{2};
    torch::jit::script::Module               module_;   // TorchScript graph
    torch::Device                            device_{torch::kCPU};
    c10::ScalarType                          module_dtype_{torch::kFloat32};
    std::thread                              runner_thread_;
};

} // namespace az73
