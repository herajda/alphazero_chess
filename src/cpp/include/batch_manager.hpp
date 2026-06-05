#pragma once

#include <vector>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <future>
#include <memory>
#include <string>
#include <torch/script.h>

namespace az73 {

// A single evaluation request: holds the flattened board tensor and a promise
struct EvalRequest {
    std::vector<float> tensor;
    std::promise<std::pair<std::vector<float>, float>> promise;
};

class __attribute__((visibility("hidden"))) BatchManager {
public:
    // Get the singleton instance used by legacy single-model call sites.
    static BatchManager& instance();

    BatchManager() = default;
    ~BatchManager();
    BatchManager(const BatchManager&) = delete;
    BatchManager& operator=(const BatchManager&) = delete;

    // Initialize with the path to a TorchScript module and batch size.
    void init(const std::string& model_path, size_t batch_size);

    // Enqueue a flat tensor, return a future for {policy, value}.
    std::future<std::pair<std::vector<float>, float>>
    enqueue(const std::vector<float>& flat_tensor);
    std::string current_path_;
private:
    // Main loop that gathers requests and runs batched TorchScript inference.
    void run_loop(std::string model_path, std::shared_ptr<std::promise<void>> ready);


    std::mutex                               mtx_;
    std::condition_variable                  cv_;
    std::deque<std::shared_ptr<EvalRequest>> queue_;
    bool                                     running_{false};
    size_t                                   batch_size_{1};
    torch::jit::script::Module               module_;   // TorchScript graph
    torch::Device                            device_{torch::kCPU};
    std::thread                              runner_thread_;
};

} // namespace az73
