#pragma once

#include <vector>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <future>
#include <memory>

#include <pybind11/embed.h>

namespace az73 {

// A single evaluation request: holds the flattened board tensor and a promise
struct EvalRequest {
    std::vector<float> tensor;
    std::promise<std::pair<std::vector<float>, float>> promise;
};

class __attribute__((visibility("hidden"))) BatchManager {
public:
    // Get the singleton instance
    static BatchManager& instance();

    // Initialize with the Python Agent object and desired batch size
    void init(pybind11::object agent_py, size_t batch_size);

    // Enqueue a flat tensor, return a future for {policy, value}
    std::future<std::pair<std::vector<float>, float>>
    enqueue(const std::vector<float>& flat_tensor);

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
    pybind11::object                         agent_py_;
};

} // namespace az73
