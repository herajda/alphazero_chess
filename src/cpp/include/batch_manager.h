#ifndef BATCH_MANAGER_H
#define BATCH_MANAGER_H

#include <pybind11/pybind11.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <future>
#include <vector>

namespace py = pybind11;

class BatchManager {
public:
    BatchManager(py::object eval_callback, int batch_size = 32);
    ~BatchManager();
    std::pair<std::vector<float>, float> submit(const std::vector<float>& tensor);

private:
    struct EvalRequest {
        std::vector<float> tensor;
        std::promise<std::pair<std::vector<float>, float>> result;
    };

    std::queue<EvalRequest> request_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    py::object eval_callback;
    std::thread batch_thread;
    bool running = true;
    const int batch_size;

    void process_batch();
};

#endif
