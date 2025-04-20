#include "batch_manager.hpp"
#include <pybind11/numpy.h>
#include <thread>
#include <chrono>
#include <iostream>

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

void BatchManager::init(pybind11::object agent_py, size_t batch_size) {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        agent_py_   = std::move(agent_py);
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

        // Acquire GIL for Python calls
        pybind11::gil_scoped_acquire acquire;

        // Build NumPy array of shape [B,8,8,119]
        size_t B = batch.size();
        std::vector<pybind11::ssize_t> shape = { (pybind11::ssize_t)B, 8, 8, 119 };
        pybind11::array_t<float> input(shape);
        auto buf = input.mutable_unchecked<4>();
        for (size_t i = 0; i < B; ++i) {
            auto const& flat = batch[i]->tensor;
            for (size_t j = 0; j < flat.size(); ++j) {
                size_t plane = j / 64;
                size_t rem   = j % 64;
                size_t r     = rem / 8;
                size_t c     = rem % 8;
                buf(i, r, c, plane) = flat[j];
            }
        }

        // Call Python: (policy, value) = agent_py_.predict(input)
        auto result = agent_py_.attr("predict")(input).cast<pybind11::tuple>();

        // element 0: policy array [B, A]
        auto pol_array = result[0].cast<pybind11::array_t<float>>();
        auto pol       = pol_array.unchecked<2>();  

        // element 1: value array  [B, 1]
        auto val_array = result[1].cast<pybind11::array_t<float>>();
        auto val       = val_array.unchecked<2>();

        // Dispatch results to promises
        for (size_t i = 0; i < B; ++i) {
            size_t A = pol.shape(1);
            std::vector<float> policy(A);
            for (size_t a = 0; a < A; ++a)
                policy[a] = pol(i, a);
            float v = val(i, 0);
            batch[i]->promise.set_value({std::move(policy), v});
        }
    }
}

} // namespace az73
