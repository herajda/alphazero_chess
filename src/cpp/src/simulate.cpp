// src/cpp/src/simulate.cpp
#include "simulate.hpp"
#include "mcts.hpp"
#include "chess_game.hpp"
#include "batch_manager.hpp"

#include <pybind11/stl.h>
#include <thread>
#include <mutex>
#include <fstream>
#include <vector>
#include <cstdint>
#include <random>

namespace az73 {

// Number of floats per record: state (8×8×119) + policy (8×8×73) + z
static constexpr uint64_t RECORD_FLOATS = 8ULL*8*119 + 8ULL*8*73 + 1;
static constexpr uint64_t RECORD_BYTES  = RECORD_FLOATS * sizeof(float);
static constexpr uint64_t HEADER_BYTES  = 24;  // [capacity(8), size(8), head(8)]

// Helpers to read/write 64‑bit header fields
static int64_t read_i64(std::fstream &f, std::streamoff off) {
    int64_t v;
    f.seekg(off);
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}
static void write_i64(std::fstream &f, std::streamoff off, int64_t v) {
    f.seekp(off);
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

void simulate_games_buffered(
    const std::string &model_path,
    int num_games,
    int num_threads,
    int num_simulations,
    double alpha,
    double epsilon,
    int sampling_moves,
    const std::string &filename,
    int64_t capacity
) {
    // 1) Load model into BatchManager
    BatchManager::instance().init(model_path, num_threads);
    py::gil_scoped_release no_gil;

    // 2) Open (or create) the ring‑buffer file
    std::fstream f(filename, std::ios::in|std::ios::out|std::ios::binary);
    if (!f) {
        // File doesn't exist: create and initialize header + preallocate
        f.open(filename, std::ios::out|std::ios::binary);
        int64_t zero = 0;
        f.write(reinterpret_cast<char*>(&capacity), sizeof(capacity));
        f.write(reinterpret_cast<char*>(&zero), sizeof(zero));   // size = 0
        f.write(reinterpret_cast<char*>(&zero), sizeof(zero));   // head = 0
        // Preallocate all records to zero
        std::vector<char> blank(capacity * RECORD_BYTES, 0);
        f.write(blank.data(), blank.size());
        f.close();
        // Reopen for read/write
        f.open(filename, std::ios::in|std::ios::out|std::ios::binary);
    }

    // 3) Read header fields
    int64_t size = read_i64(f, 8);
    int64_t head = read_i64(f, 16);

    std::mutex file_mtx;

    // Append one (state,policy,z) into the ring buffer
    auto append_one = [&](const std::vector<float> &state,
                          const std::vector<float> &policy,
                          float z) {
        std::lock_guard lk(file_mtx);
        uint64_t idx    = head % capacity;
        uint64_t offset = HEADER_BYTES + idx * RECORD_BYTES;
        f.seekp(offset);
        // write state
        f.write(reinterpret_cast<const char*>(state.data()),
                state.size() * sizeof(float));
        // write policy
        f.write(reinterpret_cast<const char*>(policy.data()),
                policy.size() * sizeof(float));
        // write z
        f.write(reinterpret_cast<const char*>(&z), sizeof(z));
        // update header
        head = (head + 1) % capacity;
        if (size < capacity) ++size;
        write_i64(f, 8,  size);
        write_i64(f, 16, head);
    };

    // Worker lambda: simulate `count` games
    auto worker = [&](int count) {
        thread_local std::mt19937_64 rng(std::random_device{}());
        for (int i = 0; i < count; ++i) {
            ChessGame game;
            Trajectory traj;
            MCTArgs args{num_simulations, alpha, epsilon, sampling_moves};

            // play game
            while (!game.winner().has_value()) {
                auto policy = run_mcts(game, args);
                int action;
                if ((int)traj.size() >= sampling_moves) {
                    action = std::distance(
                        policy.begin(),
                        std::max_element(policy.begin(), policy.end())
                    );
                } else {
                    std::discrete_distribution<int> dist(policy.begin(), policy.end());
                    action = dist(rng);
                }
                auto tensor = game.encodeTensor();
                std::vector<float> flat(tensor.begin(), tensor.end());
                traj.emplace_back(flat, policy, (float)game.to_play());
                game.makeMove((uint16_t)action);
            }

            // compute z and append each step
            int w = *game.winner();
            for (auto &e : traj) {
                float to_play = std::get<2>(e);
                float z_val   = (w == -1 ? 0.0f
                                 : to_play == (float)w ? 1.0f
                                                       : -1.0f);
                std::get<2>(e) = z_val;
                append_one(std::get<0>(e), std::get<1>(e), z_val);
            }
        }
    };

    // 4) Launch threads
    int per     = (num_games + num_threads - 1) / num_threads;
    int started = 0;
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads && started < num_games; ++t) {
        int cnt = std::min(per, num_games - started);
        threads.emplace_back(worker, cnt);
        started += cnt;
    }
    for (auto &th : threads) th.join();
    f.close();
}

} // namespace az73

PYBIND11_MODULE(chess_engine, m) {
    m.doc() = "C++ self-play with batched inference and on‑disk ring buffer";
    m.def("simulate_games_buffered",
          &az73::simulate_games_buffered,
          py::arg("model_path"),
          py::arg("num_games"),
          py::arg("num_threads"),
          py::arg("num_simulations"),
          py::arg("alpha"),
          py::arg("epsilon"),
          py::arg("sampling_moves"),
          py::arg("filename"),
          py::arg("replay_buffer_capacity"));
}
