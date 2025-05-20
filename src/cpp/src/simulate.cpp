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
#include <iostream>
#include <filesystem>
#include <atomic>

namespace az73 {
// --------------------------------------------------------------------
// Evaluate vs random
// --------------------------------------------------------------------
std::tuple<int,int,int,int,int,int>
evaluate_vs_random(
    const std::string &model_path,
    int num_games_per_color,
    int num_threads,
    int num_simulations,
    double alpha,
    double epsilon,
    int sampling_moves
) {
    // 1) initialize model (batch size == num_threads)
    BatchManager::instance().init(model_path, num_threads);
    // release GIL while doing C++ work
    py::gil_scoped_release no_gil;

    // 2) counters
    std::atomic<int> w_win{0}, w_loss{0}, w_draw{0};
    std::atomic<int> b_win{0}, b_loss{0}, b_draw{0};

    // Worker: each thread plays up to `per` White games and `per` Black games
    auto worker = [&](int per_color) {
        std::mt19937_64 rng{std::random_device{}()};
        for (int color : {1, 0}) {  // 1 = White, 0 = Black
            for (int i = 0; i < per_color; ++i) {
                ChessGame game;
                MCTArgs args{ num_simulations, alpha, epsilon, sampling_moves };
                // play until end
                while (!game.winner().has_value()) {
                    if (game.to_play() == color) {
                        // our agent
                        auto policy = run_mcts(game, args);
                        //for (size_t i = 0; i < policy.size(); ++i) {
                        //    if (policy[i] != 0.0f) {
                        //        std::cout << i << ": " << policy[i] << "\n";
                        //    }
                        //}
                        auto legal = game.legalMoves();
                        //std::cout << "[";
                        //for (size_t i = 0; i < legal.size(); ++i) {
                        //    std::cout << legal[i];
                        //    if (i + 1 != legal.size())
                        //        std::cout << ", ";
                        //}
                        //std::cout << "]" << std::endl << std::endl;

                        // pick best move (no noise/exploration)
                        uint16_t best = legal.front();
                        float best_p = policy[best];
                        for (auto a : legal) {
                            if (policy[a] > best_p) {
                                best_p = policy[a];
                                best = a;
                            }
                        }
                        std::cout << "Best move: " << best << " (p=" << best_p << ") UCI: " 
                        << chess::uci::moveToUci(az73::decode_action(best, game.currentBoard())) << std::endl;
                        game.makeMove(best);
                    } else {
                        // random baseline
                        auto legal = game.legalMoves();
                        std::uniform_int_distribution<size_t> uni(0, legal.size() - 1);
                        size_t random_idx = uni(rng);
                        std::cout << "Random move UCI: " << chess::uci::moveToUci(az73::decode_action(legal[random_idx], game.currentBoard())) << std::endl;
                        game.makeMove(legal[random_idx]);
                    }
                }
                // record result
                int outcome = *game.winner();  // 1=White,0=Black,-1=draw
                if (color == 1) {  // we played White
                    if (outcome == -1)      w_draw++;
                    else if (outcome == 1)  w_win++;
                    else                    w_loss++;
                } else {          // we played Black
                    if (outcome == -1)      b_draw++;
                    else if (outcome == 0)  b_win++;
                    else                    b_loss++;
                }
            }
        }
    };

    // 3) launch threads
    int per = (num_games_per_color + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, per);
    }
    for (auto &th : threads) th.join();

    // 4) return aggregated results
    return { w_win.load(), w_loss.load(), w_draw.load(),
             b_win.load(), b_loss.load(), b_draw.load() };
}
// --------------------------------------------------------------------
// Number of floats per record: state (8x8x119) + policy (8x8x73) + z
static constexpr uint64_t RECORD_FLOATS = 8ULL * 8 * 119 + 8ULL * 8 * 73 + 1;
static constexpr uint64_t RECORD_BYTES = RECORD_FLOATS * sizeof(float);
static constexpr uint64_t HEADER_BYTES = 24; // [capacity(8), size(8), head(8)]

// Helpers to read/write 64-bit header fields
static int64_t read_i64(std::fstream &f, std::streamoff off) {
    int64_t v;
    f.seekg(off);
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

static void write_i64(std::fstream &f, std::streamoff off, int64_t v) {
    f.seekp(off);
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    f.flush();
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
    std::cerr << "[AZ] simulate_games_buffered: model='" << model_path << "' -> buffer='" << filename << "' cap=" << capacity << "\n";

    // 1) Load model
    BatchManager::instance().init(model_path, 16);
    py::gil_scoped_release no_gil;

    // 2) Open (or create) ring-buffer file
    std::fstream f(filename, std::ios::in | std::ios::out | std::ios::binary);
    if (!f) {
        std::cerr << "[AZ] creating new buffer file\n";
        std::ofstream of(filename, std::ios::binary | std::ios::trunc);
        int64_t zero = 0;
        of.write(reinterpret_cast<const char*>(&capacity), sizeof(capacity));
        of.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
        of.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
        of.close();
        std::filesystem::resize_file(filename, HEADER_BYTES + capacity * RECORD_BYTES);
        f.open(filename, std::ios::in | std::ios::out | std::ios::binary);
    }
    std::cerr << "[AZ] buffer file opened\n";

    // 3) Read header
    int64_t size = read_i64(f, 8);
    int64_t head = read_i64(f, 16);
    std::cerr << "[AZ] initial size=" << size << " head=" << head << "\n";

    std::mutex file_mtx;
    auto append_one = [&](const std::vector<float> &state, const std::vector<float> &policy, float z) {
        std::lock_guard lk(file_mtx);
        uint64_t idx = head % capacity;
        uint64_t offset = HEADER_BYTES + idx * RECORD_BYTES;
        std::cerr << "[AZ] append idx=" << idx << " offset=" << offset << " z=" << z << "\n";
        f.seekp(offset);
        f.write(reinterpret_cast<const char*>(state.data()), state.size() * sizeof(float));
        f.write(reinterpret_cast<const char*>(policy.data()), policy.size() * sizeof(float));
        f.write(reinterpret_cast<const char*>(&z), sizeof(z));
        f.flush();
        head = (head + 1) % capacity;
        if (size < capacity) ++size;
        write_i64(f, 8, size);
        write_i64(f, 16, head);
        std::cerr << "[AZ] new size=" << size << " new head=" << head << "\n";
    };

    // --- dynamic, perfectly-balanced scheduling --------------------
    std::atomic<int> game_idx{0};
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            // each thread gets its own RNG
            std::mt19937_64 rng{std::random_device{}()};

            // grab one game at a time
            while (true) {
                int idx = game_idx.fetch_add(1, std::memory_order_relaxed);
                if (idx >= num_games)
                    break;

                // ----- simulate exactly one game -----
                // (copy the body of 'worker' here, but for a single game)
                std::vector<std::vector<float>> states;
                std::vector<std::vector<float>> policies;
                std::vector<float> toplays;

                ChessGame game;
                MCTArgs args{ num_simulations, alpha, epsilon, sampling_moves };
                while (!game.winner().has_value()) {
                    auto tensor = game.encodeTensor();
                    std::vector<float> flat(tensor.begin(), tensor.end());
                    auto policy = run_mcts(game, args);
                    auto legal = game.legalMoves();

                    std::vector<float> masked(ACTION_SPACE, 0.0f);
                    for (auto mv : legal) masked[mv] = policy[mv];
                    float sum = std::accumulate(masked.begin(), masked.end(), 0.0f);

                    int action;
                    if (sum == 0.0f) {
                        std::uniform_int_distribution<size_t> u(0, legal.size() - 1);
                        action = legal[u(rng)];
                    } else if ((int)states.size() >= sampling_moves) {
                        action = int(std::distance(masked.begin(),
                                                   std::max_element(masked.begin(), masked.end())));
                    } else {
                        std::discrete_distribution<int> d(masked.begin(), masked.end());
                        action = d(rng);
                    }

                    states.push_back(std::move(flat));
                    policies.push_back(std::move(policy));
                    toplays.push_back(float(game.to_play()));
                    game.makeMove(uint16_t(action));
                }

                int w = *game.winner();
                for (size_t k = 0; k < states.size(); ++k) {
                    float z = (w == -1 ? 0.0f
                                : toplays[k] == float(w) ? 1.0f
                                                         : -1.0f);
                    append_one(states[k], policies[k], z);
                }
                // ----------------------------------------
            }
        });
    }
    for (auto &th : threads) th.join();
    f.close();
    std::cerr << "[AZ] simulate_games_buffered DONE\n";
}

} // namespace az73

PYBIND11_MODULE(chess_engine, m) {
    m.doc() = "C++ self-play with batched inference and on-disk ring buffer";
    m.def("simulate_games_buffered", &az73::simulate_games_buffered,
        py::arg("model_path"),
        py::arg("num_games"),
        py::arg("num_threads"),
        py::arg("num_simulations"),
        py::arg("alpha"),
        py::arg("epsilon"),
        py::arg("sampling_moves"),
        py::arg("filename"),
        py::arg("replay_buffer_capacity"));
    
    m.def("evaluate_vs_random", &az73::evaluate_vs_random,
          py::arg("model_path"),
          py::arg("games_per_color"),
          py::arg("num_threads"),
          py::arg("num_simulations"),
          py::arg("alpha"),
          py::arg("epsilon"),
          py::arg("sampling_moves"),
          "Evaluate the AlphaZero agent vs a random agent, returning "
          "(white_wins, white_losses, white_draws, black_wins, black_losses, black_draws).");
}
