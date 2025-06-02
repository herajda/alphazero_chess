// src/cpp/include/simulate.hpp
#pragma once

#include <pybind11/pybind11.h>
#include <vector>
#include <tuple>
#include <string>
#include <cstdint>

namespace py = pybind11;

// A single game trajectory: sequence of (state, policy, z)
using Trajectory = std::vector<std::tuple<std::vector<float>, std::vector<float>, float>>;

namespace az73 {

/**
 * Buffered self‑play: run `num_games` games, and for each move immediately
 * append (state, policy, z) into a fixed‑size on‑disk ring buffer.
 */
void simulate_games_buffered(
    const std::string &model_path,
    int num_games,
    int num_threads,
    int num_simulations,
    double alpha,
    double epsilon,
    int sampling_moves,
    const std::string &filename,
    int64_t replay_buffer_capacity
);
// add just below evaluate_vs_random declaration
std::tuple<int,int,int,int,int,int>
evaluate_vs_stockfish(
    const std::string &model_path,     // TorchScript network
    const std::string &sf_bin,         // path to Stockfish executable
    int games_per_color,               // #games as White and as Black
    int num_threads,                   // evaluation threads
    int sf_depth,                      // “go depth” (ignored if elo≥0)
    int sf_elo,                        // set to ≥0 to use UCI_Elo mode
    int num_simulations,               // our MCTS sims / move
    double alpha, double epsilon,
    int sampling_moves
);


} // namespace az73
