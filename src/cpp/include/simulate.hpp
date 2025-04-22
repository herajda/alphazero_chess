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

} // namespace az73
