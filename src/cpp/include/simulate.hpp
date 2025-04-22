#pragma once

#include <pybind11/pybind11.h>
#include <vector>
#include <tuple>

namespace py = pybind11;

using Trajectory = std::vector<std::tuple<std::vector<float>, std::vector<float>, float>>;
using AllGames   = std::vector<Trajectory>;

// Simulate num_games self‑play games, append each finished trajectory to output_file in binary,
// and do *not* retain them in RAM.
void simulate_games_dump(py::object agent_py,
                         int num_games,
                         int num_threads,
                         int num_simulations,
                         double alpha,
                         double epsilon,
                         int sampling_moves,
                         const std::string &output_file);
