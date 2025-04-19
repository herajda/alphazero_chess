#pragma once

#include <pybind11/pybind11.h>
#include <vector>
#include <tuple>

namespace py = pybind11;

using Trajectory = std::vector<std::tuple<std::vector<float>, std::vector<float>, float>>;
using AllGames   = std::vector<Trajectory>;

// Exposed entry-point: simulate num_games self-play games in C++, return all trajectories
AllGames simulate_games(py::object agent_py,
                        int num_games,
                        int num_threads,
                        int num_simulations,
                        double alpha,
                        double epsilon,
                        int sampling_moves);

