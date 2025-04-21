#include "simulate.hpp"
#include "mcts.hpp"
#include "chess.hpp"
#include "chess_game.hpp"
#include "batch_manager.hpp"
#include <pybind11/stl.h>
#include <thread>
#include <mutex>
#include <algorithm>
#include <random>

namespace az73 {

AllGames simulate_games(const std::string& model_path,
                        int num_games,
                        int num_threads,
                        int num_simulations,
                        double alpha,
                        double epsilon,
                        int sampling_moves) {
    // Initialize the BatchManager with a TorchScript model
    BatchManager::instance().init(model_path, num_threads);
    py::gil_scoped_release no_gil;
    AllGames all_games;
    all_games.reserve(num_games);
    std::mutex mtx;

    auto worker = [&](int count) {
        thread_local std::mt19937_64 rng(std::random_device{}());
        for (int i = 0; i < count; ++i) {
            ChessGame game;
            Trajectory traj;
            MCTArgs args{num_simulations, alpha, epsilon, sampling_moves};

            // Play until game end
            while (true) {
                auto win = game.winner();
                if (win.has_value()) break;

                // Encode and MCTS
                auto tensor = game.encodeTensor();
                std::vector<float> flat(tensor.begin(), tensor.end());
                auto policy = run_mcts(game, args);

                // Select action (explore or exploit)
                int action;
                if ((int)traj.size() >= sampling_moves) {
                    action = std::distance(policy.begin(),
                        std::max_element(policy.begin(), policy.end()));
                } else {
                    std::discrete_distribution<int> dist(policy.begin(), policy.end());
                    action = dist(rng);
                }

                //std::cout << "action " << chess::uci::moveToUci(az73::decode_action(action, game.currentBoard())) << " game: " << game.currentBoard().getFen() << std::endl;
                traj.emplace_back(flat, policy, (float)game.to_play());
                game.makeMove((uint16_t)action);
            }

            // Compute outcome z for each state
            int w = *game.winner();
            for (auto& e : traj) {
                float to_play = std::get<2>(e);
                float z;
                if (w == -1) z = 0.0f;
                else z = (to_play == (float)w ? 1.0f : -1.0f);
                std::get<2>(e) = z;
            }

            std::lock_guard<std::mutex> lk(mtx);
            all_games.push_back(std::move(traj));
        }
    };

    int per = (num_games + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    int started = 0;
    for (int t = 0; t < num_threads && started < num_games; ++t) {
        int count = std::min(per, num_games - started);
        threads.emplace_back(worker, count);
        started += count;
    }
    for (auto& th : threads) th.join();
    return all_games;
}

} // namespace az73
PYBIND11_MODULE(chess_engine, m) {
    m.doc() = "C++ self-play simulator with batched TorchScript inference";
    m.def("simulate_games", &az73::simulate_games,
          py::arg("model_path"), py::arg("num_games"),
          py::arg("num_threads"), py::arg("num_simulations"),
          py::arg("alpha"), py::arg("epsilon"), py::arg("sampling_moves"));
}
