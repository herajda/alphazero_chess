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
#include <fstream>

namespace az73 {
void simulate_games_dump(const std::string& model_path,
                        int num_games,
                        int num_threads,
                        int num_simulations,
                        double alpha,
                        double epsilon,
                        int sampling_moves,
			const std::string &output_file) {
    // Initialize the BatchManager with a TorchScript model
    BatchManager::instance().init(model_path, num_threads);
    py::gil_scoped_release no_gil;
    std::ofstream out(output_file, std::ios::binary | std::ios::app);
    std::mutex file_mtx;
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

                std::cout << "action " << chess::uci::moveToUci(az73::decode_action(action, game.currentBoard())) << " game: " << game.currentBoard().getFen() << std::endl;
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

	    // **dump to disk** and free memory
            {
                std::lock_guard<std::mutex> lk(file_mtx);
                int32_t steps = static_cast<int32_t>(traj.size());
                out.write(reinterpret_cast<const char*>(&steps), sizeof(steps));

                for (auto &e : traj) {
                    auto &state  = std::get<0>(e);
                    auto &policy = std::get<1>(e);
                    float  z      = std::get<2>(e);

                    // 1) write flat state (8*8*119 floats)
                    out.write(reinterpret_cast<const char*>(state.data()),
                              state.size() * sizeof(float));

                    // 2) write policy length + data
                    int32_t p_len = static_cast<int32_t>(policy.size());
                    out.write(reinterpret_cast<const char*>(&p_len), sizeof(p_len));
                    out.write(reinterpret_cast<const char*>(policy.data()),
                              p_len * sizeof(float));

                    // 3) write z
                    out.write(reinterpret_cast<const char*>(&z), sizeof(z));
                }
            }
            // drop traj out of RAM
            std::vector<std::tuple<std::vector<float>,std::vector<float>,float>>().swap(traj);
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
    out.close();
}

} // namespace az73
PYBIND11_MODULE(chess_engine, m) {
    m.doc() = "C++ self-play simulator with batched TorchScript inference";
    m.def("simulate_games_dump", &az73::simulate_games_dump,
          py::arg("model_path"), py::arg("num_games"),
          py::arg("num_threads"), py::arg("num_simulations"),
          py::arg("alpha"), py::arg("epsilon"), py::arg("sampling_moves"),
          py::arg("output_file"));
}
