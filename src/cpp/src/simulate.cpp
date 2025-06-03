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
#include <csignal> // <-- add this

namespace az73 {

// Global atomic flag for interruption
std::atomic<bool> interrupted{false};

// Signal handler
void handle_sigint(int) {
    interrupted = true;
    std::cerr << "[AZ] Caught SIGINT, stopping simulation..." << std::endl;
}
// --------------------------------------------------------------------
// Evaluate vs STOCKFISH (UCI) -- threaded & batched
// --------------------------------------------------------------------
namespace {

// very small helper: spawn Stockfish, return a pair of i/o streams
struct UciEngine {
    std::string bin;
    FILE *in  = nullptr;   // write TO engine
    FILE *out = nullptr;   // read  FROM engine

    explicit UciEngine(const std::string &path) : bin(path) {
        int in_pipe [2], out_pipe[2];
        pipe(in_pipe);  pipe(out_pipe);

        pid_t pid = fork();
        if (pid == 0) {               // child → Stockfish
            dup2(in_pipe [0], STDIN_FILENO);
            dup2(out_pipe[1], STDOUT_FILENO);
            close(in_pipe [1]); close(out_pipe[0]);
            execl(bin.c_str(), bin.c_str(), nullptr);
            _exit(127);
        }
        // parent
        close(in_pipe [0]);  close(out_pipe[1]);
        in  = fdopen(in_pipe [1],  "w");
        out = fdopen(out_pipe[0],  "r");

        // initialise UCI
        fprintf(in, "uci\n");  fflush(in);
        wait_ready();
    }
    ~UciEngine() { if (in) { fputs("quit\n", in); fflush(in);} }

    void send(const std::string &cmd) { fputs(cmd.c_str(), in); fputc('\n', in); fflush(in);}
    std::string readline() {
        char buf[256];
        if (!fgets(buf, sizeof(buf), out)) return {};
        return std::string(buf);
    }
    void wait_ready() {
        send("isready");
        std::string s;
        while ((s = readline()).find("readyok") == std::string::npos) {}
    }
    void set_options(int elo, int threads) {
        if (elo >= 0) {
            send("setoption name UCI_LimitStrength value true");
            send("setoption name UCI_Elo          value " + std::to_string(elo));
        }
        send("setoption name Threads value " + std::to_string(threads));
        wait_ready();
    }

    // get bestmove in UCI for the current position
    std::string bestmove(const std::string &fen,
                         int depth, bool use_depth) {
        send("position fen " + fen);
        if (use_depth) send("go depth " + std::to_string(depth));
        else           send("go movetime 100");          // fallback
        std::string s;
        while ((s = readline()).rfind("bestmove ",0)) {}  // wait
        auto mv   = s.substr(9);           // drop "bestmove "
        mv.erase(mv.find_first_of(" \t\r\n")); // trim right
        return mv;
    }
}; // struct UciEngine
}  // anonymous namespace

    // helper: returns a *legal* chess::Move that corresponds to action_id
    static chess::Move id_to_legal_move(const chess::Board &board,
                                        uint16_t action_id)
    {
        using namespace chess;
        Movelist legal;
        movegen::legalmoves<movegen::MoveGenType::ALL>(legal, board,
                PieceGenType::PAWN   | PieceGenType::KNIGHT | PieceGenType::BISHOP |
                PieceGenType::ROOK   | PieceGenType::QUEEN  | PieceGenType::KING);
        
        for (const Move &m : legal)           // round-trip test
            if (encode(board, m) == action_id)
                return m;                     // found exact match
        
        // *** should never happen, but be defensive ***
        return legal.empty() ? Move::NULL_MOVE : legal[0];
    }
// public API
std::tuple<int,int,int,int,int,int>
evaluate_vs_stockfish(
    const std::string &model_path,
    const std::string &sf_bin,
    int games_per_color,
    int num_threads,
    int sf_depth,
    int sf_elo,
    int num_simulations,
    double alpha,
    double epsilon,
    int sampling_moves)
{
    using namespace chess;
    constexpr std::size_t GPU_BATCH = 256;
    BatchManager::instance().init(model_path, GPU_BATCH);

    std::atomic<int> wW{0}, wL{0}, wD{0}, bW{0}, bL{0}, bD{0};
    std::atomic<int> gidx{0};

    auto worker = [&](int tid) {
        UciEngine sf(sf_bin);
        sf.set_options(sf_elo, 1);   // 1 thread/engine

        while (true) {
            int idx = gidx.fetch_add(1);
            if (idx >= 2 * games_per_color) break;

            bool agent_is_white = (idx < games_per_color);
            int  us_turn_code  = agent_is_white ? 1 : 0;

            ChessGame game;
            MCTArgs args{num_simulations, alpha, epsilon, sampling_moves};

            while (!game.winner().has_value()) {
                if (game.to_play() == us_turn_code) {
                    auto policy = run_mcts(game, args);
                    auto legal  = game.legalMoves();
                    uint16_t best = *std::max_element(
                        legal.begin(), legal.end(),
                        [&](uint16_t a, uint16_t b){return policy[a] < policy[b];});
                    game.makeMove(best);

                } else {
                    std::string fen   = game.currentBoard().getFen();

                    std::string uci   = sf.bestmove(fen, sf_depth, sf_elo < 0);
                    chess::Move  mv   = chess::uci::uciToMove(game.currentBoard(), uci);
                    uint16_t a = encode(game.currentBoard(), mv);

                    game.makeMove(a);
                }
            }
            int res = *game.winner();       // 1 white win, 0 black win, -1 draw
            if (agent_is_white) {
                if (res ==  1) wW++; else
                if (res ==  0) wL++; else wD++;
            } else {
                if (res ==  0) bW++; else
                if (res ==  1) bL++; else bD++;
            }
        }
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < num_threads; ++t) pool.emplace_back(worker, t);
    for (auto &th : pool) th.join();

    return {wW.load(), wL.load(), wD.load(),
            bW.load(), bL.load(), bD.load()};
}


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
    constexpr std::size_t GPU_BATCH = 256;
    BatchManager::instance().init(model_path, GPU_BATCH);
    py::gil_scoped_release no_gil;

    // 2) counters
    std::atomic<int> w_win{0}, w_loss{0}, w_draw{0};
    std::atomic<int> b_win{0}, b_loss{0}, b_draw{0};

    // 3) shared atomic index for “which game to play next”
    //    we want total = 2 * num_games_per_color (first half = White‐side games, next half = Black‐side)
    std::atomic<int> game_idx{0};

    auto worker = [&](int tid) {
        std::mt19937_64 rng{std::random_device{}()};

        while (true) {
            int idx = game_idx.fetch_add(1, std::memory_order_relaxed);
            if (idx >= 2 * num_games_per_color)
                break;

            // determine color from idx:
            //   idx in [0 .. num_games_per_color-1]  → play as White
            //   idx in [num_games_per_color .. 2*num_games_per_color-1] → play as Black
            int color = (idx < num_games_per_color ? 1 : 0);
            ChessGame game;
            MCTArgs args{num_simulations, alpha, epsilon, sampling_moves};

            // play until game over
            while (!game.winner().has_value()) {
                if (game.to_play() == color) {
                    // our agent turn
                    auto policy = run_mcts(game, args);
                    auto legal = game.legalMoves();
                    uint16_t best = legal.front();
                    float best_p = policy[best];
                    for (auto a : legal) {
                        if (policy[a] > best_p) {
                            best_p = policy[a];
                            best = a;
                        }
                    }
                    game.makeMove(best);

                } else {
                    // random‐move baseline
                    auto legal = game.legalMoves();
                    std::uniform_int_distribution<size_t> uni(0, legal.size() - 1);
                    size_t random_idx = uni(rng);
                    game.makeMove(legal[random_idx]);
                }
            }

            int outcome = *game.winner();  // 1=White, 0=Black, –1=draw
            if (color == 1) {
                // we played White
                if      (outcome == -1) w_draw++;
                else if (outcome ==  1) w_win++;
                else                    w_loss++;
            } else {
                // we played Black
                if      (outcome == -1) b_draw++;
                else if (outcome ==  0) b_win++;
                else                    b_loss++;
            }
        }
    };

    // 4) launch exactly num_threads threads
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto &th : threads) th.join();

    // 5) return aggregated counts
    return {
        w_win.load(),  w_loss.load(),  w_draw.load(),
        b_win.load(),  b_loss.load(),  b_draw.load()
    };
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
    // Install signal handler (only once, safe for repeated calls)
    static std::once_flag sig_flag;
    std::call_once(sig_flag, []() {
        std::signal(SIGINT, handle_sigint);
    });



    // 1) Load model
    BatchManager::instance().init(model_path, 64);

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
                // Check for interruption before starting a new game
                if (interrupted) break;

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
                    // Check for interruption inside game loop
                    if (interrupted) break;
                    auto raw = game.encodeTensor();

                    std::vector<float> flat;
                    flat.reserve(raw.size());
                    for (uint8_t b : raw)
                        flat.push_back(static_cast<float>(b));      // 0.f / 1.f

                    auto policy = run_mcts(game, args);
                    //std::cout << "[AZ] policy size = " << policy.size() << std::endl;
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

                    // Print FEN and action in UCI format
                    //std::cout << "[AZ] size " << size << " FEN: " << game.currentBoard().getFen() << std::endl;
                    //std::cout << "[AZ] Action: " 
                    //          << chess::uci::moveToUci(az73::decode_action(action, game.currentBoard()))
                    //          << std::endl;

                    states.push_back(std::move(flat));
                    policies.push_back(std::move(policy));
                    toplays.push_back(float(game.to_play()));
                    game.makeMove(uint16_t(action));
                }

                // If interrupted, don't write partial games
                if (interrupted) break;

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

    // If interrupted, raise Python KeyboardInterrupt
    if (interrupted) {
        PyErr_SetInterrupt(); // sets the Python interrupt flag
        throw py::error_already_set();
    }
}

} // namespace az73
namespace py = pybind11;
using az73::BatchManager;
using az73::MCTArgs;
using az73::run_mcts;

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
    m.def("select_move",
        // Lambda takes: path to TorchScript, FEN, MCTS params → returns best flat action
        [](const std::string &model_path,
           const std::string &fen,
           int num_simulations,
           double alpha,
           double epsilon,
           int sampling_moves) -> uint16_t
        {
            // 1) Initialize the batched model (use a reasonable batch size)
            constexpr size_t BATCH = 1;
            BatchManager::instance().init(model_path, BATCH);

            // 2) Build game from FEN
            ChessGame game(fen);

            // 3) Run MCTS
            MCTArgs args{num_simulations, alpha, epsilon, sampling_moves};
            auto policy = run_mcts(game, args);

            // 4) Pick the highest-probability legal move
            auto legal = game.legalMoves();
            uint16_t best = legal.front();
            float best_p = policy[best];
            for (auto a : legal) {
                if (policy[a] > best_p) {
                    best_p = policy[a];
                    best   = a;
                }
            }
            return best;
        },
        py::arg("model_path"),
        py::arg("fen"),
        py::arg("num_simulations"),
        py::arg("alpha"),
        py::arg("epsilon"),
        py::arg("sampling_moves"),
        R"pbdoc(
            select_move(model_path, fen, num_simulations, alpha, epsilon, sampling_moves) -> action_index

            Run MCTS from the given FEN and return the chosen move (flattened 0–4671).
        )pbdoc"
    ); 
    m.def("evaluate_vs_stockfish", &az73::evaluate_vs_stockfish,
      py::arg("model_path"),
      py::arg("sf_bin"),
      py::arg("games_per_color"),
      py::arg("num_threads"),
      py::arg("sf_depth"),
      py::arg("sf_elo"),
      py::arg("num_simulations"),
      py::arg("alpha"),
      py::arg("epsilon"),
      py::arg("sampling_moves"),
      "Evaluate the AlphaZero agent against Stockfish and return "
      "(wW,wL,wD,bW,bL,bD).");
}
