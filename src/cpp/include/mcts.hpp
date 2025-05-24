#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <random>
#include "chess_game.hpp"

namespace az73 {

// Total number of possible actions: 8x8x73
constexpr int ACTION_SPACE = 8 * 8 * 73;

struct MCTArgs {
    int num_simulations;
    double alpha;
    double epsilon;
    int sampling_moves;
};

class MCTNode {
public:
    MCTNode(float prior, const ChessGame& game);
    [[nodiscard]] float value() const;
    [[nodiscard]] bool is_expanded() const;
    void expand();
    void add_exploration_noise(double epsilon, double alpha);
    [[nodiscard]] std::pair<uint16_t, MCTNode*> select_child();
    void update(float v);
    const ChessGame& game() const noexcept { return game_; }
    /* ---------- asynchronous helpers ---------- */
    bool pending()   const noexcept { return pending_; }
    void mark_pending() noexcept    { pending_ = true; }
    /** finish expansion when the network output arrives */
    void complete_expand(const std::vector<float>& policy, float v);
    [[nodiscard]] const std::unordered_map<uint16_t, std::unique_ptr<MCTNode>>& children() const;
    [[nodiscard]] int visit_count() const;
    [[nodiscard]] float total_value() const;

    ChessGame game_;

private:
    float prior_;
    int visit_count_;
    float total_value_;
    std::unordered_map<uint16_t, std::unique_ptr<MCTNode>> children_;
    static thread_local std::mt19937_64 rng_;
    /* ---- helper that turns raw π into safe child priors ---- */
    static std::vector<float>
    sanitise_priors(const std::vector<float>& policy,
                    const std::vector<uint16_t>& legal);
    bool pending_{false};
};

// Run MCTS from `root_game` under parameters in `args`, return policy vector [4672]
std::vector<float> run_mcts(const ChessGame& root_game, const MCTArgs& args);

} // namespace az73
