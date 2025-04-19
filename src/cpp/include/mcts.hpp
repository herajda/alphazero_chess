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
    float value() const;
    bool is_expanded() const;
    void expand();
    void add_exploration_noise(double epsilon, double alpha);
    std::pair<uint16_t, MCTNode*> select_child();
    void update(float v);
    const std::unordered_map<uint16_t, std::unique_ptr<MCTNode>>& children() const;
    int visit_count() const;

private:
    float prior_;
    int visit_count_;
    float total_value_;
    ChessGame game_;
    std::unordered_map<uint16_t, std::unique_ptr<MCTNode>> children_;
    static thread_local std::mt19937_64 rng_;
};

// Run MCTS from `root_game` under parameters in `args`, return policy vector [4672]
std::vector<float> run_mcts(const ChessGame& root_game, const MCTArgs& args);

} // namespace az73
