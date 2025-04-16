#ifndef MCTS_H
#define MCTS_H

#include "chess_game.h"
#include "batch_manager.h"
#include <unordered_map>
#include <memory>

class MCTNode {
public:
    float prior;
    std::unique_ptr<ChessGame> game;
    std::unordered_map<int, std::unique_ptr<MCTNode>> children;
    int visit_count = 0;
    float total_value = 0.0f;

    MCTNode(float p) : prior(p) {}
    float value() const { return visit_count > 0 ? total_value / visit_count : 0.0f; }
    bool is_evaluated() const { return visit_count > 0; }
    void evaluate(BatchManager& batch_mgr);
    std::pair<int, MCTNode*> select_child() const;
};

std::vector<float> mcts(ChessGame& game, BatchManager& batch_mgr, int num_simulations, bool explore, float epsilon, float alpha);

#endif
