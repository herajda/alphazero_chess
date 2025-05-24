#include "mcts.hpp"
#include "batch_manager.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace az73 {

// initialize thread-local RNG
thread_local std::mt19937_64 MCTNode::rng_{std::random_device{}()};

MCTNode::MCTNode(float prior, const ChessGame& game)
    : prior_(prior), visit_count_(0), total_value_(0.0f), game_(game) {}

float MCTNode::value() const {
    return visit_count_ == 0 ? 0.0f : total_value_ / visit_count_;
}

bool MCTNode::is_expanded() const {
    return visit_count_ > 0;
}

void MCTNode::expand() {
    // --- 1) Check for terminal parent, unchanged from yours ---
    if (auto win = game_.winner()) {
        children_.clear();
        float v = (win.value() == -1 ? 0.0f
                   : win.value() == game_.to_play() ? -1.0f
                   : 1.0f);
        visit_count_ = 1;
        total_value_ = v;
        return;
    }

    // --- 2) Get net policy + value ---
    auto tensor = game_.encodeTensor();
    std::vector<float> flat(tensor.begin(), tensor.end());
    auto [policy, v] = BatchManager::instance()
                          .enqueue(flat).get();

    //// --- 3) Build children & collect terminals ---
    auto legal = game_.legalMoves();
    //std::vector<uint16_t> win_moves;
    //// stash the raw network priors so we can rescale later
    std::unordered_map<uint16_t, float> orig_prior;
    for (auto a : legal) {
        ChessGame next = game_;
        next.makeMove(a);

        // store original
        orig_prior[a] = policy[a];
        children_[a] = std::make_unique<MCTNode>(policy[a], next);

        //// detect terminal children
        //if (auto w = next.winner()) {
        //    if (w.value() != game_.to_play())
        //        win_moves.push_back(a);
        //    // draws (w == -1) are ignored here
        //}
    }

    //// --- 4) Override the priors if we saw any wins or losses ---
    //if (!win_moves.empty()) {
    //    // Case A: we have winning moves → uniform over those
    //    float p = 1.0f / win_moves.size();
    //    for (auto &kv : children_) kv.second->prior_ = 0.0f;
    //    for (auto a : win_moves) children_[a]->prior_ = p;
    //}
    // else: no wins *and* no losses → leave net priors untouched

    // --- 5) Finish usual expand bookkeeping ---
    visit_count_ = 1;
    total_value_ = v;
}


void MCTNode::add_exploration_noise(double epsilon, double alpha) {
    size_t K = children_.size();
    if (K == 0) return;
    std::gamma_distribution<double> gamma(alpha, 1.0);
    std::vector<double> noise(K);
    double sum = 0;
    for (size_t i = 0; i < K; ++i) {
        noise[i] = gamma(rng_);
        sum += noise[i];
    }
    if (sum <= 0) sum = 1;
    for (auto &n : noise) n /= sum;
    std::vector<double> new_p;
    new_p.reserve(K);
    size_t idx = 0;
    for (auto &kv : children_) {
        double p = (1 - epsilon) * kv.second->prior_ + epsilon * noise[idx++];
        new_p.push_back(p);
    }
    double psum = std::accumulate(new_p.begin(), new_p.end(), 0.0);
    if (psum <= 0) psum = K;
    idx = 0;
    for (auto &kv : children_) {
        kv.second->prior_ = new_p[idx++] / psum;
    }
}

std::pair<uint16_t, MCTNode*> MCTNode::select_child() {
    double best = -1e9;
    uint16_t best_a = 0;
    MCTNode* best_n = nullptr;
    double N = visit_count_;
    double C = std::log((1 + N + 1965.2) / 1965.2) + 1.25;
    for (auto &kv : children_) {
        auto *c = kv.second.get();
        double Q = c->value();
        double P = c->prior_;
        double Nsa = c->visit_count_;
        double u = Q + C * P * std::sqrt(N) / (Nsa + 1);
        if (u > best) {
            best = u;
            best_a = kv.first;
            best_n = c;
        }
    }
    return {best_a, best_n};
}

void MCTNode::update(float v) {
    visit_count_++;
    total_value_ += v;
}

const std::unordered_map<uint16_t, std::unique_ptr<MCTNode>>& MCTNode::children() const {
    return children_;
}

int MCTNode::visit_count() const {
    return visit_count_;
}
float MCTNode::total_value() const {
    return total_value_;
}

std::vector<float> run_mcts(const ChessGame& root_game, const MCTArgs& args) {
    MCTNode root(1.0f, root_game);
    root.expand();
    root.add_exploration_noise(args.epsilon, args.alpha);
    std::deque<MCTNode*> path;
    for (int i = 0; i < args.num_simulations; ++i) {
        MCTNode* node = &root;
        path.clear();
        while (node->is_expanded() && !node->children().empty()) {
            auto [a, next] = node->select_child();
            path.push_back(next);
            node = next;
        }
        float val;
        if (!node->is_expanded()) {
            node->expand();
            path.pop_back();
            val = -node->value();

        }
        else {
            val = node->value();
        }
        
        while (!path.empty()) {
            MCTNode* it = path.back();
            it->update(val);
            val = -val;
            path.pop_back();
        }
    }
    //std::cout << "--- MCTS root children stats ---\n";
    //std::cout << "FEN: " << root.game_.currentBoard().getFen() << std::endl;
    //for (auto &kv : root.children()) {
    //    uint16_t action = kv.first;
    //    MCTNode* child = kv.second.get();
    //    std::cout
    //        << "Action " << action << " UCI: " << chess::uci::moveToUci(az73::decode_action(action, root.game_.currentBoard()))
    //        << " | total_value = " << child->total_value()
    //        << " | visits = "      << child->visit_count()
    //        << "\n";
    //}
    //std::cout << "--------------------------------\n";
    std::vector<float> policy(ACTION_SPACE, 0.0f);
    float tot = 0;
    for (auto &kv : root.children()) tot += kv.second->visit_count();
    if (tot > 0) {
        for (auto &kv : root.children()) {
            policy[kv.first] = kv.second->visit_count() / tot;
        }
    }
    return policy;
}

} // namespace az73
