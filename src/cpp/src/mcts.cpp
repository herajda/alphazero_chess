#include "mcts.hpp"
#include "batch_manager.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

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
    // Terminal?
    auto win = game_.winner();
    if (win.has_value()) {
        children_.clear();
        float v;
        int w = win.value();
        if (w == -1) v = 0.0f;
        else if (w == game_.to_play()) v = 1.0f;
        else v = -1.0f;
        visit_count_ = 1;
        total_value_  = v;
        return;
    }
    // Non-terminal: ask network
    auto tensor = game_.encodeTensor();
    std::vector<float> flat(tensor.begin(), tensor.end());
    auto fut = BatchManager::instance().enqueue(flat);
    auto [policy, v] = fut.get();

    // Expand children
    auto legal = game_.legalMoves();
    children_.clear();
    for (uint16_t a : legal) {
        ChessGame next = game_;
        next.makeMove(a);
        children_[a] = std::make_unique<MCTNode>(policy[a], next);
    }
    visit_count_ = 1;
    total_value_  = v;
}

void MCTNode::add_exploration_noise(double epsilon, double alpha) {
    size_t K = children_.size();
    if (K == 0) return;
    // sample Dirichlet(K, alpha)
    std::gamma_distribution<double> gamma(alpha, 1.0);
    std::vector<double> noise(K);
    double sum = 0;
    for (size_t i = 0; i < K; ++i) {
        noise[i] = gamma(rng_);
        sum += noise[i];
    }
    if (sum <= 0) sum = 1;
    for (auto &n : noise) n /= sum;
    
    // apply to priors
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

std::vector<float> run_mcts(const ChessGame& root_game, const MCTArgs& args) {
    MCTNode root(1.0f, root_game);
    root.expand();
    root.add_exploration_noise(args.epsilon, args.alpha);

    std::vector<MCTNode*> path;
    for (int i = 0; i < args.num_simulations; ++i) {
        MCTNode* node = &root;
        path.clear();
        while (node->is_expanded() && !node->children().empty()) {
            auto [a, next] = node->select_child();
            path.push_back(node);
            node = next;
        }
        if (!node->is_expanded()) node->expand();
        float val = node->value();
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            (*it)->update(val);
            val = -val;
        }
    }
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
