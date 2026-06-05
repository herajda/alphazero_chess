#include "mcts.hpp"
#include "batch_manager.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <limits>

namespace az73 {

float terminal_value_for_side_to_move(int winner, int to_play) {
    if (winner == -1) {
        return 0.0f;
    }
    return winner == to_play ? 1.0f : -1.0f;
}

std::vector<float> normalise_legal_policy(const std::vector<float> &policy,
                                          const std::vector<uint16_t> &legal) {
    std::vector<float> priors(legal.size(), 0.0f);
    if (legal.empty()) {
        return priors;
    }

    bool looks_like_probs = !policy.empty();
    double full_sum = 0.0;
    for (float p : policy) {
        if (!std::isfinite(p) || p < 0.0f) {
            looks_like_probs = false;
            break;
        }
        full_sum += p;
    }
    looks_like_probs = looks_like_probs && std::abs(full_sum - 1.0) < 1e-3;

    if (looks_like_probs) {
        double legal_sum = 0.0;
        for (std::size_t i = 0; i < legal.size(); ++i) {
            const uint16_t a = legal[i];
            if (a < policy.size()) {
                priors[i] = policy[a];
                legal_sum += priors[i];
            }
        }
        if (legal_sum > 0.0) {
            for (float &p : priors) {
                p = static_cast<float>(p / legal_sum);
            }
            return priors;
        }
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    for (uint16_t a : legal) {
        if (a < policy.size() && std::isfinite(policy[a])) {
            max_logit = std::max(max_logit, policy[a]);
        }
    }
    if (!std::isfinite(max_logit)) {
        std::fill(priors.begin(), priors.end(), 1.0f / static_cast<float>(legal.size()));
        return priors;
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < legal.size(); ++i) {
        const uint16_t a = legal[i];
        if (a < policy.size() && std::isfinite(policy[a])) {
            priors[i] = std::exp(policy[a] - max_logit);
            sum += priors[i];
        }
    }
    if (sum <= 0.0 || !std::isfinite(sum)) {
        std::fill(priors.begin(), priors.end(), 1.0f / static_cast<float>(legal.size()));
        return priors;
    }
    for (float &p : priors) {
        p = static_cast<float>(p / sum);
    }
    return priors;
}

double puct_score_from_parent(float child_value,
                              float prior,
                              int parent_visit_count,
                              int child_visit_count) {
    const double parent_visits = std::max(1, parent_visit_count);
    const double safe_prior = std::isfinite(prior) && prior > 0.0f ? prior : 0.0;
    const double child_q = std::isfinite(child_value) ? -child_value : 0.0;
    const double c_puct = std::log((1.0 + parent_visits + 1965.2) / 1965.2) + 1.25;
    return child_q + c_puct * safe_prior * std::sqrt(parent_visits) / (child_visit_count + 1);
}

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
    expand(BatchManager::instance());
}

void MCTNode::expand(BatchManager& manager) {
    if (auto win = game_.winner()) {
        children_.clear();
        float v = terminal_value_for_side_to_move(win.value(), game_.to_play());
        visit_count_ = 1;
        total_value_ = v;
        return;
    }

    auto tensor = game_.encodeTensor();
    std::vector<float> flat(tensor.begin(), tensor.end());
    auto [policy, v] = manager.enqueue(flat).get();

    auto legal = game_.legalMoves();
    auto priors = sanitise_priors(policy, legal);

    std::size_t idx = 0;
    for (auto a : legal) {
        ChessGame next = game_;
        next.makeMove(a);
        children_[a] = std::make_unique<MCTNode>(priors[idx++], next);
    }

    visit_count_ = 1;
    total_value_ = v;
}


void MCTNode::add_exploration_noise(double epsilon, double alpha) {

    size_t K = children_.size();
    if (K == 0)
        return;
    std::gamma_distribution<double> gamma(alpha, 1.0);
    std::vector<double> noise(K);
    double sum = 0;
    for (size_t i = 0; i < K; ++i)
    {
        noise[i] = gamma(rng_);
        sum += noise[i];
    }
    if (sum <= 0)
        sum = 1;
    for (auto &n : noise)
        n /= sum;
    std::vector<double> new_p;
    new_p.reserve(K);
    size_t idx = 0;
    for (auto &kv : children_)
    {
        double p = (1 - epsilon) * kv.second->prior_ + epsilon * noise[idx++];
        new_p.push_back(p);
    }
    double psum = std::accumulate(new_p.begin(), new_p.end(), 0.0);
    if (psum <= 0)
        psum = K;
    idx = 0;
    for (auto &kv : children_)
    {
        kv.second->prior_ = new_p[idx++] / psum;
    }
}

std::pair<uint16_t, MCTNode *> MCTNode::select_child()
{
    double best = -1e9;
    uint16_t best_a = 0;
    MCTNode *best_n = nullptr;
    for (auto &kv : children_)
    {
        auto *c = kv.second.get();
        double u = puct_score_from_parent(
            c->value(),
            c->prior_,
            visit_count_,
            c->visit_count_);
        if (u > best)
        {
            best = u;
            best_a = kv.first;
            best_n = c;
        }
    }

    /* If every score was NaN, fall back to the first child */
    if (best_n == nullptr)
    {
        auto it = children_.begin();
        best_a = it->first;
        best_n = it->second.get();
    }
    return {best_a, best_n};
}

void MCTNode::update(float v)
{
    visit_count_++;
    total_value_ += v;
}
/* static */
std::vector<float> MCTNode::sanitise_priors(const std::vector<float> &policy,
                                            const std::vector<uint16_t> &legal)
{
    return normalise_legal_policy(policy, legal);
}

void MCTNode::complete_expand(const std::vector<float> &policy, float v)
{
    // already expanded – nothing to do
    if (!children_.empty())
        return;
    auto legal = game_.legalMoves();
    children_.clear();
    auto priors = sanitise_priors(policy, legal);
    std::size_t idx = 0;
    for (uint16_t a : legal)
    {
        ChessGame next = game_;
        next.makeMove(a);
        children_[a] = std::make_unique<MCTNode>(priors[idx++], next);
    }
    visit_count_ = 1;
    total_value_ = v;
    pending_ = false;
}

const std::unordered_map<uint16_t, std::unique_ptr<MCTNode>> &MCTNode::children() const
{
    return children_;
}

int MCTNode::visit_count() const
{
    return visit_count_;
}
float MCTNode::total_value() const {
    return total_value_;
}

std::vector<float> run_mcts(const ChessGame &root_game, const MCTArgs &args)
{
    return run_mcts(root_game, args, BatchManager::instance());
}

std::vector<float> run_mcts(const ChessGame &root_game, const MCTArgs &args, BatchManager& manager)
{
    MCTNode root(1.0f, root_game);
    root.expand(manager);
    root.add_exploration_noise(args.epsilon, args.alpha);


    for (int i = 0; i < args.num_simulations; ++i) {
        std::deque<MCTNode*> path;
        MCTNode* node = &root;
        path.push_back(node);

        while (node->is_expanded() && !node->children().empty()) {
            auto [a, next] = node->select_child();
            node = next;
            path.push_back(node);
        }

        
        bool expanded_now = false;
        if (!node->is_expanded()) {
            node->expand(manager);
            expanded_now = true;
        }

        float val = node->value();

        auto it = path.rbegin(); 

        if (expanded_now) {
            ++it;
            val = -val;
        }
        for (; it != path.rend(); ++it) {
            (*it)->update(val);
            val = -val;
        }
    }
    //std::cout << "--- MCTS root children stats ---\n";
    //std::cout << root.visit_count() << " visits, "
    //          << root.total_value() << " total value\n";
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
    for (auto &kv : root.children())
        tot += kv.second->visit_count();
    if (tot > 0)
    {
        for (auto &kv : root.children())
        {
            policy[kv.first] = kv.second->visit_count() / tot;
        }
    }
    return policy;
}

} // namespace az73
