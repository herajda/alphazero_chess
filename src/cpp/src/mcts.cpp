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
    auto win = game_.winner();
    if (win.has_value()) {
        children_.clear();
        float v;
        int w = win.value();
        if (w == -1) v = 0.0f;
        else if (w == game_.to_play()) v = 1.0f;
        else v = -1.0f;
        visit_count_ = 1;
        total_value_ = v;
        return;
    }
    auto tensor = game_.encodeTensor();
    std::vector<float> flat(tensor.begin(), tensor.end());
    auto fut = BatchManager::instance().enqueue(flat);
    auto [policy, v] = fut.get();
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
}

void MCTNode::add_exploration_noise(double epsilon, double alpha)
{
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
    double N = visit_count_;
    double C = std::log((1 + N + 1965.2) / 1965.2) + 1.25;
    for (auto &kv : children_)
    {
        auto *c = kv.second.get();
        // double Q = c->value();
        // double P = c->prior_;
        double Q = std::isfinite(c->value()) ? c->value() : 0.0;
        double P = std::isfinite(c->prior_) ? c->prior_ : 0.0;
        double Nsa = c->visit_count_;
        double u = Q + C * P * std::sqrt(N) / (Nsa + 1);
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
    std::vector<float> priors;
    priors.reserve(legal.size());

    for (uint16_t a : legal)
    {
        float p = policy[a];
        if (!std::isfinite(p) || p <= 0.f)
            p = 1e-8f; // clamp bad or zero probs
        priors.push_back(p);
    }

    float sum = std::accumulate(priors.begin(), priors.end(), 0.0f);
    if (sum <= 0.f)
        sum = 1.f;
    for (float &p : priors)
        p /= sum; // explicit renorm

    return priors;
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

std::vector<float> run_mcts(const ChessGame &root_game, const MCTArgs &args)
{
    MCTNode root(1.0f, root_game);
    root.expand();
    root.add_exploration_noise(args.epsilon, args.alpha);

    struct Pending
    {
        std::future<std::pair<std::vector<float>, float>> fut;
        std::vector<MCTNode *> path; // ancestors of the leaf (root‑exclusive)
        MCTNode *leaf;
    };
    std::vector<Pending> in_flight;

    int completed = 0;
    while (completed < args.num_simulations)
    {
        for (auto it = in_flight.begin(); it != in_flight.end();)
        {
            if (it->fut.wait_for(std::chrono::seconds(0)) ==
                std::future_status::ready)
            {

                auto [policy, v] = it->fut.get();
                it->leaf->complete_expand(policy, v);

                /* backup */
                float val = v;
                for (auto p = it->path.rbegin(); p != it->path.rend(); ++p)
                {
                    (*p)->update(val);
                    val = -val;
                }
                ++completed;
                it = in_flight.erase(it);
            }
            else
                ++it;
        }
        if (completed >= args.num_simulations)
            break;

        /* ---------- launch a new simulation ---------- */
        std::vector<MCTNode *> path;
        MCTNode *node = &root;

        while (node->is_expanded() && !node->children().empty())
        {

            auto [a, next] = node->select_child();
            path.push_back(node);
            node = next;
        }

        /* two cases:
         *   (a) node already waiting for eval  -> skip
         *   (b) brand‑new leaf                -> enqueue net call
         */

        if (!node->is_expanded() && !node->pending())
        {
            auto tensor = node->game().encodeTensor();
            std::vector<float> flat(tensor.begin(), tensor.end());
            Pending p;
            p.fut = BatchManager::instance().enqueue(flat);
            p.path = std::move(path);
            p.leaf = node;
            node->mark_pending();
            in_flight.emplace_back(std::move(p));
        }
    }

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
