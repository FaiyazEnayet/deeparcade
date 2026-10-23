// Native Q-learning trainer: line-by-line port of scripts/train.py.
// See trainer.hpp for the protocol map. The Python file is the
// semantics authority; line citations below refer to it (and to
// src/narcade/qlearn.py / policies.py / snake.py where noted).
#include "narcade/trainer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "narcade/policies.hpp"
#include "narcade/snake.hpp"

namespace narcade {

double elo_expected_score(double elo_a, double elo_b) {
    // train.py:25: 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
    return 1.0 / (1.0 + std::pow(10.0, (elo_b - elo_a) / 400.0));
}

namespace {

using Table = QLearningAgent::Table;

// --- The agent, mirroring src/narcade/qlearn.py lines 18-32. ---
// QLearningAgent defaults (qlearn.py:12): alpha=0.15, gamma=0.9.
constexpr double kAlpha = 0.15;
constexpr double kGamma = 0.9;

int agent_greedy(const Table& q, int state) {
    // qlearn.py:23-28: first-max scan over actions (1, 2).
    const std::vector<double>& qs = q[state];
    int best = 0;
    for (int a : {1, 2}) {
        if (qs[a] > qs[best]) {
            best = a;
        }
    }
    return best;
}

int agent_act(const Table& q, PythonRandom& rng, double epsilon, int state) {
    // qlearn.py:18-21: random() is drawn unconditionally; below epsilon
    // the action is rng.choice([STRAIGHT, LEFT, RIGHT]) == index draw.
    if (rng.random() < epsilon) {
        return static_cast<int>(rng.choice_index(3));
    }
    return agent_greedy(q, state);
}

void agent_update(Table& q, int s, int a, double r, int s_next, bool done) {
    // qlearn.py:30-32, operation order preserved:
    //   target = r if done else r + gamma * max(q[s_next])
    //   q[s][a] += alpha * (target - q[s][a])
    double target = r;
    if (!done) {
        const std::vector<double>& next = q[s_next];
        target = r + kGamma * std::max({next[0], next[1], next[2]});
    }
    q[s][a] += kAlpha * (target - q[s][a]);
}

// train.py:27-34 play_episode: reset, then step policy.act(env) until
// done; returns env.score. (Python also accumulates the reward total
// into a local it never returns; that dead accumulation is omitted.)
template <typename ActFn>
int play_episode(SnakeEnv& env, ActFn&& act) {
    env.reset();
    bool done = false;
    while (!done) {
        done = env.step(act(env)).done;
    }
    return env.score();
}

}  // namespace

TrainResult train_agent(const TrainConfig& config) {
    // train.py:34-35.
    Table q(32, std::vector<double>(3, 0.0));
    PythonRandom rng(config.seed);
    double epsilon = config.eps_start;
    const double eps_decay =
        (config.eps_start - config.eps_end) / config.episodes;

    TrainResult result;
    result.ep_scores.reserve(config.episodes);
    result.ep_rewards.reserve(config.episodes);

    // train.py:39-43: elo = 1200.0, K = 32.0, and the baselines dict,
    // created ONCE before the loop. Iteration order is insertion
    // order: "random" first, then "greedy". The RandomPolicy is
    // stateful: its RNG carries across every checkpoint and game.
    double elo = 1200.0;
    constexpr double kK = 32.0;
    RandomPolicy base_random(1234);
    auto act_random = [&base_random](const SnakeEnv&) {
        return base_random.act();
    };
    auto act_greedy = [](const SnakeEnv& env) {
        return GreedyPolicy::act(env);
    };

    for (int ep = 1; ep <= config.episodes; ++ep) {
        // train.py:47-57. The env constructor already reset once
        // (snake.py:33); train.py calls reset() again, consuming the
        // food-placement RNG a second time — mirrored deliberately.
        SnakeEnv env(config.width, config.height, config.seed + ep);
        int s = env.reset();
        bool done = false;
        double ep_r = 0.0;
        while (!done) {
            const int a = agent_act(q, rng, epsilon, s);
            const StepResult step = env.step(a);
            agent_update(q, s, a, step.reward, step.state, step.done);
            s = step.state;
            ep_r += step.reward;
            done = step.done;
        }
        result.ep_scores.push_back(env.score());
        result.ep_rewards.push_back(ep_r);
        // train.py:60: decay AFTER the episode, floored at eps_end.
        epsilon = std::max(config.eps_end, epsilon - eps_decay);

        if (ep % 100 == 0) {
            // train.py:62-64: mean of the last 100. Python sums ints
            // exactly then divides; rewards sum as floats in order.
            long long score_sum = 0;
            double reward_sum = 0.0;
            for (int i = ep - 100; i < ep; ++i) {
                score_sum += result.ep_scores[i];
                reward_sum += result.ep_rewards[i];
            }
            result.curve_scores.push_back(score_sum / 100.0);
            result.curve_rewards.push_back(reward_sum / 100.0);
        }

        if (ep % config.checkpoint_every == 0) {
            // train.py:66-80. QTablePolicy(agent.q) in Python holds the
            // live table; no updates intervene during the checkpoint,
            // so this snapshot copy is equivalent.
            const QTablePolicy policy(q);
            auto act_ai = [&policy](const SnakeEnv& env) {
                return policy.act(env);
            };
            for (int b = 0; b < 2; ++b) {  // random, then greedy
                for (int g = 0; g < config.elo_games; ++g) {
                    // train.py:70-73: BOTH envs get the same seed.
                    SnakeEnv e1(config.width, config.height,
                                900000 + ep + g);
                    SnakeEnv e2(config.width, config.height,
                                900000 + ep + g);
                    const int s_ai = play_episode(e1, act_ai);
                    const int s_base =
                        (b == 0) ? play_episode(e2, act_random)
                                 : play_episode(e2, act_greedy);
                    const double actual =
                        (s_ai > s_base) ? 1.0 : (s_ai == s_base ? 0.5 : 0.0);
                    // train.py:78: baseline rating is fixed at 1200.0.
                    elo += kK * (actual - elo_expected_score(elo, 1200.0));
                }
            }
            result.elo_hist.push_back({ep, elo});
            if (config.verbose) {
                // train.py:83-85.
                std::printf("ep %6d  mean100=%5.2f  eps=%.3f  elo=%.0f\n",
                            ep, result.curve_scores.back(), epsilon, elo);
                std::fflush(stdout);
            }
        }
    }

    result.q = q;
    result.best_action = QTablePolicy(q).best_action();
    result.final_elo = elo;
    result.final_epsilon = epsilon;
    result.final_rng_state = rng.save();
    return result;
}

}  // namespace narcade
