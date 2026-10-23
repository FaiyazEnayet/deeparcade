// Native Q-learning trainer: a C++20 port of scripts/train.py (the
// semantics authority), mirroring the ENTIRE training protocol, not
// just the final table:
//
//   * episode loop with per-episode env seeds seed+ep (train.py:47-60)
//   * exploration schedule: eps_decay = (eps_start - eps_end)/episodes
//     computed once, applied AFTER each episode as
//     max(eps_end, eps - eps_decay) (train.py:35, 60)
//   * learning curves: mean of the last 100 episode scores/rewards,
//     appended every 100 episodes BEFORE the checkpoint block
//     (train.py:62-64)
//   * checkpoint Elo against persistent seeded baselines in dict order
//     (random: RandomPolicy(seed=1234), then greedy; train.py:43),
//     elo_games paired games per baseline on envs seeded
//     900000+ep+g for BOTH sides (train.py:66-80), Elo starts at
//     1200.0, K = 32.0, baseline rating fixed at 1200.0
//   * output artifacts qtable.json / policy.json / curve.json /
//     elo.json / train_meta.json with train.py's field sets
//
// The agent itself (act / greedy / update) mirrors src/narcade/
// qlearn.py lines 18-32 operation for operation. It is re-implemented
// here over a raw table + PythonRandom because the CPP-02
// QLearningAgent's public API exposes neither an epsilon setter nor
// its RNG state, both of which the training protocol mutates/reads;
// the operations are identical to the fixture-verified qlearn.cpp.
#ifndef NARCADE_TRAINER_HPP
#define NARCADE_TRAINER_HPP

#include <cstdint>
#include <vector>

#include "narcade/qlearn.hpp"  // QLearningAgent::Table
#include "narcade/rng.hpp"     // PythonRandomState

namespace narcade {

// scripts/train.py expected_score (lines 24-25).
double elo_expected_score(double elo_a, double elo_b);

// train() keyword arguments (train.py:31-33) with their defaults.
struct TrainConfig {
    int episodes = 30000;
    std::int64_t seed = 7;
    double eps_start = 1.0;
    double eps_end = 0.05;
    int checkpoint_every = 2000;
    int elo_games = 50;
    int width = 12;
    int height = 12;
    bool verbose = true;
};

struct EloCheckpoint {
    int episode = 0;
    double elo = 0.0;
};

// Everything train() computes (train.py:96) plus the per-episode
// streams and the agent's final RNG state, which train.py keeps in
// locals (ep_scores / ep_rewards, lines 37, 58-59) and which the
// parity driver needs to verify the episode stream and the RNG
// consumption sequence exactly.
struct TrainResult {
    QLearningAgent::Table q;
    std::vector<int> best_action;  // QTablePolicy(agent.q).to_jsonable()
    std::vector<double> curve_scores;
    std::vector<double> curve_rewards;
    std::vector<EloCheckpoint> elo_hist;
    std::vector<int> ep_scores;
    std::vector<double> ep_rewards;
    double final_elo = 1200.0;
    double final_epsilon = 1.0;
    PythonRandomState final_rng_state;
};

// Runs the full protocol. If verbose, prints train.py's per-checkpoint
// line (train.py:83-85) to stdout.
TrainResult train_agent(const TrainConfig& config);

}  // namespace narcade

#endif  // NARCADE_TRAINER_HPP
