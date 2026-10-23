// Evaluation engine: a C++20 port of scripts/evaluate.py's paired
// evaluation protocol (the semantics authority; line citations below
// refer to scripts/evaluate.py).
//
// SCOPE (mandatory declaration, mirrored in the CLI help, the CLI
// report's "scope" field, and the driver): this is a NON-NEURAL
// evaluator. evaluate.py evaluates five players; the fifth, "pro", is
// a DQN loaded from a PyTorch .pt checkpoint (evaluate.py lines 20-38,
// 109). Until a separate native DQN inference module is verified, this
// engine covers only random, greedy, rookie (Q-table), and grandmaster
// (MCTS). It never silently omits the DQN player and never claims full
// evaluate.py equivalence.
#ifndef NARCADE_EVALUATE_HPP
#define NARCADE_EVALUATE_HPP

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "narcade/mcts.hpp"
#include "narcade/policies.hpp"
#include "narcade/snake.hpp"

namespace narcade {

// One episode's outcome: evaluate.py's run_episodes records only the
// score (line 53); steps are carried alongside because callers (and
// the differential driver) compare them too. steps is env.steps at
// termination, exactly as Python's env.steps reads.
struct EpisodeResult {
    int score = 0;
    std::int64_t steps = 0;
};

// A player in the paired protocol. evaluate.py's policies are objects
// with act(env) (lines 104-110); stateful players (RandomPolicy's RNG,
// the single shared MCTSPlayer) keep their state across episodes
// because run_episodes receives one policy_fn per player (line 43).
class Player {
public:
    virtual ~Player() = default;
    virtual int act(SnakeEnv& env) = 0;
};

// evaluate.py line 105: RandomPolicy(seed=999), one instance for all
// episodes. Its act() ignores the env (policies.py lines 13-14).
class RandomPlayer final : public Player {
public:
    explicit RandomPlayer(std::int64_t seed = 999) : policy_(seed) {}
    int act(SnakeEnv& /*env*/) override { return policy_.act(); }

private:
    RandomPolicy policy_;
};

// evaluate.py line 106: GreedyPolicy(), stateless.
class GreedyPlayer final : public Player {
public:
    int act(SnakeEnv& env) override { return GreedyPolicy::act(env); }
};

// evaluate.py line 107: QTablePolicy(qtable) over runs/qtable.json.
class QTablePlayer final : public Player {
public:
    explicit QTablePlayer(QTablePolicy::Table table)
        : policy_(std::move(table)) {}
    int act(SnakeEnv& env) override { return policy_.act(env); }

private:
    QTablePolicy policy_;
};

// evaluate.py line 122: ONE MCTSPlayer(simulations=mcts_sims,
// seed=31337) reused across all its paired episodes; rollout_steps and
// c keep the Python defaults (mcts.py line 40: 100, 1.0). act() reads
// only width/height/body/food/heading from the env and searches with
// the default empty wall set (mcts.py search(env, walls=frozenset())).
class MctsPlayer final : public Player {
public:
    MctsPlayer(int simulations, std::int64_t seed = 31337)
        : player_(simulations, /*rollout_steps=*/100, /*c=*/1.0, seed) {}
    int act(SnakeEnv& env) override;

private:
    MCTSPlayer player_;
};

// Canonical player names from evaluate.py main(): "random", "greedy",
// "rookie", "grandmaster" (lines 104-110, 123). Aliases "qtable" and
// "mcts" are accepted by the CLI and mapped to "rookie"/"grandmaster".
std::unique_ptr<Player> make_player(const std::string& canonical_name,
                                    int mcts_simulations,
                                    const QTablePolicy::Table* qtable);

// Loads runs/qtable.json: a bare 32x3 JSON array of numbers (the exact
// file evaluate.py lines 102-103 load for QTablePolicy). Throws
// std::runtime_error on malformed input or a wrong element count.
QTablePolicy::Table load_qtable_json(const std::string& path);

// run_episodes (evaluate.py lines 43-54): one episode per seed, in
// order. Lifecycle per episode, mirrored exactly: construct
// SnakeEnv(width, height, seed) — whose constructor already calls
// reset() (snake.py line 32) — then call reset() AGAIN (evaluate.py
// line 49; the double reset consumes the food RNG twice), then step
// the policy until done and record env.score (and env.steps).
std::vector<EpisodeResult> run_episodes(Player& player,
                                        const std::vector<std::int64_t>& seeds,
                                        int width = 12, int height = 12);

// summarize (evaluate.py lines 57-62). mean/std are the Python-rounded
// outputs (round(x, 2)); raw_mean/raw_std carry the pre-rounding
// values for differential comparison. median mirrors
// statistics.median (middle value, or mean of the two middle values);
// std mirrors statistics.pstdev (population), 0.0 when n <= 1.
struct Summary {
    std::int64_t n = 0;
    double mean = 0.0;
    double std = 0.0;
    double median = 0.0;
    int max = 0;
    int min = 0;
    double raw_mean = 0.0;
    double raw_std = 0.0;
};
Summary summarize(const std::vector<int>& scores);

// elo_ratings (evaluate.py lines 65-78): ratings start at 1200.0,
// k = 32, expected score 1 / (1 + 10 ** ((elo_b - elo_a) / 400)),
// round-robin over paired seeds in the results' iteration order,
// sequential in-place updates. elo_ratings_raw returns the pre-rounding
// ratings; elo_ratings applies Python's round() (round-half-even) to
// int, exactly like line 78.
using OrderedResults =
    std::vector<std::pair<std::string, std::vector<int>>>;
std::vector<std::pair<std::string, double>> elo_ratings_raw(
    const OrderedResults& results, double k = 32.0);
std::vector<std::pair<std::string, std::int64_t>> elo_ratings(
    const OrderedResults& results, double k = 32.0);

// head_to_head (evaluate.py lines 81-95): per ordered pair, win/draw
// counts over the paired seeds; win_rate/draw_rate are the
// Python-rounded outputs (round(x, 3)); the diagonal is nullopt
// (Python None).
struct HeadToHeadCell {
    std::int64_t wins = 0;
    std::int64_t draws = 0;
    double win_rate = 0.0;
    double draw_rate = 0.0;
};
using HeadToHeadTable =
    std::vector<std::vector<std::optional<HeadToHeadCell>>>;
HeadToHeadTable head_to_head(const OrderedResults& results);

// Python round() semantics helpers (round-half-even on the exact
// binary value, via the C library's correctly rounded decimal
// formatting / the current FP rounding mode), shared with the CLI.
double py_round(double x, int ndigits);
std::int64_t py_round_int(double x);

}  // namespace narcade

#endif  // NARCADE_EVALUATE_HPP
