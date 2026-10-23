// Policies: a C++20 port of src/narcade/policies.py (the semantics
// authority). RandomPolicy and QLearningAgent own CPython-compatible
// RNGs; GreedyPolicy and QTablePolicy are stateless evaluators over a
// SnakeEnv snapshot, exactly like their Python counterparts.
#ifndef NARCADE_POLICIES_HPP
#define NARCADE_POLICIES_HPP

#include <cstdint>
#include <vector>

#include "narcade/rng.hpp"
#include "narcade/snake.hpp"

namespace narcade {

// policies.py RandomPolicy: act() is rng.choice([STRAIGHT, LEFT, RIGHT]);
// the chosen element equals its index.
class RandomPolicy {
public:
    explicit RandomPolicy(std::int64_t seed) : rng_(seed) {}
    int act() { return static_cast<int>(rng_.choice_index(3)); }

private:
    PythonRandom rng_;
};

// policies.py GreedyPolicy: move toward the food, avoid instant death.
// Stateless; act() is a pure function of the env snapshot.
class GreedyPolicy {
public:
    // Mirrors GreedyPolicy.act(env). Precondition (as in Python, which
    // would raise TypeError subscripting None): the env has food, i.e.
    // the episode is not in the terminal-win state.
    static int act(const SnakeEnv& env);
};

// policies.py QTablePolicy: greedy action from a 32x3 Q-table.
class QTablePolicy {
public:
    using Table = std::vector<std::vector<double>>;

    explicit QTablePolicy(Table qtable) : qtable_(std::move(qtable)) {}

    // Mirrors QTablePolicy.act(env): greedy over qtable[encode_state()].
    int act(const SnakeEnv& env) const;

    // The "best_action" field of QTablePolicy.to_jsonable(): Python's
    // max(range(3), key=...) returns the FIRST maximal index, the same
    // first-max tie rule as the greedy scan.
    std::vector<int> best_action() const;

    const Table& qtable() const { return qtable_; }

private:
    Table qtable_;
};

}  // namespace narcade

#endif  // NARCADE_POLICIES_HPP
