// QLearningAgent: a C++20 port of src/narcade/qlearn.py (the semantics
// authority). The update rule and the action-selection RNG consumption
// order mirror the Python line for line so results are bit-comparable
// against native/golden/q_updates.json.
#ifndef NARCADE_QLEARN_HPP
#define NARCADE_QLEARN_HPP

#include <cstdint>
#include <vector>

#include "narcade/rng.hpp"

namespace narcade {

class QLearningAgent {
public:
    using Table = std::vector<std::vector<double>>;

    // Mirrors QLearningAgent(n_states=32, n_actions=3, alpha=0.15,
    // gamma=0.9, epsilon=1.0, seed=None). Python's seed=None draws OS
    // entropy and appears in no fixture; the C++ port requires the seed
    // to be stated (0 if the caller does not care).
    QLearningAgent(int n_states, int n_actions, double alpha, double gamma,
                   double epsilon, std::int64_t seed);

    // qlearn.py lines 18-21: random() is drawn unconditionally; below
    // epsilon the action is rng.choice([STRAIGHT, LEFT, RIGHT]), whose
    // elements equal their indices (0, 1, 2).
    int act(int state);

    // qlearn.py lines 23-28: first-max scan (strictly-greater replaces).
    int greedy(int state) const;

    // qlearn.py lines 30-32:
    //   target = r if done else r + gamma * max(q[s_next])
    //   q[s][a] += alpha * (target - q[s][a])
    void update(int s, int a, double r, int s_next, bool done);

    const Table& q() const { return q_; }
    void set_q(const Table& q) { q_ = q; }
    int n_states() const { return n_states_; }
    int n_actions() const { return n_actions_; }
    double alpha() const { return alpha_; }
    double gamma() const { return gamma_; }
    double epsilon() const { return epsilon_; }

private:
    int n_states_;
    int n_actions_;
    double alpha_;
    double gamma_;
    double epsilon_;
    Table q_;
    PythonRandom rng_;
};

}  // namespace narcade

#endif  // NARCADE_QLEARN_HPP
