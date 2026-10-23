// C++20 port of src/narcade/qlearn.py. The Python file is the semantics
// authority; comments cite its lines so divergences are auditable.
#include "narcade/qlearn.hpp"

#include <algorithm>

namespace narcade {

QLearningAgent::QLearningAgent(int n_states, int n_actions, double alpha,
                               double gamma, double epsilon,
                               std::int64_t seed)
    : n_states_(n_states),
      n_actions_(n_actions),
      alpha_(alpha),
      gamma_(gamma),
      epsilon_(epsilon),
      q_(static_cast<std::size_t>(n_states),
         std::vector<double>(static_cast<std::size_t>(n_actions), 0.0)),
      rng_(seed) {}

int QLearningAgent::act(int state) {
    // Lines 19-21: the random() draw happens even when epsilon == 0;
    // choice([STRAIGHT, LEFT, RIGHT]) picks among the literal 3 actions,
    // whose values equal their indices.
    if (rng_.random() < epsilon_) {
        return static_cast<int>(rng_.choice_index(3));
    }
    return greedy(state);
}

int QLearningAgent::greedy(int state) const {
    // Lines 24-28: best starts at 0 and is replaced only by a strictly
    // greater value, scanning a = 1, 2 (n_actions_ == 3 in every use).
    const std::vector<double>& qs = q_[static_cast<std::size_t>(state)];
    int best = 0;
    for (int a = 1; a < n_actions_; ++a) {
        if (qs[static_cast<std::size_t>(a)] >
            qs[static_cast<std::size_t>(best)]) {
            best = a;
        }
    }
    return best;
}

void QLearningAgent::update(int s, int a, double r, int s_next, bool done) {
    // Lines 31-32, same operation order as Python:
    //   target = r if done else r + gamma * max(q[s_next])
    //   q[s][a] += alpha * (target - q[s][a])
    double target = r;
    if (!done) {
        const std::vector<double>& next = q_[static_cast<std::size_t>(s_next)];
        double mx = next[0];
        for (std::size_t i = 1; i < next.size(); ++i) {
            mx = std::max(mx, next[i]);
        }
        target = r + gamma_ * mx;
    }
    double& cell = q_[static_cast<std::size_t>(s)][static_cast<std::size_t>(a)];
    cell += alpha_ * (target - cell);
}

}  // namespace narcade
