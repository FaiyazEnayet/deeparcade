// C++20 port of src/narcade/policies.py. The Python file is the
// semantics authority; comments cite its lines.
#include "narcade/policies.hpp"

#include <cstdlib>

namespace narcade {
namespace {

// Headings: 0=up, 1=right, 2=down, 3=left (snake.py line 12).
constexpr int kDirs[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};

// Equivalent of SnakeEnv._cell_blocked(r, c, ignore_tail) (snake.py
// lines 54-60), expressed through the C++ env's public accessors
// (SnakeEnv::cell_blocked itself is private): out of bounds is blocked;
// otherwise membership in the body with body[0] (the tail) discarded
// when ignore_tail. Body cells are unique in every reachable state, so
// scanning with the front element skipped is the same predicate as
// Python's set difference (the same argument snake.cpp relies on).
// Any divergence from the Python predicate is caught by the frozen
// policy corpus (native/golden/policies_corpus.txt), whose expected
// actions were produced by Python's GreedyPolicy calling the real
// _cell_blocked on >= 34,532 states.
bool cell_blocked(const SnakeEnv& env, int r, int c, bool ignore_tail) {
    if (!(0 <= r && r < env.height() && 0 <= c && c < env.width())) {
        return true;
    }
    std::size_t i = 0;
    for (const Cell& cell : env.body()) {
        if (ignore_tail && i == 0) {
            ++i;
            continue;
        }
        ++i;
        if (cell.r == r && cell.c == c) {
            return true;
        }
    }
    return false;
}

}  // namespace

int GreedyPolicy::act(const SnakeEnv& env) {
    // policies.py lines 27-44. Actions are tried in the order
    // (STRAIGHT, LEFT, RIGHT); the heading is recomputed from
    // env.heading for each candidate; a candidate is discarded when
    // its target cell is blocked with ignore_tail=True; among the safe
    // candidates the first one achieving the minimal Manhattan
    // distance to the food wins (strictly-less replacement).
    const Cell head = env.body().back();
    const Cell food = *env.food();  // live states only (see header).
    bool any_safe = false;
    bool have_best = false;
    int best = kStraight;
    int best_d = 0;
    for (int a : {kStraight, kLeft, kRight}) {
        int h = env.heading();
        if (a == kLeft) {
            h = (h + 3) % 4;
        } else if (a == kRight) {
            h = (h + 1) % 4;
        }
        const int nr = head.r + kDirs[h][0];
        const int nc = head.c + kDirs[h][1];
        if (cell_blocked(env, nr, nc, /*ignore_tail=*/true)) {
            continue;
        }
        any_safe = true;
        const int d = std::abs(nr - food.r) + std::abs(nc - food.c);
        if (!have_best || d < best_d) {
            have_best = true;
            best_d = d;
            best = a;
        }
    }
    if (any_safe) {
        return best;
    }
    return kStraight;  // doomed; pick anything (policies.py line 44).
}

int QTablePolicy::act(const SnakeEnv& env) const {
    // policies.py lines 52-59: same first-max scan as QLearningAgent.
    const std::vector<double>& qs =
        qtable_[static_cast<std::size_t>(env.encode_state())];
    int best = 0;
    for (int a = 1; a < 3; ++a) {
        if (qs[static_cast<std::size_t>(a)] >
            qs[static_cast<std::size_t>(best)]) {
            best = a;
        }
    }
    return best;
}

std::vector<int> QTablePolicy::best_action() const {
    // policies.py lines 61-64: max(range(3), key=lambda a: row[a])
    // returns the first maximal index on ties.
    std::vector<int> out;
    out.reserve(qtable_.size());
    for (const std::vector<double>& row : qtable_) {
        int best = 0;
        for (int a = 1; a < 3; ++a) {
            if (row[static_cast<std::size_t>(a)] >
                row[static_cast<std::size_t>(best)]) {
                best = a;
            }
        }
        out.push_back(best);
    }
    return out;
}

}  // namespace narcade
