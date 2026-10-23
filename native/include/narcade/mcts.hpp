// MCTSPlayer: a C++20 port of src/narcade/mcts.py (the semantics
// authority). UCT search with greedy rollouts over lightweight state
// (body list, food, heading) — deliberately independent of SnakeEnv,
// exactly as in Python, including the places where _step_state's
// transition rules differ from SnakeEnv.step (walls, (-1,-1) food on a
// full board). Behavioral parity is gated on native/golden/
// mcts_oracle.json plus the frozen session corpus
// native/golden/mcts_cases.txt (>= 1,000 seeded searches, exact
// action agreement, root-child values bit-exact).
#ifndef NARCADE_MCTS_HPP
#define NARCADE_MCTS_HPP

#include <cstdint>
#include <unordered_set>
#include <vector>

#include "narcade/rng.hpp"
#include "narcade/snake.hpp"  // Cell

namespace narcade {

// Wall cells for search, encoded as (r << 32) | (uint32)c. Python
// passes a set of (r, c) tuples; membership is all that is queried.
using MctsWalls = std::unordered_set<std::int64_t>;

inline std::int64_t mcts_cell_key(int r, int c) {
    return (static_cast<std::int64_t>(r) << 32) |
           static_cast<std::uint32_t>(c);
}

// Lightweight search state: the pieces of the env that
// MCTSPlayer.search reads (width, height, body tuple, food, heading).
// `body` is tail-first, head-last, like tuple(env.body) in Python.
// `food` may be (-1, -1) inside search after a full-board eat.
struct MctsState {
    int width = 0;
    int height = 0;
    std::vector<Cell> body;
    Cell food{-1, -1};
    int heading = 1;
};

// One root child in insertion order (Python dict order).
struct MctsRootChild {
    int action = 0;
    long long visits = 0;
    double value = 0.0;
};

struct MctsSearchResult {
    int action = 0;  // kStraight when the root has no children
    std::vector<MctsRootChild> root_children;
};

class MCTSPlayer {
public:
    // Mirrors MCTSPlayer(simulations=300, rollout_steps=100, c=1.0,
    // seed=...). The single PythonRandom is shared by selection,
    // expansion, and rollouts, and across search() calls (the Python
    // RNG lifecycle); it is never reseeded per call.
    MCTSPlayer(int simulations, int rollout_steps, double c,
               std::int64_t seed);

    // Best relative action from the given state (Python search()).
    int search(const MctsState& state, const MctsWalls& walls);
    int act(const MctsState& state, const MctsWalls& walls) {
        return search(state, walls);
    }

    // search() plus the root's children in insertion order, for
    // differential comparison against the Python reference.
    MctsSearchResult search_with_stats(const MctsState& state,
                                       const MctsWalls& walls);

private:
    struct StepStateResult {
        std::vector<Cell> body;
        Cell food;
        int heading = 1;
        bool ate = false;
        bool dead = false;
    };

    StepStateResult step_state(const std::vector<Cell>& body, Cell food,
                               int heading, const MctsWalls& walls,
                               int w, int h, int action);
    double rollout(std::vector<Cell> body, Cell food, int heading,
                   const MctsWalls& walls, int w, int h, double disc0,
                   double gamma);

    int simulations_;
    int rollout_steps_;
    double c_;
    PythonRandom rng_;
};

}  // namespace narcade

#endif  // NARCADE_MCTS_HPP
