// MCTSPlayer port: line-by-line mirror of src/narcade/mcts.py.
// Line references below are to mcts.py. Where Python builds a set or a
// list per call, the C++ builds the equivalent container; only
// membership and order are observable, and both are preserved.
#include "narcade/mcts.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

namespace narcade {
namespace {

// DIRS (snake.py, imported by mcts.py): 0=up, 1=right, 2=down, 3=left.
constexpr int kDirs[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};

// ACTIONS (mcts.py line 15): (STRAIGHT, LEFT, RIGHT).
constexpr int kActions[3] = {kStraight, kLeft, kRight};

// _rel_heading (mcts.py lines 18-23).
int rel_heading(int heading, int action) {
    if (action == kLeft) {
        return (heading + 3) % 4;
    }
    if (action == kRight) {
        return (heading + 1) % 4;
    }
    return heading;
}

// _blocked (mcts.py lines 42-46): static helper in Python; the two call
// sites inline the same predicate there, and do so here via this
// function. `body_set` holds cell keys.
bool blocked(const std::unordered_set<std::int64_t>& body_set,
             const MctsWalls& walls, int w, int h, int r, int c) {
    if (!(0 <= r && r < h && 0 <= c && c < w)) {
        return true;
    }
    const std::int64_t key = mcts_cell_key(r, c);
    return body_set.count(key) != 0 || walls.count(key) != 0;
}

std::unordered_set<std::int64_t> body_set_of(
    const std::vector<Cell>& body) {
    std::unordered_set<std::int64_t> set;
    set.reserve(body.size());
    for (const Cell& cell : body) {
        set.insert(mcts_cell_key(cell.r, cell.c));
    }
    return set;
}

// _Node (mcts.py lines 26-35). Python children is a dict keyed by
// action; iteration order is insertion order. Here child_order records
// that order and children[] indexes by action (actions are 0..2).
struct Node {
    Node* parent = nullptr;
    int action = -1;
    std::array<std::unique_ptr<Node>, 3> children{};
    std::vector<int> child_order;
    std::vector<int> untried{kStraight, kLeft, kRight};
    long long visits = 0;
    double value = 0.0;

    Node(Node* parent_, int action_) : parent(parent_), action(action_) {}
};

// _uct (mcts.py lines 144-154): iterate children in insertion order,
// strictly-greater replacement (first wins ties).
int uct(const Node& node, double c) {
    int best = -1;
    double best_v = -1e18;
    for (const int a : node.child_order) {
        const Node& ch = *node.children[a];
        double v = 0.0;
        if (ch.visits == 0) {
            v = 1e17;
        } else {
            v = (ch.value / static_cast<double>(ch.visits) +
                 c * std::sqrt(std::log(static_cast<double>(node.visits)) /
                               static_cast<double>(ch.visits)));
        }
        if (v > best_v) {
            best_v = v;
            best = a;
        }
    }
    return best;
}

}  // namespace

MCTSPlayer::MCTSPlayer(int simulations, int rollout_steps, double c,
                       std::int64_t seed)
    : simulations_(simulations),
      rollout_steps_(rollout_steps),
      c_(c),
      rng_(seed) {}

// _step_state (mcts.py lines 84-102).
MCTSPlayer::StepStateResult MCTSPlayer::step_state(
    const std::vector<Cell>& body, Cell food, int heading,
    const MctsWalls& walls, int w, int h, int action) {
    heading = rel_heading(heading, action);
    const int hr = body.back().r;
    const int hc = body.back().c;
    const int nr = hr + kDirs[heading][0];
    const int nc = hc + kDirs[heading][1];
    const bool ate = (nr == food.r && nc == food.c);
    std::unordered_set<std::int64_t> occ = body_set_of(body);
    if (!ate) {
        occ.erase(mcts_cell_key(body.front().r, body.front().c));
    }
    if (blocked(occ, walls, w, h, nr, nc)) {
        // Python returns (body, food, heading, False, True): the new
        // heading is kept, body and food are the originals.
        return {body, food, heading, false, true};
    }
    std::vector<Cell> new_body = body;
    new_body.push_back(Cell{nr, nc});
    Cell new_food = food;
    if (ate) {
        const std::unordered_set<std::int64_t> occ2 = body_set_of(new_body);
        std::vector<Cell> free_cells;
        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < w; ++c) {
                const std::int64_t key = mcts_cell_key(r, c);
                if (occ2.count(key) == 0 && walls.count(key) == 0) {
                    free_cells.push_back(Cell{r, c});
                }
            }
        }
        new_food = free_cells.empty()
                       ? Cell{-1, -1}
                       : free_cells[rng_.choice_index(free_cells.size())];
    } else {
        new_body.erase(new_body.begin());
    }
    return {new_body, new_food, heading, ate, false};
}

// _rollout (mcts.py lines 104-129), with _greedy_rollout_action
// (lines 53-82) inlined at its single call site.
double MCTSPlayer::rollout(std::vector<Cell> body, Cell food, int heading,
                           const MctsWalls& walls, int w, int h,
                           double disc0, double gamma) {
    // _rollout (mcts.py lines 104-129).
    double disc = disc0;
    double value = 0.0;
    for (int i = 0; i < rollout_steps_; ++i) {
        // --- _greedy_rollout_action (mcts.py lines 53-82), inlined ---
        const int hr = body.back().r;
        const int hc = body.back().c;
        const std::unordered_set<std::int64_t> body_set =
            body_set_of(body);
        std::optional<std::array<int, 4>> best;
        std::optional<int> best_d;
        std::vector<std::array<int, 4>> safe;
        for (const int a : kActions) {
            const int hh = rel_heading(heading, a);
            const int nr = hr + kDirs[hh][0];
            const int nc = hc + kDirs[hh][1];
            std::unordered_set<std::int64_t> occ = body_set;
            occ.erase(mcts_cell_key(body.front().r, body.front().c));
            if (blocked(occ, walls, w, h, nr, nc)) {
                continue;
            }
            safe.push_back({a, hh, nr, nc});
            const int d = std::abs(nr - food.r) + std::abs(nc - food.c);
            if (!best_d.has_value() || d < *best_d) {
                best_d = d;
                best = {a, hh, nr, nc};
            }
        }
        std::optional<std::array<int, 4>> mv;
        if (best.has_value()) {
            mv = best;
        } else if (!safe.empty()) {
            mv = safe[rng_.choice_index(safe.size())];
        }
        if (!mv.has_value()) {
            break;
        }
        // --- _step_state with the chosen action ---
        const StepStateResult res =
            step_state(body, food, heading, walls, w, h, (*mv)[0]);
        body = res.body;
        food = res.food;
        heading = res.heading;
        if (res.dead) {
            break;
        }
        if (res.ate) {
            value += disc;
        }
        disc *= gamma;
    }
    return value;
}

MctsSearchResult MCTSPlayer::search_with_stats(const MctsState& state,
                                               const MctsWalls& walls) {
    // search (mcts.py lines 156-207).
    const int w = state.width;
    const int h = state.height;
    const std::vector<Cell> root_body = state.body;
    const Cell root_food = state.food;
    const int root_heading = state.heading;

    Node root(nullptr, -1);
    constexpr double gamma = 0.95;
    for (int sim = 0; sim < simulations_; ++sim) {
        // --- selection ---
        Node* node = &root;
        std::vector<Cell> body = root_body;
        Cell food = root_food;
        int heading = root_heading;
        bool dead = false;
        double path_value = 0.0;
        double disc = 1.0;
        while (!node->child_order.empty() && node->untried.empty() &&
               !dead) {
            const int a = uct(*node, c_);
            node = node->children[a].get();
            const StepStateResult res =
                step_state(body, food, heading, walls, w, h, a);
            body = res.body;
            food = res.food;
            heading = res.heading;
            dead = res.dead;
            if (res.ate) {
                path_value += disc;
            }
            disc *= gamma;
        }
        // --- expansion ---
        if (!dead && !node->untried.empty()) {
            const std::uint64_t idx =
                rng_.randrange(node->untried.size());
            const int a = node->untried[static_cast<std::size_t>(idx)];
            node->untried.erase(node->untried.begin() +
                                 static_cast<std::ptrdiff_t>(idx));
            auto child = std::make_unique<Node>(node, a);
            Node* child_ptr = child.get();
            node->children[a] = std::move(child);
            node->child_order.push_back(a);
            node = child_ptr;
            const StepStateResult res =
                step_state(body, food, heading, walls, w, h, a);
            body = res.body;
            food = res.food;
            heading = res.heading;
            dead = res.dead;
            if (res.ate) {
                path_value += disc;
            }
            disc *= gamma;
        }
        // --- simulation ---
        double value = 0.0;
        if (dead) {
            value = path_value;
        } else {
            value = path_value +
                    rollout(body, food, heading, walls, w, h, disc, gamma);
        }
        // --- backprop ---
        for (Node* n = node; n != nullptr; n = n->parent) {
            n->visits += 1;
            n->value += value;
        }
    }

    // Most-visited root child wins (mcts.py lines 200-206).
    MctsSearchResult result;
    int best_a = kStraight;
    long long best_v = -1;
    for (const int a : root.child_order) {
        const Node& ch = *root.children[a];
        result.root_children.push_back(
            MctsRootChild{a, ch.visits, ch.value});
        if (ch.visits > best_v) {
            best_v = ch.visits;
            best_a = a;
        }
    }
    result.action = best_a;
    return result;
}

int MCTSPlayer::search(const MctsState& state, const MctsWalls& walls) {
    return search_with_stats(state, walls).action;
}

}  // namespace narcade
