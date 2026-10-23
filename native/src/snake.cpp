// C++20 port of src/narcade/snake.py. The Python file is the semantics
// authority; comments below cite its lines so divergences are auditable.
#include "narcade/snake.hpp"

#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace narcade {
namespace {

// Headings: 0=up, 1=right, 2=down, 3=left (snake.py line 12).
constexpr int kDirs[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};

}  // namespace

SnakeEnv::SnakeEnv(int width, int height, std::int64_t seed)
    : SnakeEnv(width, height, seed,
               static_cast<std::int64_t>(20) * width * height) {}

SnakeEnv::SnakeEnv(int width, int height, std::int64_t seed,
                   std::int64_t max_steps)
    : width_(width),
      height_(height),
      rng_(seed),
      max_steps_(max_steps) {
    reset();  // SnakeEnv.__init__ ends with self.reset() (line 35).
}

int SnakeEnv::reset() {
    // snake.py lines 37-46.
    const int cy = height_ / 2;
    const int cx = width_ / 2;
    body_ = {{cy, cx - 2}, {cy, cx - 1}, {cy, cx}};
    heading_ = 1;  // right
    score_ = 0;
    steps_ = 0;
    done_ = false;
    place_food();
    return encode_state();
}

void SnakeEnv::place_food() {
    // snake.py lines 48-52: free cells in row-major order (r outer,
    // c inner) excluding body cells; food = rng.choice(free) if any free
    // cell exists, else None. No RNG draw when the free list is empty.
    std::vector<char> occupied(
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_),
        0);
    for (const Cell& cell : body_) {
        occupied[static_cast<std::size_t>(cell.r) *
                     static_cast<std::size_t>(width_) +
                 static_cast<std::size_t>(cell.c)] = 1;
    }
    std::vector<Cell> free_cells;
    for (int r = 0; r < height_; ++r) {
        for (int c = 0; c < width_; ++c) {
            if (!occupied[static_cast<std::size_t>(r) *
                              static_cast<std::size_t>(width_) +
                          static_cast<std::size_t>(c)]) {
                free_cells.push_back({r, c});
            }
        }
    }
    if (free_cells.empty()) {
        food_.reset();
        return;
    }
    food_ = free_cells[static_cast<std::size_t>(
        rng_.choice_index(free_cells.size()))];
}

bool SnakeEnv::cell_blocked(int r, int c, bool ignore_tail) const {
    // snake.py lines 54-60: out of bounds is blocked; otherwise membership
    // in the body set with body[0] (the tail) discarded when ignore_tail.
    // Body cells are unique in every reachable state, so a scan with the
    // front element skipped is the same predicate as the set difference.
    if (!(0 <= r && r < height_ && 0 <= c && c < width_)) {
        return true;
    }
    std::size_t i = 0;
    for (const Cell& cell : body_) {
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

int SnakeEnv::encode_state() const {
    // snake.py lines 62-93.
    const Cell head = body_.back();
    const int h = heading_;
    const int* ahead = kDirs[h];
    const int* left = kDirs[(h + 3) % 4];
    const int* right = kDirs[(h + 1) % 4];
    const int ds = cell_blocked(head.r + ahead[0], head.c + ahead[1], true)
                       ? 1
                       : 0;
    const int dl = cell_blocked(head.r + left[0], head.c + left[1], true)
                       ? 1
                       : 0;
    const int dr = cell_blocked(head.r + right[0], head.c + right[1], true)
                       ? 1
                       : 0;

    int food_dir = 0;  // food is None (terminal win): food_dir 0 (line 78).
    if (food_.has_value()) {
        const int dy = food_->r - head.r;
        const int dx = food_->c - head.c;
        int fa = 0;
        int ra = 0;
        if (h == 0) {        // up
            fa = -dy;
            ra = dx;
        } else if (h == 1) {  // right
            fa = dx;
            ra = dy;
        } else if (h == 2) {  // down
            fa = dy;
            ra = -dx;
        } else {  // left
            fa = -dx;
            ra = -dy;
        }
        if (std::abs(fa) >= std::abs(ra)) {
            food_dir = fa > 0 ? 0 : 3;
        } else {
            food_dir = ra > 0 ? 1 : 2;
        }
    }
    return ((ds * 2 + dl) * 2 + dr) * 4 + food_dir;
}

StepResult SnakeEnv::step(int action) {
    // snake.py lines 97-143, in the same order.
    if (done_) {
        // Line 100: RuntimeError, checked before action validation.
        throw std::runtime_error(
            "step() called on finished episode; call reset() first");
    }
    // Lines 102-107: heading updates only for LEFT/RIGHT; any other
    // non-STRAIGHT action raises ValueError with the heading untouched.
    if (action == kLeft) {
        heading_ = (heading_ + 3) % 4;
    } else if (action == kRight) {
        heading_ = (heading_ + 1) % 4;
    } else if (action != kStraight) {
        throw std::invalid_argument("bad action");
    }

    const Cell head = body_.back();
    const int nr = head.r + kDirs[heading_][0];
    const int nc = head.c + kDirs[heading_][1];
    ++steps_;  // line 113: steps increments before collision resolution.

    const bool ate = food_.has_value() && food_->r == nr && food_->c == nc;
    // Line 117: collision against the post-move body; the tail vacates
    // unless we eat (ignore_tail = not ate).
    if (cell_blocked(nr, nc, /*ignore_tail=*/!ate)) {
        done_ = true;
        // Death: body unmoved, heading already updated; encode_state()
        // sees exactly that state, as in Python.
        return {encode_state(), kDeathReward, true, std::string("death")};
    }

    // Line 121: distance to the food position before the move. food is
    // never None in a live (non-terminal) state.
    const int dist_before = std::abs(head.r - food_->r) +
                            std::abs(head.c - food_->c);
    body_.push_back({nr, nc});  // line 122: body.append((nr, nc)).
    double reward;
    if (ate) {
        ++score_;       // line 124.
        place_food();   // line 125: may set food to None on a full board.
        reward = kFoodReward;
        if (!food_.has_value()) {
            // Lines 127-130: board completely filled: win.
            done_ = true;
            return {encode_state(), reward, true, std::string("win")};
        }
    } else {
        body_.pop_front();  // line 132: body.popleft().
        const int dist_after = std::abs(nr - food_->r) +
                               std::abs(nc - food_->c);
        if (dist_after < dist_before) {
            reward = kCloserReward;
        } else if (dist_after > dist_before) {
            reward = kFartherReward;
        } else {
            reward = 0.0;
        }
    }

    if (steps_ >= max_steps_) {
        // Lines 139-142: timeout checked after the move fully resolves;
        // the death and win returns above happen before this check.
        done_ = true;
        return {encode_state(), reward, true, std::string("timeout")};
    }
    return {encode_state(), reward, false, std::nullopt};
}

}  // namespace narcade
