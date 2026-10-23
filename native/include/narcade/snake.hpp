// SnakeEnv: a C++20 port of src/narcade/snake.py (the semantics authority).
//
// Behavioral parity with the Python reference is gated on differential
// testing (the frozen session corpus native/golden/snake_corpus.txt,
// >= 110,000 transitions, zero mismatches; regenerated from the live
// reference by tools/native_reference.py). Every
// ordering quirk of the Python step() is preserved deliberately; see
// snake.cpp for the line-by-line correspondence.
#ifndef NARCADE_SNAKE_HPP
#define NARCADE_SNAKE_HPP

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <utility>

#include "narcade/rng.hpp"

namespace narcade {

// Board cell: (row, col). Row 0 is the top row (snake.py DIRS comment).
struct Cell {
    int r = 0;
    int c = 0;
    bool operator==(const Cell& other) const {
        return r == other.r && c == other.c;
    }
};

// Relative actions (snake.py: STRAIGHT, LEFT, RIGHT = 0, 1, 2).
inline constexpr int kStraight = 0;
inline constexpr int kLeft = 1;
inline constexpr int kRight = 2;

// Mirrors the (state, reward, done, info) tuple returned by SnakeEnv.step.
// `reason` is std::nullopt exactly when Python's info dict has no "reason"
// key (a non-terminal step); otherwise "death", "win", or "timeout".
struct StepResult {
    int state = 0;
    double reward = 0.0;
    bool done = false;
    std::optional<std::string> reason;
};

class SnakeEnv {
public:
    // Reward constants (snake.py class attributes).
    static constexpr double kFoodReward = 10.0;
    static constexpr double kDeathReward = -10.0;
    static constexpr double kCloserReward = 0.5;
    static constexpr double kFartherReward = -0.5;

    // Mirrors SnakeEnv(width, height, seed): max_steps = 20 * width * height.
    SnakeEnv(int width, int height, std::int64_t seed);
    // Mirrors SnakeEnv(width, height, seed, max_steps=max_steps).
    SnakeEnv(int width, int height, std::int64_t seed, std::int64_t max_steps);

    // Rebuild the initial centered body, zero score/steps, place food.
    // Returns encode_state(), like Python reset().
    int reset();

    // The exact 32-state encoding documented in snake.py encode_state().
    int encode_state() const;

    // Advance one step. Throws std::runtime_error if the episode is done
    // (Python RuntimeError, checked first) and std::invalid_argument for
    // an action other than 0/1/2 (Python ValueError). Both exceptions are
    // raised before any state mutation, exactly as in Python.
    StepResult step(int action);

    // Attribute mirrors (Python: env.width, env.body, env.food, ...).
    int width() const { return width_; }
    int height() const { return height_; }
    const std::deque<Cell>& body() const { return body_; }
    std::optional<Cell> food() const { return food_; }
    int heading() const { return heading_; }
    int score() const { return score_; }
    std::int64_t steps() const { return steps_; }
    bool done() const { return done_; }
    std::int64_t max_steps() const { return max_steps_; }

private:
    void place_food();
    bool cell_blocked(int r, int c, bool ignore_tail) const;

    int width_;
    int height_;
    PythonRandom rng_;
    std::int64_t max_steps_;
    std::deque<Cell> body_;  // front = tail (body[0]), back = head (body[-1])
    std::optional<Cell> food_;
    int heading_ = 1;  // 0=up, 1=right, 2=down, 3=left
    int score_ = 0;
    std::int64_t steps_ = 0;
    bool done_ = false;
};

}  // namespace narcade

#endif  // NARCADE_SNAKE_HPP
