// snake_diff_runner: batch differential-test endpoint for the C++ Snake
// simulator. Driven by tools/native_fuzz.py, which runs the Python
// SnakeEnv over the same episode specs and compares every field.
//
// Protocol (line-based text, stdin -> stdout):
//   Input, per episode:
//     EPISODE <width> <height> <seed> <max_steps|-1> <n_actions>
//     <action>                      (n_actions lines, one integer each)
//   Output, per episode:
//     INIT <state> <food> <body>
//     then one line per action:
//       S <state_before> <state_after> <reward_hex> <done> <reason_code>
//         <food> <score> <steps> <body>
//       X runtime_error | X invalid_argument   (when step() raised)
//     DONE
//   <food> is "r,c" or "-1,-1" when there is no food. <body> is "r,c"
//   pairs joined by ';', tail first. <reward_hex> is the IEEE754 double
//   bit pattern as 16 lowercase hex digits (same encoding as
//   tools/native_reference.py). reason_code: 0 none, 1 death, 2 win,
//   3 timeout.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "narcade/snake.hpp"

namespace {

std::string reward_hex(double v) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx",
                  static_cast<unsigned long long>(bits));
    return buf;
}

std::string food_str(const std::optional<narcade::Cell>& food) {
    if (!food.has_value()) {
        return "-1,-1";
    }
    return std::to_string(food->r) + "," + std::to_string(food->c);
}

std::string body_str(const std::deque<narcade::Cell>& body) {
    std::string out;
    bool first = true;
    for (const narcade::Cell& cell : body) {
        if (!first) {
            out += ";";
        }
        first = false;
        out += std::to_string(cell.r) + "," + std::to_string(cell.c);
    }
    return out;
}

int reason_code(const std::optional<std::string>& reason) {
    if (!reason.has_value()) {
        return 0;
    }
    if (*reason == "death") {
        return 1;
    }
    if (*reason == "win") {
        return 2;
    }
    return 3;  // timeout
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.rfind("EPISODE", 0) != 0) {
            continue;
        }
        std::istringstream header(line);
        std::string tag;
        int width = 0;
        int height = 0;
        std::int64_t seed = 0;
        std::int64_t max_steps = -1;
        int n_actions = 0;
        header >> tag >> width >> height >> seed >> max_steps >> n_actions;

        narcade::SnakeEnv env = max_steps >= 0
            ? narcade::SnakeEnv(width, height, seed, max_steps)
            : narcade::SnakeEnv(width, height, seed);
        std::cout << "INIT " << env.encode_state() << ' '
                  << food_str(env.food()) << ' ' << body_str(env.body())
                  << "\n";
        for (int i = 0; i < n_actions; ++i) {
            if (!std::getline(std::cin, line)) {
                return 1;
            }
            const int action = std::stoi(line);
            const int state_before = env.encode_state();
            try {
                const narcade::StepResult res = env.step(action);
                std::cout << "S " << state_before << ' ' << res.state << ' '
                          << reward_hex(res.reward) << ' '
                          << (res.done ? 1 : 0) << ' '
                          << reason_code(res.reason) << ' '
                          << food_str(env.food()) << ' ' << env.score() << ' '
                          << env.steps() << ' ' << body_str(env.body())
                          << "\n";
            } catch (const std::runtime_error&) {
                std::cout << "X runtime_error\n";
            } catch (const std::invalid_argument&) {
                std::cout << "X invalid_argument\n";
            }
        }
        std::cout << "DONE\n";
        std::cout.flush();
    }
    return 0;
}
