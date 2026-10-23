// mcts_diff_runner: batch differential-test endpoint for the C++
// MCTSPlayer (CPP-03). Driven by tools/native_mcts_fuzz.py, which
// runs the Python reference over the same states; every compared
// value must agree.
//
// Protocol (line-based text, stdin -> stdout). Doubles travel as 16
// lowercase hex digits holding the IEEE754 bit pattern.
//
//   PLAYER <simulations> <rollout_steps> <c_hex> <seed>
//     Replaces the runner's persistent MCTSPlayer (its RNG state
//     carries across SEARCH calls, like the Python lifecycle).
//     Replies "OK".
//   ENV <width> <height> <seed>
//     Creates the runner's persistent SnakeEnv (default max_steps).
//     Replies "INIT <state> <food> <body>".
//   SEARCH
//     Searches the persistent env's current state with the
//     persistent player. Replies "ACTION <a>", "ROOT <k>", then k
//     lines "CHILD <action> <visits> <value_hex>" in insertion order.
//   STEP <action>
//     Steps the persistent env. Replies the snake S/X line:
//       "S <state_before> <state_after> <reward_hex> <done> <reason>
//        <food> <score> <steps> <body>" or "X runtime_error" /
//        "X invalid_argument".
//   SEARCHSTATE <w> <h> <food_r> <food_c> <heading> <n_body>
//     then n_body lines "<r> <c>" (tail first). One-shot search on
//     the serialized state with the persistent player; replies
//     ACTION/ROOT/CHILD lines as for SEARCH.
//   <food> is "r,c" or "-1,-1"; <body> is "r,c" pairs joined by ';',
//   tail first; reason codes: 0 none, 1 death, 2 win, 3 timeout.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

#include "narcade/mcts.hpp"
#include "narcade/snake.hpp"

namespace {

std::string double_hex(double v) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx",
                  static_cast<unsigned long long>(bits));
    return buf;
}

double parse_double_hex(const std::string& tok) {
    const std::uint64_t bits = std::stoull(tok, nullptr, 16);
    double v = 0.0;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

std::string food_str(const std::optional<narcade::Cell>& food) {
    if (!food.has_value()) {
        return "-1,-1";
    }
    return std::to_string(food->r) + "," + std::to_string(food->c);
}

template <typename Container>
std::string body_str(const Container& body) {
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

void write_search_result(const narcade::MctsSearchResult& res) {
    std::cout << "ACTION " << res.action << "\n";
    std::cout << "ROOT " << res.root_children.size() << "\n";
    for (const narcade::MctsRootChild& ch : res.root_children) {
        std::cout << "CHILD " << ch.action << ' ' << ch.visits << ' '
                  << double_hex(ch.value) << "\n";
    }
    std::cout.flush();
}

narcade::MctsState env_state(const narcade::SnakeEnv& env) {
    narcade::MctsState state;
    state.width = env.width();
    state.height = env.height();
    state.body.assign(env.body().begin(), env.body().end());
    // A live env always has food unless the board is full (terminal
    // win); search on such a state is outside the Python contract
    // (it would index None), so map it to the (-1,-1) sentinel.
    state.food = env.food().value_or(narcade::Cell{-1, -1});
    state.heading = env.heading();
    return state;
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::optional<narcade::MCTSPlayer> player;
    std::optional<narcade::SnakeEnv> env;
    const narcade::MctsWalls no_walls;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream header(line);
        std::string tag;
        header >> tag;
        if (tag == "PLAYER") {
            int simulations = 0, rollout_steps = 0;
            std::string c_hex;
            std::int64_t seed = 0;
            header >> simulations >> rollout_steps >> c_hex >> seed;
            player.emplace(simulations, rollout_steps,
                           parse_double_hex(c_hex), seed);
            std::cout << "OK\n";
            std::cout.flush();
        } else if (tag == "ENV") {
            int width = 0, height = 0;
            std::int64_t seed = 0;
            header >> width >> height >> seed;
            env.emplace(width, height, seed);
            std::cout << "INIT " << env->encode_state() << ' '
                      << food_str(env->food()) << ' '
                      << body_str(env->body()) << "\n";
            std::cout.flush();
        } else if (tag == "SEARCH") {
            if (!player.has_value() || !env.has_value()) {
                return 1;
            }
            write_search_result(
                player->search_with_stats(env_state(*env), no_walls));
        } else if (tag == "STEP") {
            if (!env.has_value()) {
                return 1;
            }
            int action = 0;
            header >> action;
            const int state_before = env->encode_state();
            try {
                const narcade::StepResult res = env->step(action);
                std::cout << "S " << state_before << ' ' << res.state
                          << ' ' << double_hex(res.reward) << ' '
                          << (res.done ? 1 : 0) << ' '
                          << reason_code(res.reason) << ' '
                          << food_str(env->food()) << ' ' << env->score()
                          << ' ' << env->steps() << ' '
                          << body_str(env->body()) << "\n";
            } catch (const std::runtime_error&) {
                std::cout << "X runtime_error\n";
            } catch (const std::invalid_argument&) {
                std::cout << "X invalid_argument\n";
            }
            std::cout.flush();
        } else if (tag == "SEARCHSTATE") {
            if (!player.has_value()) {
                return 1;
            }
            narcade::MctsState state;
            int fr = -1, fc = -1, n_body = 0;
            header >> state.width >> state.height >> fr >> fc >>
                state.heading >> n_body;
            state.food = narcade::Cell{fr, fc};
            for (int i = 0; i < n_body; ++i) {
                if (!std::getline(std::cin, line)) {
                    return 1;
                }
                std::istringstream cell_line(line);
                narcade::Cell cell;
                cell_line >> cell.r >> cell.c;
                state.body.push_back(cell);
            }
            write_search_result(
                player->search_with_stats(state, no_walls));
        }
        // Unknown lines are ignored, mirroring the other runners.
    }
    return 0;
}
