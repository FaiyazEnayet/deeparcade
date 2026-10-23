// policies_diff_runner: batch differential-test endpoint for the C++
// policies and tabular Q-learning agent (CPP-02). Driven by
// tools/native_policies_fuzz.py, which parses the golden fixture
// (native/golden/q_updates.json) and runs the Python reference over the
// same inputs; every compared value must agree.
//
// Protocol (line-based text, stdin -> stdout). Doubles travel as 16
// lowercase hex digits holding the IEEE754 bit pattern (the encoding
// used by tools/native_reference.py and snake_diff_runner).
//
//   QUPDATES <alpha_hex> <gamma_hex> <epsilon_hex> <seed> <n>
//     then n lines:  U <s> <a> <r_hex> <s_next> <done>
//     replies: "QRESULT" then 32 lines of "<hex> <hex> <hex>".
//   QACT <seed> <epsilon_hex>
//     then: "QTABLE" + 32 table lines, "NSTATES <n>" + n state lines.
//     replies: "ACTIONS" then one line with the n actions.
//     (alpha/gamma take the Python class defaults 0.15/0.9, as in the
//     fixture's agent construction; they do not affect act().)
//   SETTABLE
//     then 32 table lines. Sets the runner's QTablePolicy table used by
//     BESTACTION and EPISODE. No reply.
//   BESTACTION
//     replies: "BEST" then one line with the 32 best actions.
//   RANDSEED <seed> <n>
//     replies: "RAND" then one line with n RandomPolicy actions.
//   EPISODE <width> <height> <seed> <max_steps|-1> <n_actions>
//     then n action lines (one integer each).
//     replies: "INIT <state> <food> <body>", then per action a line
//       "P <greedy> <qtable>"   ("P -1 -1" once the episode is done;
//                                policies are evaluated on the state
//                                BEFORE the step)
//       followed by the snake_diff_runner step line ("S ..." / "X ..."),
//     then "DONE".
//   <food> is "r,c" or "-1,-1"; <body> is "r,c" pairs joined by ';',
//   tail first; reason codes: 0 none, 1 death, 2 win, 3 timeout.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "narcade/policies.hpp"
#include "narcade/qlearn.hpp"
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

std::string join_ints(const std::vector<int>& values) {
    std::string out;
    bool first = true;
    for (int v : values) {
        if (!first) {
            out += ' ';
        }
        first = false;
        out += std::to_string(v);
    }
    return out;
}

// Reads 32 lines of "<hex> <hex> <hex>" into a 32x3 table.
narcade::QTablePolicy::Table read_table() {
    narcade::QTablePolicy::Table table;
    std::string line;
    for (int s = 0; s < 32; ++s) {
        if (!std::getline(std::cin, line)) {
            throw std::runtime_error("unexpected EOF in table");
        }
        std::istringstream row(line);
        std::string t0, t1, t2;
        row >> t0 >> t1 >> t2;
        table.push_back({parse_double_hex(t0), parse_double_hex(t1),
                         parse_double_hex(t2)});
    }
    return table;
}

void write_table(const narcade::QLearningAgent::Table& table) {
    for (const std::vector<double>& row : table) {
        std::cout << double_hex(row[0]) << ' ' << double_hex(row[1]) << ' '
                  << double_hex(row[2]) << "\n";
    }
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    narcade::QTablePolicy current_table(
        narcade::QTablePolicy::Table(32, std::vector<double>(3, 0.0)));
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream header(line);
        std::string tag;
        header >> tag;
        if (tag == "QUPDATES") {
            std::string alpha_hex, gamma_hex, epsilon_hex;
            std::int64_t seed = 0;
            int n = 0;
            header >> alpha_hex >> gamma_hex >> epsilon_hex >> seed >> n;
            narcade::QLearningAgent agent(
                32, 3, parse_double_hex(alpha_hex),
                parse_double_hex(gamma_hex), parse_double_hex(epsilon_hex),
                seed);
            for (int i = 0; i < n; ++i) {
                if (!std::getline(std::cin, line)) {
                    return 1;
                }
                std::istringstream u(line);
                std::string utag, r_hex;
                int s = 0, a = 0, s_next = 0, done = 0;
                u >> utag >> s >> a >> r_hex >> s_next >> done;
                agent.update(s, a, parse_double_hex(r_hex), s_next,
                             done != 0);
            }
            std::cout << "QRESULT\n";
            write_table(agent.q());
            std::cout.flush();
        } else if (tag == "QACT") {
            std::string epsilon_hex;
            std::int64_t seed = 0;
            header >> seed >> epsilon_hex;
            if (!std::getline(std::cin, line)) {  // "QTABLE"
                return 1;
            }
            const narcade::QLearningAgent::Table table = read_table();
            narcade::QLearningAgent agent(32, 3, 0.15, 0.9,
                                          parse_double_hex(epsilon_hex),
                                          seed);
            agent.set_q(table);
            if (!std::getline(std::cin, line)) {  // "NSTATES <n>"
                return 1;
            }
            std::istringstream ns(line);
            std::string ntag;
            int n = 0;
            ns >> ntag >> n;
            std::vector<int> actions;
            for (int i = 0; i < n; ++i) {
                if (!std::getline(std::cin, line)) {
                    return 1;
                }
                actions.push_back(agent.act(std::stoi(line)));
            }
            std::cout << "ACTIONS\n" << join_ints(actions) << "\n";
            std::cout.flush();
        } else if (tag == "SETTABLE") {
            current_table = narcade::QTablePolicy(read_table());
        } else if (tag == "BESTACTION") {
            std::cout << "BEST\n"
                      << join_ints(current_table.best_action()) << "\n";
            std::cout.flush();
        } else if (tag == "RANDSEED") {
            std::int64_t seed = 0;
            int n = 0;
            header >> seed >> n;
            narcade::RandomPolicy policy(seed);
            std::vector<int> actions;
            for (int i = 0; i < n; ++i) {
                actions.push_back(policy.act());
            }
            std::cout << "RAND\n" << join_ints(actions) << "\n";
            std::cout.flush();
        } else if (tag == "EPISODE") {
            int width = 0, height = 0, n_actions = 0;
            std::int64_t seed = 0, max_steps = -1;
            header >> width >> height >> seed >> max_steps >> n_actions;
            narcade::SnakeEnv env = max_steps >= 0
                ? narcade::SnakeEnv(width, height, seed, max_steps)
                : narcade::SnakeEnv(width, height, seed);
            std::cout << "INIT " << env.encode_state() << ' '
                      << food_str(env.food()) << ' '
                      << body_str(env.body()) << "\n";
            for (int i = 0; i < n_actions; ++i) {
                if (!std::getline(std::cin, line)) {
                    return 1;
                }
                const int action = std::stoi(line);
                if (env.done()) {
                    std::cout << "P -1 -1\n";
                } else {
                    std::cout << "P "
                              << narcade::GreedyPolicy::act(env) << ' '
                              << current_table.act(env) << "\n";
                }
                const int state_before = env.encode_state();
                try {
                    const narcade::StepResult res = env.step(action);
                    std::cout << "S " << state_before << ' ' << res.state
                              << ' ' << double_hex(res.reward) << ' '
                              << (res.done ? 1 : 0) << ' '
                              << reason_code(res.reason) << ' '
                              << food_str(env.food()) << ' ' << env.score()
                              << ' ' << env.steps() << ' '
                              << body_str(env.body()) << "\n";
                } catch (const std::runtime_error&) {
                    std::cout << "X runtime_error\n";
                } catch (const std::invalid_argument&) {
                    std::cout << "X invalid_argument\n";
                }
            }
            std::cout << "DONE\n";
            std::cout.flush();
        }
        // Unknown lines are ignored, mirroring snake_diff_runner.
    }
    return 0;
}
