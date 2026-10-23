// analyze_main: native counterpart of scripts/analyze.py's main().
// Runs the same policies over the same seeds (analyze.py:59-77) using
// the native move-level instrumentation (narcade/trace.hpp) and
// writes moves.json in analyze.py's output shape.
//
// DQN ("pro") is EXCLUDED, explicitly: native DQN inference is a
// separate module (CPP-06d), so this program writes the four non-DQN
// policies and says so on stderr. It never silently omits it.
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "narcade/mcts.hpp"
#include "narcade/policies.hpp"
#include "narcade/trace.hpp"

namespace {

std::string fmt_double(double v) {
    char buf[40];
    const auto res = std::to_chars(buf, buf + sizeof(buf), v);
    return std::string(buf, res.ptr);
}

std::string join_path(const std::string& dir, const std::string& name) {
    if (!dir.empty() && dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

// Minimal reader for runs/qtable.json: a 32x3 nested array of JSON
// numbers (the shape scripts/train.py writes). Scans number tokens in
// order with strtod (correctly rounded, like Python's json parser).
narcade::QTablePolicy::Table load_qtable(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    std::vector<double> values;
    const char* p = text.c_str();
    while (*p) {
        if ((*p >= '0' && *p <= '9') || *p == '-') {
            char* end = nullptr;
            const double v = std::strtod(p, &end);
            if (end == p) {
                throw std::runtime_error("bad number in " + path);
            }
            values.push_back(v);
            p = end;
        } else {
            ++p;
        }
    }
    if (values.size() != 96) {
        throw std::runtime_error("qtable must have 96 values, got " +
                                 std::to_string(values.size()));
    }
    narcade::QTablePolicy::Table table;
    for (int s = 0; s < 32; ++s) {
        table.push_back({values[3 * s], values[3 * s + 1],
                         values[3 * s + 2]});
    }
    return table;
}

std::string summary_json(const narcade::MoveStats::Summary& s) {
    std::ostringstream os;
    os << "{\"actions\":[" << fmt_double(s.actions[0]) << ","
       << fmt_double(s.actions[1]) << "," << fmt_double(s.actions[2])
       << "],\"deaths\":{";
    bool first = true;
    for (const auto& [reason, count] : s.deaths) {
        if (!first) os << ",";
        first = false;
        os << "\"" << reason << "\":" << count;
    }
    os << "},\"food_per_100_steps\":" << fmt_double(s.food_per_100_steps)
       << ",\"episodes\":" << s.episodes << "}";
    return os.str();
}

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s [--out DIR] [--n N] [--mcts-n N] "
                 "[--mcts-sims N]\n",
                 argv0);
}

}  // namespace

int main(int argc, char** argv) {
    std::string out_dir = "runs";
    int n = 200;         // analyze.py: --n default 200
    int mcts_n = 30;     // analyze.py: --mcts_n default 30
    int mcts_sims = 80;  // analyze.py: --mcts_sims default 80
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                usage(argv[0]);
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--out") {
            out_dir = next("--out");
        } else if (arg == "--n") {
            n = std::stoi(next("--n"));
        } else if (arg == "--mcts-n") {
            mcts_n = std::stoi(next("--mcts-n"));
        } else if (arg == "--mcts-sims") {
            mcts_sims = std::stoi(next("--mcts-sims"));
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    try {
        // analyze.py:62-68: qtable loaded from the out dir; the
        // RandomPolicy(seed=77) instance persists across ALL its
        // episodes (its RNG lifecycle is part of the protocol).
        const auto qtable = load_qtable(join_path(out_dir, "qtable.json"));
        narcade::RandomPolicy random_policy(77);
        const narcade::QTablePolicy rookie(qtable);
        // analyze.py:75: MCTSPlayer(simulations=sims, seed=4242);
        // rollout_steps and c keep mcts.py's defaults (100, 1.0).
        narcade::MCTSPlayer grandmaster(mcts_sims, 100, 1.0, 4242);

        auto act_random = [&random_policy](const narcade::SnakeEnv&) {
            return random_policy.act();
        };
        auto act_greedy = [](const narcade::SnakeEnv& env) {
            return narcade::GreedyPolicy::act(env);
        };
        auto act_rookie = [&rookie](const narcade::SnakeEnv& env) {
            return rookie.act(env);
        };
        auto act_mcts = [&grandmaster](const narcade::SnakeEnv& env) {
            narcade::MctsState state;
            state.width = env.width();
            state.height = env.height();
            state.body.assign(env.body().begin(), env.body().end());
            state.food = env.food().value_or(narcade::Cell{-1, -1});
            state.heading = env.heading();
            return grandmaster.act(state, narcade::MctsWalls{});
        };

        // analyze.py:70-77: fast policies at seed0=31000, grandmaster
        // at seed0=33000 with its own episode count.
        struct Named {
            const char* name;
            narcade::MoveStats::Summary summary;
        };
        std::vector<Named> results;
        std::printf("analyzing random...\n");
        results.push_back({"random", narcade::analyze_traced(act_random, n, 31000)});
        std::printf("analyzing greedy...\n");
        results.push_back({"greedy", narcade::analyze_traced(act_greedy, n, 31000)});
        std::printf("analyzing rookie...\n");
        results.push_back({"rookie", narcade::analyze_traced(act_rookie, n, 31000)});
        std::fprintf(stderr,
                     "note: 'pro' (DQN) excluded: native DQN inference "
                     "is a separate module (CPP-06d)\n");
        std::printf("analyzing grandmaster...\n");
        std::fflush(stdout);
        results.push_back(
            {"grandmaster",
             narcade::analyze_traced(act_mcts, mcts_n, 33000)});

        std::ostringstream os;
        os << "{\n";
        bool first = true;
        for (const auto& r : results) {
            if (!first) os << ",\n";
            first = false;
            os << "  \"" << r.name << "\": " << summary_json(r.summary);
        }
        os << "\n}\n";
        const std::string moves_path = join_path(out_dir, "moves.json");
        std::ofstream out(moves_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("cannot open " + moves_path);
        }
        out << os.str();

        for (const auto& r : results) {
            std::printf("  %-12s actions=[%.3f, %.3f, %.3f] "
                        "food/100steps=%.2f episodes=%d\n",
                        r.name, r.summary.actions[0], r.summary.actions[1],
                        r.summary.actions[2], r.summary.food_per_100_steps,
                        r.summary.episodes);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
