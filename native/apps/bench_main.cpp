// bench_main: native benchmark suite (CPP-06c).
//
// Fixed, deterministic workloads over the C++ engine, timed with
// std::chrono::steady_clock. The numbers are for the record only; no
// performance claim is published from them. Every workload prints its
// size, wall time, rate, and a checksum of the results so two runs
// (or two machines) can confirm they executed identical work.
//
// Subcommands:
//   env   SnakeEnv steps/sec: episodes on 12x12 with GreedyPolicy,
//         seeds 1000, 1001, ... until the step budget is met
//         (default budget 1,000,000 steps). Includes env construction.
//   rng   PythonRandom random() draws/sec (default 10,000,000 draws,
//         seed 12345). The printed sum is the checksum.
//   mcts  MCTSPlayer searches/sec at simulations=80 on one fixed
//         mid-game state (a 12x12 position reached by playing seed
//         4242 with GreedyPolicy for up to 60 steps). A fresh player
//         (seed 31337, rollout_steps=100, c=1.0) per search, so every
//         search does identical work. Default 200 searches.
//   eval  Full episodes/sec for the greedy and qtable players on the
//         scripts/evaluate.py protocol shape: SnakeEnv(12, 12, seed)
//         + reset(), seeds 5000..5000+n-1, step until done. The
//         qtable player loads runs/qtable.json (override with
//         --qtable). Default 200 episodes per player.
//   all   Run all four with default budgets.
//
// Options: --steps N --draws N --searches N --episodes N --qtable PATH
#include <chrono>
#include <cstdint>
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
#include "narcade/rng.hpp"
#include "narcade/snake.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_s(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double>(t1 - t0).count();
}

// Minimal JSON reader for one shape only: a bare nested array of
// numbers (runs/qtable.json is a 32x3 list of lists). Extracts every
// numeric token in document order; the caller checks the count.
std::vector<double> scan_json_numbers(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    std::vector<double> nums;
    const char* p = text.c_str();
    while (*p) {
        const char c = *p;
        if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.') {
            char* end = nullptr;
            const double v = std::strtod(p, &end);
            if (end == p) {
                ++p;
                continue;
            }
            nums.push_back(v);
            p = end;
        } else {
            ++p;
        }
    }
    return nums;
}

narcade::QTablePolicy::Table load_qtable(const std::string& path) {
    const std::vector<double> nums = scan_json_numbers(path);
    if (nums.size() != 32u * 3u) {
        throw std::runtime_error("qtable: expected 96 numbers, got " +
                                 std::to_string(nums.size()));
    }
    narcade::QTablePolicy::Table t(32, std::vector<double>(3, 0.0));
    for (int s = 0; s < 32; ++s) {
        for (int a = 0; a < 3; ++a) {
            t[s][a] = nums[static_cast<std::size_t>(s * 3 + a)];
        }
    }
    return t;
}

int bench_env(std::int64_t step_budget) {
    std::int64_t steps = 0;
    std::int64_t score_sum = 0;
    std::int64_t episodes = 0;
    const auto t0 = Clock::now();
    for (std::int64_t seed = 1000; steps < step_budget; ++seed) {
        narcade::SnakeEnv env(12, 12, seed);
        env.reset();
        while (!env.done()) {
            env.step(narcade::GreedyPolicy::act(env));
            ++steps;
        }
        score_sum += env.score();
        ++episodes;
    }
    const auto t1 = Clock::now();
    const double dt = elapsed_s(t0, t1);
    std::printf("env: steps=%lld episodes=%lld wall=%.3fs rate=%.0f steps/s "
                "checksum(score_sum)=%lld\n",
                static_cast<long long>(steps),
                static_cast<long long>(episodes), dt,
                static_cast<double>(steps) / dt,
                static_cast<long long>(score_sum));
    return 0;
}

int bench_rng(std::int64_t draws) {
    narcade::PythonRandom rng(12345);
    double sum = 0.0;
    const auto t0 = Clock::now();
    for (std::int64_t i = 0; i < draws; ++i) {
        sum += rng.random();
    }
    const auto t1 = Clock::now();
    const double dt = elapsed_s(t0, t1);
    std::printf("rng: draws=%lld wall=%.3fs rate=%.0f draws/s "
                "checksum(sum)=%.6f\n",
                static_cast<long long>(draws), dt,
                static_cast<double>(draws) / dt, sum);
    return 0;
}

// The fixed mid-game state for the mcts workload: play seed 4242 with
// GreedyPolicy for up to 60 steps and snapshot the position.
narcade::MctsState midgame_state() {
    narcade::SnakeEnv env(12, 12, 4242);
    env.reset();
    for (int i = 0; i < 60 && !env.done(); ++i) {
        env.step(narcade::GreedyPolicy::act(env));
    }
    narcade::MctsState st;
    st.width = env.width();
    st.height = env.height();
    st.body.assign(env.body().begin(), env.body().end());
    st.food = env.food().value_or(narcade::Cell{-1, -1});
    st.heading = env.heading();
    return st;
}

int bench_mcts(std::int64_t searches) {
    const narcade::MctsState st = midgame_state();
    const narcade::MctsWalls walls;
    long long action_sum = 0;
    const auto t0 = Clock::now();
    for (std::int64_t i = 0; i < searches; ++i) {
        narcade::MCTSPlayer player(80, 100, 1.0, 31337);
        action_sum += player.search(st, walls);
    }
    const auto t1 = Clock::now();
    const double dt = elapsed_s(t0, t1);
    std::printf("mcts: searches=%lld sims=80 body_len=%zu wall=%.3fs "
                "rate=%.1f searches/s checksum(action_sum)=%lld\n",
                static_cast<long long>(searches), st.body.size(), dt,
                static_cast<double>(searches) / dt, action_sum);
    return 0;
}

// One run_episodes pass in the scripts/evaluate.py shape. act_fn is
// called on the live env before every step, exactly like policy_fn.
template <typename ActFn>
void run_eval_player(const char* name, ActFn act_fn, int episodes) {
    std::int64_t steps = 0;
    std::int64_t score_sum = 0;
    const auto t0 = Clock::now();
    for (int i = 0; i < episodes; ++i) {
        narcade::SnakeEnv env(12, 12, 5000 + i);
        env.reset();
        while (!env.done()) {
            env.step(act_fn(env));
            ++steps;
        }
        score_sum += env.score();
    }
    const auto t1 = Clock::now();
    const double dt = elapsed_s(t0, t1);
    std::printf("eval %s: episodes=%d steps=%lld wall=%.3fs rate=%.1f "
                "episodes/s (%.0f steps/s) checksum(score_sum)=%lld\n",
                name, episodes, static_cast<long long>(steps), dt,
                static_cast<double>(episodes) / dt,
                static_cast<double>(steps) / dt,
                static_cast<long long>(score_sum));
}

int bench_eval(int episodes, const std::string& qtable_path) {
    run_eval_player("greedy",
                    [](const narcade::SnakeEnv& env) {
                        return narcade::GreedyPolicy::act(env);
                    },
                    episodes);
    const narcade::QTablePolicy qpol(load_qtable(qtable_path));
    run_eval_player("qtable",
                    [&qpol](const narcade::SnakeEnv& env) {
                        return qpol.act(env);
                    },
                    episodes);
    return 0;
}

std::int64_t parse_i64(const char* s) {
    return std::strtoll(s, nullptr, 10);
}

}  // namespace

int main(int argc, char** argv) {
    std::string cmd = argc > 1 ? argv[1] : "all";
    std::int64_t steps = 1000000;
    std::int64_t draws = 10000000;
    std::int64_t searches = 200;
    int episodes = 200;
    std::string qtable = "runs/qtable.json";
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string flag = argv[i];
        if (flag == "--steps") {
            steps = parse_i64(argv[i + 1]);
        } else if (flag == "--draws") {
            draws = parse_i64(argv[i + 1]);
        } else if (flag == "--searches") {
            searches = parse_i64(argv[i + 1]);
        } else if (flag == "--episodes") {
            episodes = static_cast<int>(parse_i64(argv[i + 1]));
        } else if (flag == "--qtable") {
            qtable = argv[i + 1];
        } else {
            std::fprintf(stderr, "unknown option: %s\n", flag.c_str());
            return 2;
        }
    }
    try {
        if (cmd == "env") {
            return bench_env(steps);
        }
        if (cmd == "rng") {
            return bench_rng(draws);
        }
        if (cmd == "mcts") {
            return bench_mcts(searches);
        }
        if (cmd == "eval") {
            return bench_eval(episodes, qtable);
        }
        if (cmd == "all") {
            bench_env(steps);
            bench_rng(draws);
            bench_mcts(searches);
            bench_eval(episodes, qtable);
            return 0;
        }
        std::fprintf(stderr, "unknown subcommand: %s\n", cmd.c_str());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench error: %s\n", e.what());
        return 1;
    }
}
