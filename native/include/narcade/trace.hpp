// Native move-level trace instrumentation: the C++ counterpart of
// scripts/analyze.py (the semantics authority), RESCOPED per review:
// this module does NOT re-implement evaluate.py's Elo / head-to-head /
// summary statistics. It instruments the simulation loop itself,
// recording every action, the terminal reason, foods eaten, and steps
// per episode, and aggregates them into analyze.py's exact output
// shape:
//
//   {"actions": [round(f0, 3), round(f1, 3), round(f2, 3)],
//    "deaths": {reason: count, ...},
//    "food_per_100_steps": round(100 * foods / max(1, steps), 2),
//    "episodes": n}
//
// (analyze.py:24-47). Aggregation values are rounded decimals and
// integer counters, so native/Python agreement is exact, not
// tolerance-based.
#ifndef NARCADE_TRACE_HPP
#define NARCADE_TRACE_HPP

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "narcade/snake.hpp"

namespace narcade {

// CPython round(x, ndigits) for doubles: correctly rounded decimal
// rounding of the exact binary value, ties to even. Implemented via
// printf %.*f (glibc formats the exact binary value with the current
// round-to-nearest-even mode) + strtod, which reproduces CPython's
// _Py_dg_dtoa-based round on the values this module produces.
double python_round(double x, int ndigits);

// One episode's move-level trace, recorded inside the step loop.
struct EpisodeTrace {
    std::vector<int> actions;  // every action taken, in order
    // Terminal info["reason"] ("death" / "win" / "timeout"), or "?"
    // when the terminal info has no reason — analyze.py:40's
    // info.get("reason", "?").
    std::string reason = "?";
    int foods = 0;         // env.score at termination
    std::int64_t steps = 0;
};

// Runs one episode on an already-constructed env, mirroring
// analyze.py:29-40: reset() first (the env constructor already reset
// once — snake.py:33 — and analyze.py resets again, line 31), then
// per step: ask the policy, RECORD the action, step, count the step.
template <typename ActFn>
EpisodeTrace run_traced_episode(SnakeEnv& env, ActFn&& act) {
    EpisodeTrace trace;
    env.reset();
    bool done = false;
    while (!done) {
        const int a = act(env);
        trace.actions.push_back(a);
        const StepResult step = env.step(a);
        ++trace.steps;
        done = step.done;
        if (step.done) {
            trace.reason =
                step.reason.has_value() ? *step.reason : "?";
        }
    }
    trace.foods = env.score();
    return trace;
}

// Aggregator over episode traces, mirroring analyze.py:25-47's
// Counters and totals.
class MoveStats {
public:
    void add(const EpisodeTrace& trace);

    // analyze.py:41-47's return dict, as data.
    struct Summary {
        std::array<double, 3> actions{};  // rounded to 3 decimals
        // Death counts in first-occurrence order, like the Python
        // Counter's dict ordering.
        std::vector<std::pair<std::string, std::int64_t>> deaths;
        double food_per_100_steps = 0.0;  // rounded to 2 decimals
        int episodes = 0;
    };
    Summary summarize() const;

    std::int64_t total_actions() const {
        return actions_[0] + actions_[1] + actions_[2];
    }

private:
    std::array<std::int64_t, 3> actions_{};
    std::vector<std::pair<std::string, std::int64_t>> deaths_;
    std::int64_t foods_ = 0;
    std::int64_t steps_ = 0;
    int episodes_ = 0;
};

using PolicyFn = std::function<int(const SnakeEnv&)>;

// scripts/analyze.py analyze(policy_fn, n, seed0), lines 24-47:
// episodes on SnakeEnv(seed=seed0 + i) at the default 12x12 size,
// aggregated into the summary shape.
MoveStats::Summary analyze_traced(const PolicyFn& policy, int n,
                                  std::int64_t seed0);

}  // namespace narcade

#endif  // NARCADE_TRACE_HPP
