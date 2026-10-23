// Native move-level trace instrumentation: implementation.
// Mirrors scripts/analyze.py lines 24-47; see trace.hpp.
#include "narcade/trace.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace narcade {

double python_round(double x, int ndigits) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", ndigits, x);
    return std::strtod(buf, nullptr);
}

void MoveStats::add(const EpisodeTrace& trace) {
    for (const int a : trace.actions) {
        ++actions_[a];
    }
    // Counter semantics: first occurrence fixes the key order.
    const auto it = std::find_if(
        deaths_.begin(), deaths_.end(),
        [&](const auto& kv) { return kv.first == trace.reason; });
    if (it == deaths_.end()) {
        deaths_.push_back({trace.reason, 1});
    } else {
        ++it->second;
    }
    foods_ += trace.foods;
    steps_ += trace.steps;
    ++episodes_;
}

MoveStats::Summary MoveStats::summarize() const {
    Summary summary;
    // analyze.py:41-44: fractions of the total action count, each
    // rounded to 3 decimals; food efficiency rounded to 2.
    const std::int64_t total_a = total_actions();
    for (int a = 0; a < 3; ++a) {
        summary.actions[a] =
            python_round(actions_[a] / static_cast<double>(total_a), 3);
    }
    summary.deaths = deaths_;
    summary.food_per_100_steps = python_round(
        100.0 * static_cast<double>(foods_) /
            static_cast<double>(std::max<std::int64_t>(1, steps_)),
        2);
    summary.episodes = episodes_;
    return summary;
}

MoveStats::Summary analyze_traced(const PolicyFn& policy, int n,
                                  std::int64_t seed0) {
    MoveStats stats;
    for (int i = 0; i < n; ++i) {
        // analyze.py:29: SnakeEnv(seed=seed0 + i), default 12x12.
        SnakeEnv env(12, 12, seed0 + i);
        stats.add(run_traced_episode(env, policy));
    }
    return stats.summarize();
}

}  // namespace narcade
