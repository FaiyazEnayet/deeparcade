// Evaluation engine implementation. Every function mirrors
// scripts/evaluate.py operation by operation; see evaluate.hpp for
// the scope declaration and the per-function line citations.
#include "narcade/evaluate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace narcade {

double py_round(double x, int ndigits) {
    // Python's round(x, n) is the correctly rounded decimal rounding
    // of the exact binary value (round-half-even). glibc's printf
    // %.*f is likewise correctly rounded, and strtod returns the
    // double nearest the decimal string — the same double Python
    // returns.
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%.*f", ndigits, x);
    return std::strtod(buf, nullptr);
}

std::int64_t py_round_int(double x) {
    // Python's round(float) with no ndigits: round-half-even to an
    // integer. nearbyint under the default FP rounding mode
    // (FE_TONEAREST) is exactly round-half-even.
    return static_cast<std::int64_t>(std::nearbyint(x));
}

int MctsPlayer::act(SnakeEnv& env) {
    MctsState state;
    state.width = env.width();
    state.height = env.height();
    state.body.assign(env.body().begin(), env.body().end());
    // Python reads env.food, a tuple while the episode is live; the
    // (-1, -1) sentinel only occurs after a full-board win, when
    // search is never called again.
    state.food = env.food().value_or(Cell{-1, -1});
    state.heading = env.heading();
    return player_.search(state, MctsWalls{});
}

std::unique_ptr<Player> make_player(const std::string& canonical_name,
                                    int mcts_simulations,
                                    const QTablePolicy::Table* qtable) {
    if (canonical_name == "random") {
        return std::make_unique<RandomPlayer>(999);
    }
    if (canonical_name == "greedy") {
        return std::make_unique<GreedyPlayer>();
    }
    if (canonical_name == "rookie") {
        if (qtable == nullptr) {
            throw std::runtime_error("rookie player requires a Q-table");
        }
        return std::make_unique<QTablePlayer>(*qtable);
    }
    if (canonical_name == "grandmaster") {
        return std::make_unique<MctsPlayer>(mcts_simulations, 31337);
    }
    throw std::runtime_error("unknown player: " + canonical_name);
}

QTablePolicy::Table load_qtable_json(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open Q-table file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    std::vector<double> values;
    const char* p = text.c_str();
    while (*p != '\0') {
        char* end = nullptr;
        const double v = std::strtod(p, &end);
        if (end != p) {
            values.push_back(v);
            p = end;
            continue;
        }
        if (*p == '[' || *p == ']' || *p == ',' || *p == ' ' || *p == '\t' ||
            *p == '\n' || *p == '\r') {
            ++p;
            continue;
        }
        throw std::runtime_error("malformed Q-table JSON near: " +
                                 std::string(p, 16));
    }
    if (values.size() != 32 * 3) {
        throw std::runtime_error("Q-table must hold 32x3 values, got " +
                                 std::to_string(values.size()));
    }
    QTablePolicy::Table table(32, std::vector<double>(3, 0.0));
    for (std::size_t s = 0; s < 32; ++s) {
        for (std::size_t a = 0; a < 3; ++a) {
            table[s][a] = values[s * 3 + a];
        }
    }
    return table;
}

std::vector<EpisodeResult> run_episodes(Player& player,
                                        const std::vector<std::int64_t>& seeds,
                                        int width, int height) {
    std::vector<EpisodeResult> out;
    out.reserve(seeds.size());
    for (const std::int64_t seed : seeds) {
        // evaluate.py lines 47-53: SnakeEnv(width, height, seed) —
        // the constructor itself calls reset() (snake.py line 32) —
        // then env.reset() is called AGAIN (the double reset), then
        // step(policy_fn(env)) until done; record env.score.
        SnakeEnv env(width, height, seed);
        env.reset();
        bool done = false;
        while (!done) {
            done = env.step(player.act(env)).done;
        }
        out.push_back(EpisodeResult{env.score(), env.steps()});
    }
    return out;
}

Summary summarize(const std::vector<int>& scores) {
    // evaluate.py lines 57-62.
    Summary s;
    s.n = static_cast<std::int64_t>(scores.size());
    std::int64_t sum = 0;
    s.max = scores[0];
    s.min = scores[0];
    for (const int v : scores) {
        sum += v;
        s.max = std::max(s.max, v);
        s.min = std::min(s.min, v);
    }
    // statistics.mean over ints is the exact rational sum/n; the
    // double division of the exact integer sum is correctly rounded
    // to the same double.
    s.raw_mean = static_cast<double>(sum) / static_cast<double>(s.n);
    if (s.n > 1) {
        // statistics.pstdev: sqrt of the exact population variance.
        // Computed in long double from the exact mean; the result
        // agrees with Python's exact-arithmetic value far below
        // 1e-12 on these magnitudes.
        const long double mean = static_cast<long double>(sum) /
                                 static_cast<long double>(s.n);
        long double ss = 0.0L;
        for (const int v : scores) {
            const long double d = static_cast<long double>(v) - mean;
            ss += d * d;
        }
        const long double var = ss / static_cast<long double>(s.n);
        s.raw_std = static_cast<double>(std::sqrt(var));
    } else {
        s.raw_std = 0.0;  // line 60: std is 0.0 when len(scores) <= 1
    }
    s.mean = py_round(s.raw_mean, 2);
    s.std = py_round(s.raw_std, 2);
    std::vector<int> sorted = scores;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t mid = sorted.size() / 2;
    if (sorted.size() % 2 == 1) {
        s.median = static_cast<double>(sorted[mid]);
    } else {
        // statistics.median for even n: (a + b) / 2.
        s.median = (static_cast<double>(sorted[mid - 1]) +
                    static_cast<double>(sorted[mid])) /
                   2.0;
    }
    return s;
}

std::vector<std::pair<std::string, double>> elo_ratings_raw(
    const OrderedResults& results, double k) {
    // evaluate.py lines 65-78. names keep the results' order; ratings
    // start at 1200.0 (line 68); n is the length of the FIRST player's
    // score list (line 69); the triple loop (lines 70-77) updates both
    // ratings in place before the next pair, with line 77's exact
    // expression ((1 - ra) - (1 - ea)).
    std::vector<double> elo(results.size(), 1200.0);
    const std::size_t n = results[0].second.size();
    const std::size_t m = results.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t a = 0; a < m; ++a) {
            for (std::size_t b = a + 1; b < m; ++b) {
                const int sa = results[a].second[i];
                const int sb = results[b].second[i];
                const double ea =
                    1.0 / (1.0 + std::pow(10.0, (elo[b] - elo[a]) / 400.0));
                const double ra = sa > sb ? 1.0 : (sa == sb ? 0.5 : 0.0);
                elo[a] += k * (ra - ea);
                elo[b] += k * ((1.0 - ra) - (1.0 - ea));
            }
        }
    }
    std::vector<std::pair<std::string, double>> out;
    out.reserve(results.size());
    for (std::size_t i = 0; i < m; ++i) {
        out.emplace_back(results[i].first, elo[i]);
    }
    return out;
}

std::vector<std::pair<std::string, std::int64_t>> elo_ratings(
    const OrderedResults& results, double k) {
    const auto raw = elo_ratings_raw(results, k);
    std::vector<std::pair<std::string, std::int64_t>> out;
    out.reserve(raw.size());
    for (const auto& kv : raw) {
        out.emplace_back(kv.first, py_round_int(kv.second));  // line 78
    }
    return out;
}

HeadToHeadTable head_to_head(const OrderedResults& results) {
    // evaluate.py lines 81-95: n from the first player's list; the
    // diagonal is None; rates are round(count / n, 3).
    const std::size_t n = results[0].second.size();
    const std::size_t m = results.size();
    HeadToHeadTable table(m, std::vector<std::optional<HeadToHeadCell>>(m));
    for (std::size_t a = 0; a < m; ++a) {
        for (std::size_t b = 0; b < m; ++b) {
            if (a == b) {
                continue;
            }
            HeadToHeadCell cell;
            for (std::size_t i = 0; i < n; ++i) {
                if (results[a].second[i] > results[b].second[i]) {
                    ++cell.wins;
                } else if (results[a].second[i] == results[b].second[i]) {
                    ++cell.draws;
                }
            }
            cell.win_rate =
                py_round(static_cast<double>(cell.wins) /
                             static_cast<double>(n),
                         3);
            cell.draw_rate =
                py_round(static_cast<double>(cell.draws) /
                             static_cast<double>(n),
                         3);
            table[a][b] = cell;
        }
    }
    return table;
}

}  // namespace narcade
