// narcade_evaluate: native CLI for the CPP-05 evaluation engine.
//
// SCOPE: NON-NEURAL evaluator. scripts/evaluate.py also evaluates a
// fifth player, "pro" — a DQN loaded from a PyTorch .pt checkpoint.
// That player is NOT covered here (it requires a separately verified
// native DQN inference module), and this tool says so in its help and
// in the "scope" field of every report. It never silently omits the
// DQN player and never claims full evaluate.py equivalence.
//
// Output protocol (line based; consumed by
// tools/native_eval_fuzz.py):
//   EPISODE <name> <seed> <score> <steps>      one per episode
//   RAW_SUMMARY <name> <mean_hex> <std_hex>    pre-rounding doubles
//   RAW_ELO <name> <elo_hex>                   pre-rounding doubles
//   RAW_H2H <a> <b> <wins> <draws>             integer counts
//   REPORT                                     then one JSON line in
//   evaluate.py's paired schema (paired_seeds, scores_paired,
//   summary_paired, elo, head_to_head, n_eval, mcts_n, mcts_sims)
//   plus the "scope" declaration. Doubles in RAW lines travel as
//   IEEE754 hex bit patterns, like the other native drivers.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "narcade/evaluate.hpp"

namespace {

const char* kScope =
    "non-neural native evaluator (CPP-05): paired protocol only; "
    "covers random, greedy, rookie (Q-table), grandmaster (MCTS). "
    "evaluate.py's pro player (DQN from a PyTorch .pt checkpoint) is "
    "NOT covered, and no DQN results are reported.";

std::string double_hex(double v) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx",
                  static_cast<unsigned long long>(bits));
    return buf;
}

// JSON number: shortest round-trip is unnecessary (the driver parses
// numerically); %.17g always round-trips a double exactly.
std::string json_number(double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (const char ch : s) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            default: out += ch; break;
        }
    }
    return out;
}

void print_usage(const char* argv0) {
    std::cout
        << "usage: " << argv0
        << " [--player NAME]... [--episodes N] [--seed0 S] [--sims M]\n"
        << "       [--qtable PATH] [--width W] [--height H]\n"
        << "Players (canonical evaluate.py names or aliases):\n"
        << "  random, greedy, rookie (alias qtable), grandmaster "
           "(alias mcts)\n"
        << "Defaults: all four players, episodes=50, seed0=5000, "
           "sims=80,\n"
        << "  qtable=runs/qtable.json, width=height=12.\n"
        << "Scope: " << kScope << "\n";
}

std::string canonical(const std::string& name) {
    if (name == "qtable") {
        return "rookie";
    }
    if (name == "mcts") {
        return "grandmaster";
    }
    return name;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> requested;
    int episodes = 50;
    std::int64_t seed0 = 5000;
    int sims = 80;
    std::string qtable_path = "runs/qtable.json";
    int width = 12;
    int height = 12;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << flag << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--player") {
            requested.push_back(canonical(next("--player")));
        } else if (arg == "--episodes") {
            episodes = std::stoi(next("--episodes"));
        } else if (arg == "--seed0") {
            seed0 = std::stoll(next("--seed0"));
        } else if (arg == "--sims") {
            sims = std::stoi(next("--sims"));
        } else if (arg == "--qtable") {
            qtable_path = next("--qtable");
        } else if (arg == "--width") {
            width = std::stoi(next("--width"));
        } else if (arg == "--height") {
            height = std::stoi(next("--height"));
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    // evaluate.py main() evaluates in dict order (lines 104-110) with
    // grandmaster last (line 123); mirror that order for the selected
    // players regardless of flag order.
    const std::vector<std::string> kOrder = {"random", "greedy", "rookie",
                                             "grandmaster"};
    std::vector<std::string> players;
    for (const std::string& name : kOrder) {
        if (requested.empty() ||
            std::find(requested.begin(), requested.end(), name) !=
                requested.end()) {
            players.push_back(name);
        }
    }

    std::vector<std::int64_t> seeds;
    for (int i = 0; i < episodes; ++i) {
        seeds.push_back(seed0 + i);  // evaluate.py line 115
    }

    narcade::QTablePolicy::Table qtable;
    bool qtable_loaded = false;
    narcade::OrderedResults results;
    std::map<std::string, narcade::Summary> summaries;

    try {
        for (const std::string& name : players) {
            if (name == "rookie" && !qtable_loaded) {
                qtable = narcade::load_qtable_json(qtable_path);
                qtable_loaded = true;
            }
            auto player = narcade::make_player(
                name, sims, qtable_loaded ? &qtable : nullptr);
            const auto episodes_out =
                narcade::run_episodes(*player, seeds, width, height);
            std::vector<int> scores;
            scores.reserve(episodes_out.size());
            for (std::size_t i = 0; i < episodes_out.size(); ++i) {
                std::cout << "EPISODE " << name << ' ' << seeds[i] << ' '
                          << episodes_out[i].score << ' '
                          << episodes_out[i].steps << "\n";
                scores.push_back(episodes_out[i].score);
            }
            summaries[name] = narcade::summarize(scores);
            results.emplace_back(name, std::move(scores));
        }
    } catch (const std::exception& e) {
        std::cerr << "evaluation failed: " << e.what() << "\n";
        return 1;
    }

    const auto elo_raw = narcade::elo_ratings_raw(results);
    const auto elo = narcade::elo_ratings(results);
    const auto h2h = narcade::head_to_head(results);

    for (const auto& kv : summaries) {
        std::cout << "RAW_SUMMARY " << kv.first << ' '
                  << double_hex(kv.second.raw_mean) << ' '
                  << double_hex(kv.second.raw_std) << "\n";
    }
    for (const auto& kv : elo_raw) {
        std::cout << "RAW_ELO " << kv.first << ' ' << double_hex(kv.second)
                  << "\n";
    }
    for (std::size_t a = 0; a < results.size(); ++a) {
        for (std::size_t b = 0; b < results.size(); ++b) {
            if (a == b) {
                continue;
            }
            const auto& cell = h2h[a][b];
            std::cout << "RAW_H2H " << results[a].first << ' '
                      << results[b].first << ' ' << cell->wins << ' '
                      << cell->draws << "\n";
        }
    }

    // Report JSON in evaluate.py's paired schema (lines 129-138),
    // restricted to the covered players, plus the scope field.
    std::string json = "{";
    json += "\"scope\": \"" + json_escape(kScope) + "\", ";
    json += "\"paired_seeds\": [";
    for (std::size_t i = 0; i < seeds.size(); ++i) {
        if (i != 0) json += ", ";
        json += std::to_string(seeds[i]);
    }
    json += "], \"scores_paired\": {";
    for (std::size_t p = 0; p < results.size(); ++p) {
        if (p != 0) json += ", ";
        json += "\"" + results[p].first + "\": [";
        for (std::size_t i = 0; i < results[p].second.size(); ++i) {
            if (i != 0) json += ", ";
            json += std::to_string(results[p].second[i]);
        }
        json += "]";
    }
    json += "}, \"summary_paired\": {";
    for (std::size_t p = 0; p < results.size(); ++p) {
        if (p != 0) json += ", ";
        const auto& s = summaries[results[p].first];
        json += "\"" + results[p].first + "\": {\"n\": " +
                std::to_string(s.n) +
                ", \"mean\": " + json_number(s.mean) +
                ", \"std\": " + json_number(s.std) +
                ", \"median\": " + json_number(s.median) +
                ", \"max\": " + std::to_string(s.max) +
                ", \"min\": " + std::to_string(s.min) + "}";
    }
    json += "}, \"elo\": {";
    for (std::size_t i = 0; i < elo.size(); ++i) {
        if (i != 0) json += ", ";
        json += "\"" + elo[i].first + "\": " + std::to_string(elo[i].second);
    }
    json += "}, \"head_to_head\": {";
    for (std::size_t a = 0; a < results.size(); ++a) {
        if (a != 0) json += ", ";
        json += "\"" + results[a].first + "\": {";
        for (std::size_t b = 0; b < results.size(); ++b) {
            if (b != 0) json += ", ";
            json += "\"" + results[b].first + "\": ";
            if (a == b) {
                json += "null";
            } else {
                const auto& cell = h2h[a][b];
                json += "{\"win_rate\": " + json_number(cell->win_rate) +
                        ", \"draw_rate\": " + json_number(cell->draw_rate) +
                        "}";
            }
        }
        json += "}";
    }
    json += "}, \"n_eval\": " + std::to_string(episodes) +
            ", \"mcts_n\": " + std::to_string(episodes) +
            ", \"mcts_sims\": " + std::to_string(sims) + "}";

    std::cout << "REPORT\n" << json << "\n";
    return 0;
}
