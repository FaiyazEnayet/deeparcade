// narcade_tournament: native CLI for the tournament runner
// (native/include/narcade/tournament.hpp).
//
// Plays repeated round-robin rounds of paired episodes between the
// five evaluation players and writes a JSON standings report
// (schema "tournament-v1"). The report's "scope" field carries the
// runner's scope declaration; the same text is in the help below.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "narcade/tournament.hpp"

namespace {

void print_usage(const char* argv0) {
    std::cout
        << "usage: " << argv0
        << " [--player NAME]... [--rounds R] [--seeds-per-round S]\n"
        << "       [--seed0 S] [--sims M] [--qtable PATH] "
           "[--weights PATH]\n"
        << "       [--width W] [--height H] [--out PATH]\n"
        << "Players (evaluate.py's roster order is kept for any "
           "subset):\n"
        << "  random, greedy, rookie (alias qtable), pro, grandmaster "
           "(alias mcts)\n"
        << "Defaults: all five players, rounds=3, seeds-per-round=10, "
           "seed0=5000,\n"
        << "  sims=80, qtable=runs/qtable.json, "
           "weights=runs/dqn_weights.json,\n"
        << "  width=height=12, out=runs/tournament.json "
           "(\"--out -\" writes to stdout).\n"
        << "Scope: " << narcade::tournament_scope() << "\n";
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
    narcade::TournamentConfig config;
    std::string qtable_path = "runs/qtable.json";
    std::string weights_path = "runs/dqn_weights.json";
    std::string out_path = "runs/tournament.json";

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
        } else if (arg == "--rounds") {
            config.rounds = std::stoi(next("--rounds"));
        } else if (arg == "--seeds-per-round") {
            config.seeds_per_round = std::stoi(next("--seeds-per-round"));
        } else if (arg == "--seed0") {
            config.seed0 = std::stoll(next("--seed0"));
        } else if (arg == "--sims") {
            config.mcts_sims = std::stoi(next("--sims"));
        } else if (arg == "--qtable") {
            qtable_path = next("--qtable");
        } else if (arg == "--weights") {
            weights_path = next("--weights");
        } else if (arg == "--width") {
            config.width = std::stoi(next("--width"));
        } else if (arg == "--height") {
            config.height = std::stoi(next("--height"));
        } else if (arg == "--out") {
            out_path = next("--out");
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    // evaluate.py main()'s roster order, grandmaster last; any
    // subset keeps this order (Elo's sequential updates depend on
    // it, exactly as in the evaluator CLI).
    const std::vector<std::string> kOrder = {"random", "greedy", "rookie",
                                             "pro", "grandmaster"};
    std::vector<std::string> roster;
    for (const std::string& name : kOrder) {
        if (requested.empty() ||
            std::find(requested.begin(), requested.end(), name) !=
                requested.end()) {
            roster.push_back(name);
        }
    }

    try {
        narcade::QTablePolicy::Table qtable;
        bool qtable_loaded = false;
        narcade::DqnWeights weights;
        bool weights_loaded = false;
        std::vector<std::unique_ptr<narcade::Player>> players;
        for (const std::string& name : roster) {
            if (name == "rookie" && !qtable_loaded) {
                qtable = narcade::load_qtable_json(qtable_path);
                qtable_loaded = true;
            }
            if (name == "pro" && !weights_loaded) {
                weights = narcade::load_dqn_weights(weights_path);
                weights_loaded = true;
            }
            players.push_back(narcade::make_tournament_player(
                name, config.mcts_sims, qtable_loaded ? &qtable : nullptr,
                weights_loaded ? &weights : nullptr));
        }

        const narcade::TournamentResult result =
            narcade::run_tournament(players, roster, config);

        for (std::size_t r = 0; r < result.rounds.size(); ++r) {
            std::cout << "round " << (r + 1) << " elo:";
            for (std::size_t i = 0; i < roster.size(); ++i) {
                std::cout << ' ' << roster[i] << '='
                          << result.rounds[r].elo_after[i];
            }
            std::cout << "\n";
        }
        std::cout << "standings:\n";
        for (const auto& s : result.standings) {
            std::cout << "  " << s.name << " elo=" << s.elo
                      << " points=" << s.record.points()
                      << " W" << s.record.wins << "/D" << s.record.draws
                      << "/L" << s.record.losses << "\n";
        }

        const std::string report = narcade::tournament_report_json(result);
        if (out_path == "-") {
            std::cout << report << "\n";
        } else {
            std::ofstream out(out_path, std::ios::binary);
            if (!out) {
                std::cerr << "cannot open output path: " << out_path << "\n";
                return 1;
            }
            out << report << "\n";
            std::cout << "wrote " << out_path << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "tournament failed: " << e.what() << "\n";
        return 1;
    }
}
