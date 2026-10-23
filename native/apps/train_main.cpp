// train_main: native counterpart of scripts/train.py's main().
// Mirrors its arguments (--episodes, --seed, --out) and its five
// output artifacts (qtable.json, policy.json, curve.json, elo.json,
// train_meta.json). The train() keyword arguments are also exposed
// (--eps-start, --eps-end, --checkpoint-every, --elo-games, --width,
// --height) with train.py's defaults, so parity checks can pin small
// deterministic configurations without editing defaults.
//
// --dump-episodes <path> additionally writes the per-episode streams
// and the agent's final RNG state as JSON, for tools/
// native_trainer_check.py; train.py keeps those in locals, so this
// file is a verification surface, not a sixth artifact.
//
// JSON numbers are written with std::to_chars (shortest round-trip),
// so parsed values are bit-identical to Python's json.dump output
// even where the text rendering differs (e.g. 1200 vs 1200.0, and
// ", " separators). The parity driver compares parsed values.
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "narcade/trainer.hpp"

namespace {

std::string fmt_double(double v) {
    char buf[40];
    const auto res = std::to_chars(buf, buf + sizeof(buf), v);
    return std::string(buf, res.ptr);
}

void write_file(const std::string& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot open " + path);
    }
    out << text;
}

std::string join_path(const std::string& dir, const std::string& name) {
    if (!dir.empty() && dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

std::string qtable_json(const narcade::QLearningAgent::Table& q) {
    std::ostringstream os;
    os << "[";
    for (std::size_t s = 0; s < q.size(); ++s) {
        if (s) os << ",";
        os << "[";
        for (std::size_t a = 0; a < q[s].size(); ++a) {
            if (a) os << ",";
            os << fmt_double(q[s][a]);
        }
        os << "]";
    }
    os << "]";
    return os.str();
}

std::string int_list_json(const std::vector<int>& values) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) os << ",";
        os << values[i];
    }
    os << "]";
    return os.str();
}

std::string double_list_json(const std::vector<double>& values) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) os << ",";
        os << fmt_double(values[i]);
    }
    os << "]";
    return os.str();
}

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s [--episodes N] [--seed S] [--out DIR]\n"
                 "          [--eps-start X] [--eps-end X]\n"
                 "          [--checkpoint-every N] [--elo-games N]\n"
                 "          [--width W] [--height H] [--quiet]\n"
                 "          [--dump-episodes PATH]\n",
                 argv0);
}

}  // namespace

int main(int argc, char** argv) {
    narcade::TrainConfig config;
    std::string out_dir = "runs";
    std::string dump_path;
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
        if (arg == "--episodes") {
            config.episodes = std::stoi(next("--episodes"));
        } else if (arg == "--seed") {
            config.seed = std::stoll(next("--seed"));
        } else if (arg == "--out") {
            out_dir = next("--out");
        } else if (arg == "--eps-start") {
            config.eps_start = std::stod(next("--eps-start"));
        } else if (arg == "--eps-end") {
            config.eps_end = std::stod(next("--eps-end"));
        } else if (arg == "--checkpoint-every") {
            config.checkpoint_every = std::stoi(next("--checkpoint-every"));
        } else if (arg == "--elo-games") {
            config.elo_games = std::stoi(next("--elo-games"));
        } else if (arg == "--width") {
            config.width = std::stoi(next("--width"));
        } else if (arg == "--height") {
            config.height = std::stoi(next("--height"));
        } else if (arg == "--dump-episodes") {
            dump_path = next("--dump-episodes");
        } else if (arg == "--quiet") {
            config.verbose = false;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    const narcade::TrainResult result = narcade::train_agent(config);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();

    // train.py:108: os.makedirs(args.out, exist_ok=True).
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);

    try {
        // qtable.json: json.dump(agent.q) (train.py:112-113).
        write_file(join_path(out_dir, "qtable.json"), qtable_json(result.q));
        // policy.json: QTablePolicy(agent.q).to_jsonable()
        // (train.py:114-115; policies.py:44-47).
        {
            std::ostringstream os;
            os << "{\"states\":32,\"actions\":3,\"best_action\":"
               << int_list_json(result.best_action) << "}";
            write_file(join_path(out_dir, "policy.json"), os.str());
        }
        // curve.json: {"mean_score": ..., "mean_reward": ...}
        // (train.py:116-117).
        {
            std::ostringstream os;
            os << "{\"mean_score\":" << double_list_json(result.curve_scores)
               << ",\"mean_reward\":"
               << double_list_json(result.curve_rewards) << "}";
            write_file(join_path(out_dir, "curve.json"), os.str());
        }
        // elo.json: [{"episode": ep, "elo": elo}, ...] (train.py:118-119).
        {
            std::ostringstream os;
            os << "[";
            for (std::size_t i = 0; i < result.elo_hist.size(); ++i) {
                if (i) os << ",";
                os << "{\"episode\":" << result.elo_hist[i].episode
                   << ",\"elo\":" << fmt_double(result.elo_hist[i].elo)
                   << "}";
            }
            os << "]";
            write_file(join_path(out_dir, "elo.json"), os.str());
        }
        // train_meta.json: the meta dict (train.py:87-96, 120-121),
        // keys in insertion order. "seconds" is elapsed wall time
        // rounded to 0.1 and is the one field allowed to differ.
        {
            const double rounded_seconds =
                std::round(seconds * 10.0) / 10.0;
            std::ostringstream os;
            os << "{\n"
               << "  \"episodes\": " << config.episodes << ",\n"
               << "  \"seed\": " << config.seed << ",\n"
               << "  \"width\": " << config.width << ",\n"
               << "  \"height\": " << config.height << ",\n"
               << "  \"eps_start\": " << fmt_double(config.eps_start)
               << ",\n"
               << "  \"eps_end\": " << fmt_double(config.eps_end) << ",\n"
               << "  \"checkpoint_every\": " << config.checkpoint_every
               << ",\n"
               << "  \"elo_games\": " << config.elo_games << ",\n"
               << "  \"seconds\": " << fmt_double(rounded_seconds) << ",\n"
               << "  \"final_mean100_score\": "
               << fmt_double(result.curve_scores.back()) << ",\n"
               << "  \"final_elo\": " << fmt_double(result.final_elo)
               << "\n}\n";
            write_file(join_path(out_dir, "train_meta.json"), os.str());
        }
        if (!dump_path.empty()) {
            std::ostringstream os;
            os << "{\"scores\":" << int_list_json(result.ep_scores)
               << ",\"rewards\":" << double_list_json(result.ep_rewards)
               << ",\"final_epsilon\":" << fmt_double(result.final_epsilon)
               << ",\"final_rng_state\":{\"mt\":[";
            const auto& state = result.final_rng_state;
            for (std::size_t i = 0; i < state.mt.size(); ++i) {
                if (i) os << ",";
                os << state.mt[i];
            }
            os << "],\"index\":" << state.index << "}}";
            write_file(dump_path, os.str());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    // train.py:122-124 summary lines (seconds is this run's own).
    std::printf("done in %.1fs -> %s\n",
                std::round(seconds * 10.0) / 10.0, out_dir.c_str());
    std::printf("final mean100 score=%.2f  elo=%.0f\n",
                result.curve_scores.back(), result.final_elo);
    return 0;
}
