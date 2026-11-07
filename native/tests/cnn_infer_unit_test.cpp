// cnn_infer_unit_test: unit tests for the native CNN DQN inference
// module (native/src/cnn_infer.cpp).
//
// Coverage:
//   * all six committed DA8 checkpoint exports load through the
//     strict loader (which verifies their recorded weights SHA-256
//     values internally: six real-world verifications of the
//     canonical-rendering hash, ~46k float values each checked
//     against Python's own digest), with the documented config and
//     parameter counts (plain 13,795; residual 32,291);
//   * encode_cnn on hand-computed cases (head/body/food planes,
//     heading one-hot, the no-food terminal-win case);
//   * forward determinism and act/argmax consistency;
//   * fail-closed loader behavior on tampered documents (recorded
//     hash edited, layer shape edited, schema version bumped, kind
//     edited, metadata key renamed, file truncated). Each tamper is a
//     textual edit of a committed export written to a scratch copy;
//     the committed artifacts are never modified.
//
// Usage: cnn_infer_unit_test <runs_dir>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "narcade/cnn_infer.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cout << "FAIL: " << what << '\n';
        ++failures;
    }
}

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void write_file(const std::string& path, const std::string& text) {
    std::ofstream out(path);
    out << text;
}

bool replace_first(std::string& text, const std::string& from,
                   const std::string& to) {
    const std::size_t pos = text.find(from);
    if (pos == std::string::npos) {
        return false;
    }
    text.replace(pos, from.size(), to);
    return true;
}

// Loading `path` must fail with CnnCheckpointError specifically (a
// wrong exception type, or success, is a test failure).
void check_rejected(const std::string& path, const std::string& what) {
    try {
        (void)narcade::load_cnn_weights(path);
    } catch (const narcade::CnnCheckpointError&) {
        return;
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << what << ": wrong exception type: "
                  << e.what() << '\n';
        ++failures;
        return;
    }
    std::cout << "FAIL: " << what << ": load succeeded\n";
    ++failures;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "usage: cnn_infer_unit_test <runs_dir>\n";
        return 2;
    }
    const std::string runs = argv[1];

    // --- Committed exports load, with documented configs. ----------
    struct Case {
        const char* file;
        bool residual;
        int params;
    };
    const Case cases[6] = {
        {"cnn_dqn_plain_seed7_weights.json", false, 13795},
        {"cnn_dqn_plain_seed8_weights.json", false, 13795},
        {"cnn_dqn_plain_seed9_weights.json", false, 13795},
        {"cnn_dqn_residual_seed7_weights.json", true, 32291},
        {"cnn_dqn_residual_seed8_weights.json", true, 32291},
        {"cnn_dqn_residual_seed9_weights.json", true, 32291},
    };
    for (const Case& c : cases) {
        try {
            const narcade::CnnWeights w =
                narcade::load_cnn_weights(runs + "/" + c.file);
            check(w.config.residual == c.residual,
                  std::string(c.file) + ": residual flag");
            check(w.config.channels == std::vector<int>({16, 32}),
                  std::string(c.file) + ": channels");
            check(w.config.kernel == 3 && w.config.pool_size == 2 &&
                      w.config.hidden == 64,
                  std::string(c.file) + ": kernel/pool/hidden");
            check(w.num_params() == c.params,
                  std::string(c.file) + ": num_params");
            check(w.weights_sha256.size() == 64,
                  std::string(c.file) + ": sha256 recorded");
        } catch (const std::exception& e) {
            std::cout << "FAIL: " << c.file << ": load threw: " << e.what()
                      << '\n';
            ++failures;
        }
    }

    // --- encode_cnn hand-computed cases. ---------------------------
    {
        // Tail-first body: tail (5,5), then (5,6), head (5,7).
        const std::vector<narcade::Cell> body{{5, 5}, {5, 6}, {5, 7}};
        const narcade::CnnEncoding enc =
            narcade::encode_cnn(body, narcade::Cell{2, 3}, 1);
        check(enc.spatial[5 * 12 + 7] == 1.0f, "encode: head plane");
        check(enc.spatial[144 + 5 * 12 + 5] == 1.0f &&
                  enc.spatial[144 + 5 * 12 + 6] == 1.0f,
              "encode: body plane");
        check(enc.spatial[288 + 2 * 12 + 3] == 1.0f, "encode: food plane");
        float sum = 0.0f;
        for (float v : enc.spatial) {
            sum += v;
        }
        check(sum == 4.0f, "encode: spatial sum");
        check(enc.heading[1] == 1.0f && enc.heading[0] == 0.0f &&
                  enc.heading[2] == 0.0f && enc.heading[3] == 0.0f,
              "encode: heading one-hot");

        // Terminal win: no food, the food plane stays empty.
        const narcade::CnnEncoding won =
            narcade::encode_cnn(body, std::nullopt, 3);
        float food_sum = 0.0f;
        for (int i = 288; i < 432; ++i) {
            food_sum += won.spatial[static_cast<std::size_t>(i)];
        }
        check(food_sum == 0.0f, "encode: no-food plane empty");
        check(won.heading[3] == 1.0f, "encode: no-food heading");
    }

    // --- Forward determinism and act consistency. ------------------
    {
        const narcade::CnnNet net(
            narcade::load_cnn_weights(runs + "/cnn_dqn_plain_seed7_weights.json"));
        const std::vector<narcade::Cell> body{
            {6, 3}, {6, 4}, {6, 5}, {5, 5}, {4, 5}};
        const narcade::CnnEncoding enc =
            narcade::encode_cnn(body, narcade::Cell{4, 8}, 0);
        const narcade::CnnActivations a = net.forward(enc);
        const narcade::CnnActivations b = net.forward(enc);
        check(a.logits == b.logits, "forward: deterministic logits");
        check(a.features.size() == 32 * 2 * 2 + 4,
              "forward: feature length");
        bool finite = true;
        for (double v : a.logits) {
            finite = finite && std::isfinite(v);
        }
        check(finite, "forward: finite logits");
        int want = 0;
        for (int i = 1; i < 3; ++i) {
            if (a.logits[static_cast<std::size_t>(i)] >
                a.logits[static_cast<std::size_t>(want)]) {
                want = i;
            }
        }
        check(net.act(enc) == want, "forward: act is argmax");
    }

    // --- Tampered documents fail closed. ---------------------------
    {
        namespace fs = std::filesystem;
        const fs::path scratch =
            fs::temp_directory_path() / "cnn_infer_unit_test";
        fs::create_directories(scratch);
        const std::string original =
            read_file(runs + "/cnn_dqn_plain_seed7_weights.json");
        check(!original.empty(), "tamper: source export readable");

        auto tamper = [&](const std::string& name, std::string text,
                          const std::string& what) {
            const fs::path p = scratch / name;
            write_file(p.string(), text);
            check_rejected(p.string(), what);
        };

        {
            std::string t = original;
            const std::string marker = "\"weights_sha256\": \"";
            const std::size_t pos = t.find(marker);
            check(pos != std::string::npos, "tamper: hash marker found");
            if (pos != std::string::npos) {
                char& digit = t[pos + marker.size()];
                digit = digit == '0' ? '1' : '0';
            }
            tamper("bad_hash.json", t, "tamper: recorded hash edited");
        }
        {
            std::string t = original;
            check(replace_first(t, "\"shape\": [16, 3, 3, 3]",
                                "\"shape\": [16, 3, 3, 4]"),
                  "tamper: shape marker found");
            tamper("bad_shape.json", t, "tamper: layer shape edited");
        }
        {
            std::string t = original;
            check(replace_first(t, "\"schema_version\": 1",
                                "\"schema_version\": 2"),
                  "tamper: schema marker found");
            tamper("bad_schema.json", t, "tamper: schema version bumped");
        }
        {
            std::string t = original;
            check(replace_first(
                      t, "\"kind\": \"deeparcade-cnn-dqn-checkpoint\"",
                      "\"kind\": \"deeparcade-dqn-checkpoint\""),
                  "tamper: kind marker found");
            tamper("bad_kind.json", t, "tamper: kind edited");
        }
        {
            std::string t = original;
            check(replace_first(t, "\"training_args\"", "\"training_argz\""),
                  "tamper: metadata marker found");
            tamper("bad_metadata.json", t, "tamper: metadata key renamed");
        }
        tamper("truncated.json", original.substr(0, original.size() / 2),
               "tamper: file truncated");
        {
            // One weight value flipped to a different finite value:
            // shapes and finiteness still hold, so only the weights
            // hash can catch it.
            std::string t = original;
            const std::size_t layers_pos = t.find("\"layers\"");
            check(layers_pos != std::string::npos,
                  "tamper: layers marker found");
            bool flipped = false;
            if (layers_pos != std::string::npos) {
                for (std::size_t i = layers_pos; i + 1 < t.size(); ++i) {
                    if (t[i] == '.' && t[i + 1] >= '0' && t[i + 1] <= '8') {
                        t[i + 1] = static_cast<char>(t[i + 1] + 1);
                        flipped = true;
                        break;
                    }
                }
            }
            check(flipped, "tamper: weight digit flipped");
            tamper("bad_value.json", t, "tamper: weight value edited");
        }
        fs::remove_all(scratch);
    }

    if (failures == 0) {
        std::cout << "cnn_infer_unit_test: all checks passed\n";
        return 0;
    }
    std::cout << "cnn_infer_unit_test: " << failures << " failure(s)\n";
    return 1;
}
