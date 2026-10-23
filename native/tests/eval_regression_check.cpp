// eval_regression_check: frozen-subset regression for the native
// evaluation engine (CPP-05), replayed in hosted CI.
//
// tools/native_eval_fuzz.py verifies the native evaluator against the
// Python reference over 200 paired episodes (seeds 5000-5049, the four
// non-neural players), comparing every episode, summary, Elo rating
// and head-to-head cell. It is excluded from hosted CI for runtime:
// its Python reference pass takes tens of minutes. It remains a LOCAL
// gate, and the rule for contributors is: changes to native evaluation
// code (native/src/evaluate.cpp, the evaluator CLI, or the engines it
// composes) require a documented local eval_fuzz run.
//
// This check freezes the same Python reference computation for a
// 12-episode subset (seeds 5000-5002, all four players) in
// native/golden/eval_regression.json and replays it through the native
// evaluator here, so hosted CI still guards evaluation parity on every
// run. Compared exactly (the eval_fuzz gates, restricted to the
// subset): per-episode (score, steps); the rounded summary fields;
// raw summary mean/std and raw Elo bit-exact (recorded as IEEE754 hex
// bit patterns); Elo integers; head-to-head win/draw rates and the raw
// win/draw counts.
//
// The frozen file stands outside meta.json's corpus accounting on
// purpose: meta_check and corpus_stats_check cover the eight corpus
// files and their 21 counts, and this file is none of those. It is
// also not part of regen_check (regenerating it needs the evaluation
// protocol's Python pass, which is exactly the runtime regen_check
// avoids). Regenerate it manually, from the repo root, with:
//
//   python3 tools/native_reference.py --eval-regression
//
// (deterministic; byte-identical from the same sources). Then re-run
// this check and commit the file together with any intentional
// protocol change.
//
// Usage: eval_regression_check [eval_regression.json] [qtable.json]
//   defaults: native/golden/eval_regression.json, runs/qtable.json
// Exit code is non-zero on any mismatch.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "narcade/evaluate.hpp"

namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::ostringstream contents;
    contents << in.rdbuf();
    return contents.str();
}

// ---- Minimal JSON reader ------------------------------------------------
// Just enough JSON for the golden files: objects, arrays, strings,
// numbers, booleans and null, parsed into a small DOM so the checks
// below can address values by key exactly as the Python tools do.
// (Same reader as native/tests/corpus_stats_check.cpp.)

struct Json {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    std::string text;  // string contents, or the raw number token
    std::vector<Json> array;
    std::vector<std::pair<std::string, Json>> object;

    const Json* find(const std::string& key) const {
        if (type != Type::Object) {
            return nullptr;
        }
        for (const auto& member : object) {
            if (member.first == key) {
                return &member.second;
            }
        }
        return nullptr;
    }
};

class JsonParser {
  public:
    explicit JsonParser(const std::string& text) : text_(text) {}

    Json parse() {
        Json value = parse_value();
        skip_ws();
        if (pos_ != text_.size()) {
            fail("trailing characters after the top-level value");
        }
        return value;
    }

  private:
    const std::string& text_;
    std::size_t pos_ = 0;

    [[noreturn]] void fail(const std::string& what) const {
        throw std::runtime_error("JSON parse error at byte " +
                                 std::to_string(pos_) + ": " + what);
    }

    void skip_ws() {
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                break;
            }
            ++pos_;
        }
    }

    char peek() {
        if (pos_ >= text_.size()) {
            fail("unexpected end of input");
        }
        return text_[pos_];
    }

    bool consume(char c) {
        if (pos_ < text_.size() && text_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    void expect(char c) {
        if (!consume(c)) {
            fail(std::string("expected '") + c + "'");
        }
    }

    void expect_literal(const char* literal) {
        const std::size_t len = std::strlen(literal);
        if (text_.compare(pos_, len, literal) != 0) {
            fail(std::string("invalid literal, expected ") + literal);
        }
        pos_ += len;
    }

    Json parse_value() {
        skip_ws();
        const char c = peek();
        if (c == '{') {
            return parse_object();
        }
        if (c == '[') {
            return parse_array();
        }
        Json value;
        if (c == '"') {
            value.type = Json::Type::String;
            value.text = parse_string();
        } else if (c == 't') {
            expect_literal("true");
            value.type = Json::Type::Bool;
            value.text = "true";
        } else if (c == 'f') {
            expect_literal("false");
            value.type = Json::Type::Bool;
            value.text = "false";
        } else if (c == 'n') {
            expect_literal("null");
            value.type = Json::Type::Null;
        } else {
            value.type = Json::Type::Number;
            value.text = parse_number();
        }
        return value;
    }

    Json parse_object() {
        Json value;
        value.type = Json::Type::Object;
        expect('{');
        skip_ws();
        if (consume('}')) {
            return value;
        }
        while (true) {
            skip_ws();
            if (peek() != '"') {
                fail("expected a string object key");
            }
            std::string key = parse_string();
            skip_ws();
            expect(':');
            value.object.emplace_back(std::move(key), parse_value());
            skip_ws();
            if (consume('}')) {
                return value;
            }
            expect(',');
        }
    }

    Json parse_array() {
        Json value;
        value.type = Json::Type::Array;
        expect('[');
        skip_ws();
        if (consume(']')) {
            return value;
        }
        while (true) {
            value.array.push_back(parse_value());
            skip_ws();
            if (consume(']')) {
                return value;
            }
            expect(',');
        }
    }

    void append_utf8(std::string& out, unsigned int code_point) {
        if (code_point < 0x80) {
            out += static_cast<char>(code_point);
        } else if (code_point < 0x800) {
            out += static_cast<char>(0xc0 | (code_point >> 6));
            out += static_cast<char>(0x80 | (code_point & 0x3f));
        } else if (code_point < 0x10000) {
            out += static_cast<char>(0xe0 | (code_point >> 12));
            out += static_cast<char>(0x80 | ((code_point >> 6) & 0x3f));
            out += static_cast<char>(0x80 | (code_point & 0x3f));
        } else {
            out += static_cast<char>(0xf0 | (code_point >> 18));
            out += static_cast<char>(0x80 | ((code_point >> 12) & 0x3f));
            out += static_cast<char>(0x80 | ((code_point >> 6) & 0x3f));
            out += static_cast<char>(0x80 | (code_point & 0x3f));
        }
    }

    unsigned int parse_hex4() {
        if (pos_ + 4 > text_.size()) {
            fail("truncated \\u escape");
        }
        unsigned int value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text_[pos_++];
            value <<= 4;
            if (c >= '0' && c <= '9') {
                value |= static_cast<unsigned int>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                value |= static_cast<unsigned int>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                value |= static_cast<unsigned int>(c - 'A' + 10);
            } else {
                fail("invalid hex digit in \\u escape");
            }
        }
        return value;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (pos_ >= text_.size()) {
                fail("unterminated string");
            }
            const char c = text_[pos_++];
            if (c == '"') {
                return out;
            }
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos_ >= text_.size()) {
                fail("unterminated escape");
            }
            switch (text_[pos_++]) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned int cp = parse_hex4();
                    if (cp >= 0xd800 && cp <= 0xdbff &&
                        pos_ + 1 < text_.size() && text_[pos_] == '\\' &&
                        text_[pos_ + 1] == 'u') {
                        pos_ += 2;
                        const unsigned int lo = parse_hex4();
                        cp = 0x10000 + ((cp - 0xd800) << 10) +
                             (lo - 0xdc00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail("invalid escape"); break;
            }
        }
    }

    std::string parse_number() {
        const std::size_t start = pos_;
        if (pos_ < text_.size() && text_[pos_] == '-') {
            ++pos_;
        }
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' ||
                c == 'E' || c == '+' || c == '-') {
                ++pos_;
            } else {
                break;
            }
        }
        if (pos_ == start) {
            fail("invalid number");
        }
        return text_.substr(start, pos_ - start);
    }
};

// ---- Recorded-value accessors --------------------------------------------

// Checked member access in two forms: member() returns a pointer for
// call sites that keep the value (a reference bound from a call whose
// arguments include temporaries trips -Wdangling-reference), and req()
// returns a reference for immediate use.
const Json* member(const Json& object, const std::string& key,
                   const std::string& where) {
    const Json* value = object.find(key);
    if (value == nullptr) {
        throw std::runtime_error("eval regression file: missing " + where +
                                 "." + key);
    }
    return value;
}

const Json& req(const Json& object, const std::string& key,
                const std::string& where) {
    return *member(object, key, where);
}

std::int64_t as_int(const Json& value, const std::string& where) {
    if (value.type != Json::Type::Number) {
        throw std::runtime_error("eval regression file: " + where +
                                 " is not a number");
    }
    return std::stoll(value.text);
}

double as_double(const Json& value, const std::string& where) {
    if (value.type != Json::Type::Number) {
        throw std::runtime_error("eval regression file: " + where +
                                 " is not a number");
    }
    return std::strtod(value.text.c_str(), nullptr);
}

std::string as_string(const Json& value, const std::string& where) {
    if (value.type != Json::Type::String) {
        throw std::runtime_error("eval regression file: " + where +
                                 " is not a string");
    }
    return value.text;
}

// IEEE754 bit pattern of a double as 16 lowercase hex digits, the
// encoding the generator records raw values in.
std::string double_hex(double v) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx",
                  static_cast<unsigned long long>(bits));
    return buf;
}

std::string show_double(double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 3) {
        std::cerr << "usage: eval_regression_check "
                     "[eval_regression.json] [qtable.json]\n";
        return 2;
    }
    const std::string golden_path =
        argc >= 2 ? argv[1] : "native/golden/eval_regression.json";
    const std::string qtable_path =
        argc >= 3 ? argv[2] : "runs/qtable.json";

    // Canonical protocol order (evaluate.py main()'s dict order,
    // mirrored by the evaluator CLI): Elo's sequential updates iterate
    // the results in this order, so the native results are built in
    // this order regardless of key order in the frozen file.
    const std::vector<std::string> kOrder = {"random", "greedy", "rookie",
                                             "grandmaster"};

    std::vector<std::string> failures;
    auto fail = [&failures](const std::string& msg) {
        failures.push_back(msg);
    };

    try {
        const Json doc = JsonParser(read_file(golden_path)).parse();
        if (as_string(req(doc, "schema", "root"), "schema") !=
            "eval-regression-v1") {
            throw std::runtime_error(
                "eval regression file: unsupported schema");
        }
        const Json* protocol = member(doc, "protocol", "root");
        const Json* seeds_json = member(*protocol, "seeds", "protocol");
        std::vector<std::int64_t> seeds;
        for (const Json& s : seeds_json->array) {
            seeds.push_back(as_int(s, "protocol.seeds"));
        }
        const int sims = static_cast<int>(
            as_int(req(*protocol, "sims", "protocol"), "protocol.sims"));
        const int width = static_cast<int>(
            as_int(req(*protocol, "width", "protocol"), "protocol.width"));
        const int height = static_cast<int>(as_int(
            req(*protocol, "height", "protocol"), "protocol.height"));

        const Json* episodes_doc = member(doc, "episodes", "root");
        const Json* summary_doc = member(doc, "summary", "root");
        const Json* raw_summary_doc = member(doc, "raw_summary", "root");
        const Json* elo_doc = member(doc, "elo", "root");
        const Json* raw_elo_doc = member(doc, "raw_elo", "root");
        const Json* h2h_doc = member(doc, "head_to_head", "root");
        const Json* raw_h2h_doc = member(doc, "raw_head_to_head", "root");

        const narcade::QTablePolicy::Table qtable =
            narcade::load_qtable_json(qtable_path);

        narcade::OrderedResults results;
        std::map<std::string, narcade::Summary> summaries;
        std::size_t episodes_compared = 0;

        for (const std::string& name : kOrder) {
            const Json* eps = member(*episodes_doc, name, "episodes");
            if (eps->array.size() != seeds.size()) {
                fail(name + ": recorded " +
                     std::to_string(eps->array.size()) +
                     " episodes, protocol has " +
                     std::to_string(seeds.size()));
                continue;
            }
            auto player = narcade::make_player(name, sims, &qtable);
            const std::vector<narcade::EpisodeResult> native_eps =
                narcade::run_episodes(*player, seeds, width, height);
            std::vector<int> scores;
            bool episodes_exact = true;
            for (std::size_t i = 0; i < seeds.size(); ++i) {
                const Json& rec = eps->array[i];
                const std::int64_t seed = as_int(
                    req(rec, "seed", "episodes." + name), "episode.seed");
                const std::int64_t score = as_int(
                    req(rec, "score", "episodes." + name),
                    "episode.score");
                const std::int64_t steps = as_int(
                    req(rec, "steps", "episodes." + name),
                    "episode.steps");
                if (seed != seeds[i]) {
                    fail(name + " episode " + std::to_string(i) +
                         ": recorded seed " + std::to_string(seed) +
                         " != protocol seed " +
                         std::to_string(seeds[i]));
                    episodes_exact = false;
                }
                if (native_eps[i].score != score) {
                    fail(name + " seed " + std::to_string(seeds[i]) +
                         ": native score " +
                         std::to_string(native_eps[i].score) +
                         " != recorded " + std::to_string(score));
                    episodes_exact = false;
                }
                if (native_eps[i].steps != steps) {
                    fail(name + " seed " + std::to_string(seeds[i]) +
                         ": native steps " +
                         std::to_string(native_eps[i].steps) +
                         " != recorded " + std::to_string(steps));
                    episodes_exact = false;
                }
                scores.push_back(native_eps[i].score);
                ++episodes_compared;
            }
            if (episodes_exact) {
                std::cout << name << ": " << seeds.size()
                          << " episodes exact\n";
            }
            summaries[name] = narcade::summarize(scores);
            results.emplace_back(name, std::move(scores));

            const narcade::Summary& sum = summaries[name];
            const Json* rec_sum = member(*summary_doc, name, "summary");
            bool summary_exact = true;
            auto check_int = [&](const char* field, std::int64_t got) {
                const std::int64_t want = as_int(
                    req(*rec_sum, field, "summary." + name),
                    "summary field");
                if (got != want) {
                    fail(name + ": summary " + field + " native " +
                         std::to_string(got) + " != recorded " +
                         std::to_string(want));
                    summary_exact = false;
                }
            };
            auto check_double = [&](const char* field, double got) {
                const double want = as_double(
                    req(*rec_sum, field, "summary." + name),
                    "summary field");
                if (got != want) {
                    fail(name + ": summary " + field + " native " +
                         show_double(got) + " != recorded " +
                         show_double(want));
                    summary_exact = false;
                }
            };
            check_int("n", sum.n);
            check_double("mean", sum.mean);
            check_double("std", sum.std);
            check_double("median", sum.median);
            check_int("max", sum.max);
            check_int("min", sum.min);
            const Json* rec_raw =
                member(*raw_summary_doc, name, "raw_summary");
            if (double_hex(sum.raw_mean) !=
                as_string(req(*rec_raw, "mean_hex",
                              "raw_summary." + name),
                          "raw mean hex")) {
                fail(name + ": raw summary mean bits " +
                     double_hex(sum.raw_mean) + " != recorded " +
                     as_string(req(*rec_raw, "mean_hex",
                                   "raw_summary." + name),
                               "raw mean hex"));
                summary_exact = false;
            }
            if (double_hex(sum.raw_std) !=
                as_string(req(*rec_raw, "std_hex",
                              "raw_summary." + name),
                          "raw std hex")) {
                fail(name + ": raw summary std bits " +
                     double_hex(sum.raw_std) + " != recorded " +
                     as_string(req(*rec_raw, "std_hex",
                                   "raw_summary." + name),
                               "raw std hex"));
                summary_exact = false;
            }
            if (summary_exact) {
                std::cout << name << ": summary exact (raw bit-exact)\n";
            }
        }

        const auto elo = narcade::elo_ratings(results);
        const auto elo_raw = narcade::elo_ratings_raw(results);
        bool elo_exact = true;
        for (const auto& kv : elo) {
            const std::int64_t want = as_int(
                req(*elo_doc, kv.first, "elo"), "elo");
            if (kv.second != want) {
                fail("elo " + kv.first + ": native " +
                     std::to_string(kv.second) + " != recorded " +
                     std::to_string(want));
                elo_exact = false;
            }
        }
        for (const auto& kv : elo_raw) {
            const std::string want = as_string(
                req(*raw_elo_doc, kv.first, "raw_elo"), "raw elo hex");
            if (double_hex(kv.second) != want) {
                fail("raw elo " + kv.first + ": native bits " +
                     double_hex(kv.second) + " != recorded " + want);
                elo_exact = false;
            }
        }
        if (elo_exact) {
            std::cout << "elo: ints exact (raw bit-exact)\n";
        }

        const narcade::HeadToHeadTable h2h =
            narcade::head_to_head(results);
        bool h2h_exact = true;
        std::size_t cells_compared = 0;
        for (std::size_t a = 0; a < kOrder.size(); ++a) {
            const Json* row = member(*h2h_doc, kOrder[a], "head_to_head");
            const Json* raw_row =
                member(*raw_h2h_doc, kOrder[a], "raw_head_to_head");
            for (std::size_t b = 0; b < kOrder.size(); ++b) {
                const Json* cell_rec = member(*row, kOrder[b],
                                              "head_to_head." + kOrder[a]);
                const Json* raw_rec = member(*raw_row, kOrder[b],
                                             "raw_head_to_head." +
                                                 kOrder[a]);
                const auto& cell = h2h[a][b];
                const std::string pair =
                    kOrder[a] + " vs " + kOrder[b];
                if (a == b) {
                    if (cell.has_value() ||
                        cell_rec->type != Json::Type::Null ||
                        raw_rec->type != Json::Type::Null) {
                        fail("head_to_head " + pair +
                             ": diagonal must be null on both sides");
                        h2h_exact = false;
                    }
                    continue;
                }
                ++cells_compared;
                if (!cell.has_value()) {
                    fail("head_to_head " + pair +
                         ": native cell missing");
                    h2h_exact = false;
                    continue;
                }
                const double win_rate = as_double(
                    req(*cell_rec, "win_rate", "head_to_head cell"),
                    "win_rate");
                const double draw_rate = as_double(
                    req(*cell_rec, "draw_rate", "head_to_head cell"),
                    "draw_rate");
                if (cell->win_rate != win_rate) {
                    fail("head_to_head " + pair + ": win_rate native " +
                         show_double(cell->win_rate) + " != recorded " +
                         show_double(win_rate));
                    h2h_exact = false;
                }
                if (cell->draw_rate != draw_rate) {
                    fail("head_to_head " + pair + ": draw_rate native " +
                         show_double(cell->draw_rate) +
                         " != recorded " + show_double(draw_rate));
                    h2h_exact = false;
                }
                if (raw_rec->array.size() != 2) {
                    fail("head_to_head " + pair +
                         ": raw counts are not a [wins, draws] pair");
                    h2h_exact = false;
                    continue;
                }
                const std::int64_t wins =
                    as_int(raw_rec->array[0], "raw h2h wins");
                const std::int64_t draws =
                    as_int(raw_rec->array[1], "raw h2h draws");
                if (cell->wins != wins || cell->draws != draws) {
                    fail("head_to_head " + pair + ": counts native (" +
                         std::to_string(cell->wins) + ", " +
                         std::to_string(cell->draws) +
                         ") != recorded (" + std::to_string(wins) +
                         ", " + std::to_string(draws) + ")");
                    h2h_exact = false;
                }
            }
        }
        if (h2h_exact) {
            std::cout << "head_to_head: rates and counts exact\n";
        }

        if (!failures.empty()) {
            for (const std::string& failure : failures) {
                std::cout << "FAIL: " << failure << '\n';
            }
            std::cout << "result: FAIL\n";
            return 1;
        }
        std::cout << "result: PASS (" << episodes_compared
                  << " episodes exact; summaries, Elo and head-to-head "
                     "exact)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "eval_regression_check: " << error.what() << '\n';
        return 1;
    }
}
