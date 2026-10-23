// corpus_stats_check: independent recount of the frozen golden corpora
// vs native/golden/meta.json.
//
// tools/native_reference.py records 21 corpus counts in meta.json at
// generation time, and the meta_check CTest verifies the recorded
// values are positive whole numbers under exactly the expected key
// set, plus the golden files' SHA-256 hashes. That leaves one gap: a
// wrong-but-positive count (a generator tally bug, a hand edit)
// passes. This checker closes it by recounting the records from the
// corpus files themselves, using the same record definitions the
// generator tallied and the C++ batch runners parse (see the protocol
// headers in native/tests/*_diff_runner.cpp), and comparing every
// derived count against meta.json.
//
// Record definitions per file:
//
//   transitions.json   episodes = len(episodes);
//                      steps = sum of len(episode.steps).
//   q_updates.json     q_updates = len(updates);
//                      q_act_stream = len(act_states).
//   mcts_oracle.json   mcts_oracle_cases = len(cases).
//   snake_corpus.txt   session file ("> " inputs, "< " replies):
//                      episodes = EPISODE records; transitions = "S"
//                      step lines; exceptions = "X" step lines. On an
//                      "S" line (S <state_before> <state_after>
//                      <reward_hex> <done> <reason> <food> <score>
//                      <steps> <body>): eats = reward_hex equals the
//                      IEEE754 hex of 10.0; deaths / wins / timeouts =
//                      reason code 1 / 2 / 3. tail_vacations are not a
//                      stored field; they are reconstructed exactly as
//                      the generator defined them: a step whose new
//                      head cell (last cell of the post-step body,
//                      bodies are tail-first) equals the pre-step tail
//                      cell (body[0] before the step, tracked across
//                      the episode from INIT and each post-step body)
//                      on a step that was not already done and did
//                      not eat (new head != pre-step food).
//   policies_corpus.txt  episodes = EPISODE records; states = "P"
//   replies other than the post-terminal "P -1 -1" placeholder;
//   exceptions = "X" step lines; rand_draws = integers on the data
//   line following each "RAND" reply.
//   mcts_cases.txt     searches = "ACTION" replies; root_children =
//                      "CHILD" reply lines.
//   mcts_wall_cases.txt  wall_searches = "ACTION" replies;
//                      wall_root_children = "CHILD" reply lines (the
//                      maze-mode session; same reply records as
//                      mcts_cases.txt, counted under the wall keys).
//
// Usage: corpus_stats_check [golden-dir]   (default: native/golden)
// Exit code is non-zero on any mismatch. Native port of the former
// tools/native_corpus_stats_check.py (same derivations, same output).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Counts = std::map<std::string, long long>;
using Cell = std::pair<int, int>;

bool starts_with(const std::string& line, const std::string& prefix) {
    return line.size() >= prefix.size() &&
           line.compare(0, prefix.size(), prefix) == 0;
}

// Split on a single delimiter character, keeping empty fields (the
// semantics of Python's str.split with a one-character separator).
std::vector<std::string> split(const std::string& text, char delim) {
    std::vector<std::string> fields;
    std::string current;
    for (char ch : text) {
        if (ch == delim) {
            fields.push_back(current);
            current.clear();
        } else {
            current += ch;
        }
    }
    fields.push_back(current);
    return fields;
}

// Number of whitespace-separated tokens (Python's str.split()).
long long count_tokens(const std::string& text) {
    std::istringstream in(text);
    long long count = 0;
    std::string token;
    while (in >> token) {
        ++count;
    }
    return count;
}

Cell parse_cell(const std::string& token) {
    const std::vector<std::string> parts = split(token, ',');
    return {std::stoi(parts.at(0)), std::stoi(parts.at(1))};
}

std::vector<Cell> parse_body(const std::string& token) {
    std::vector<Cell> body;
    for (const std::string& cell : split(token, ';')) {
        body.push_back(parse_cell(cell));
    }
    return body;
}

std::optional<Cell> parse_food(const std::string& token) {
    const Cell cell = parse_cell(token);
    if (cell.first == -1 && cell.second == -1) {
        return std::nullopt;
    }
    return cell;
}

// IEEE754 hex of 10.0: the reward a step gets for eating, as the
// generator's eats tally compares it (dbl_hex(10.0) in
// tools/native_reference.py).
std::string eat_reward_hex() {
    const double value = 10.0;
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx",
                  static_cast<unsigned long long>(bits));
    return buf;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path.string());
    }
    std::ostringstream contents;
    contents << in.rdbuf();
    return contents.str();
}

std::ifstream open_lines(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path.string());
    }
    return in;
}

// ---- Minimal JSON reader ------------------------------------------------
// Just enough JSON for the golden files: objects, arrays, strings,
// numbers, booleans and null, parsed into a small DOM so the counters
// below can address values by key exactly as the Python checker does.

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
                    if (cp >= 0xd800 && cp <= 0xdbff && pos_ + 6 <= text_.size() &&
                        text_[pos_] == '\\' && text_[pos_ + 1] == 'u') {
                        pos_ += 2;
                        const unsigned int low = parse_hex4();
                        if (low >= 0xdc00 && low <= 0xdfff) {
                            cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
                        } else {
                            append_utf8(out, cp);
                            cp = low;
                        }
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail("invalid string escape");
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
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
                c == '+' || c == '-') {
                ++pos_;
            } else {
                break;
            }
        }
        if (pos_ == start) {
            fail("expected a JSON value");
        }
        return text_.substr(start, pos_ - start);
    }
};

Json load_json(const std::filesystem::path& path) {
    return JsonParser(read_file(path)).parse();
}

const Json& require_array(const Json& root, const std::string& key,
                          const std::string& file) {
    const Json* value = root.find(key);
    if (value == nullptr || value->type != Json::Type::Array) {
        throw std::runtime_error(file + ": missing array key \"" + key + "\"");
    }
    return *value;
}

// ---- Corpus counters ----------------------------------------------------

Counts count_transitions(const std::filesystem::path& path) {
    const Json data = load_json(path);
    const Json& episodes = require_array(data, "episodes", path.filename().string());
    long long steps = 0;
    for (const Json& episode : episodes.array) {
        steps += static_cast<long long>(
            require_array(episode, "steps", path.filename().string()).array.size());
    }
    return {
        {"transitions_episodes", static_cast<long long>(episodes.array.size())},
        {"transitions_steps", steps},
    };
}

Counts count_q_updates(const std::filesystem::path& path) {
    const Json data = load_json(path);
    return {
        {"q_updates", static_cast<long long>(
                          require_array(data, "updates", path.filename().string())
                              .array.size())},
        {"q_act_stream", static_cast<long long>(
                             require_array(data, "act_states", path.filename().string())
                                 .array.size())},
    };
}

Counts count_mcts_oracle(const std::filesystem::path& path) {
    const Json data = load_json(path);
    return {{"mcts_oracle_cases",
             static_cast<long long>(
                 require_array(data, "cases", path.filename().string()).array.size())}};
}

Counts count_snake_corpus(const std::filesystem::path& path) {
    Counts counts{
        {"snake_episodes", 0}, {"snake_transitions", 0},
        {"snake_exceptions", 0}, {"snake_eats", 0},
        {"snake_deaths", 0}, {"snake_wins", 0},
        {"snake_timeouts", 0}, {"snake_tail_vacations", 0},
    };
    const std::string eat_hex = eat_reward_hex();
    std::optional<std::vector<Cell>> pre_body;
    std::optional<Cell> pre_food;
    bool pre_done = false;
    std::ifstream in = open_lines(path);
    std::string line;
    while (std::getline(in, line)) {
        if (starts_with(line, "> EPISODE ")) {
            counts["snake_episodes"] += 1;
            pre_body.reset();
            pre_food.reset();
            pre_done = false;
        } else if (starts_with(line, "< INIT ")) {
            const std::vector<std::string> fields =
                split(line.substr(std::string("< INIT ").size()), ' ');
            pre_food = parse_food(fields.at(1));
            pre_body = parse_body(fields.at(2));
        } else if (starts_with(line, "< X ")) {
            // step() raised; env state (and the done flag) unchanged.
            counts["snake_exceptions"] += 1;
        } else if (starts_with(line, "< S ")) {
            const std::vector<std::string> fields =
                split(line.substr(std::string("< S ").size()), ' ');
            const std::string& reward_hex = fields.at(2);
            const int done = std::stoi(fields.at(3));
            const int reason = std::stoi(fields.at(4));
            const std::optional<Cell> post_food = parse_food(fields.at(5));
            const std::vector<Cell> post_body = parse_body(fields.at(8));
            counts["snake_transitions"] += 1;
            if (reward_hex == eat_hex) {
                counts["snake_eats"] += 1;
            }
            if (reason == 1) {
                counts["snake_deaths"] += 1;
            } else if (reason == 2) {
                counts["snake_wins"] += 1;
            } else if (reason == 3) {
                counts["snake_timeouts"] += 1;
            }
            if (!pre_done && pre_body.has_value() && !pre_body->empty()) {
                const Cell target = post_body.back();
                const bool ate =
                    pre_food.has_value() && target == *pre_food;
                if (target == pre_body->front() && !ate) {
                    counts["snake_tail_vacations"] += 1;
                }
            }
            pre_body = post_body;
            pre_food = post_food;
            pre_done = done == 1;
        }
    }
    return counts;
}

Counts count_policies_corpus(const std::filesystem::path& path) {
    Counts counts{
        {"policies_episodes", 0},
        {"policies_states", 0},
        {"policies_exceptions", 0},
        {"policies_rand_draws", 0},
    };
    bool expect_rand_data = false;
    std::ifstream in = open_lines(path);
    std::string line;
    while (std::getline(in, line)) {
        if (starts_with(line, "> EPISODE ")) {
            counts["policies_episodes"] += 1;
        } else if (starts_with(line, "< P ")) {
            if (line != "< P -1 -1") {
                counts["policies_states"] += 1;
            }
        } else if (starts_with(line, "< X ")) {
            counts["policies_exceptions"] += 1;
        } else if (line == "< RAND") {
            expect_rand_data = true;
        } else if (expect_rand_data) {
            counts["policies_rand_draws"] +=
                count_tokens(line.size() >= 2 ? line.substr(2) : "");
            expect_rand_data = false;
        }
    }
    return counts;
}

Counts count_mcts_cases(const std::filesystem::path& path,
                        const std::string& searches_key,
                        const std::string& children_key) {
    Counts counts{{searches_key, 0}, {children_key, 0}};
    std::ifstream in = open_lines(path);
    std::string line;
    while (std::getline(in, line)) {
        if (starts_with(line, "< ACTION ")) {
            counts[searches_key] += 1;
        } else if (starts_with(line, "< CHILD ")) {
            counts[children_key] += 1;
        }
    }
    return counts;
}

Counts load_recorded_counts(const std::filesystem::path& meta_path) {
    const Json meta = load_json(meta_path);
    const Json* recorded = meta.find("counts");
    if (recorded == nullptr || recorded->type != Json::Type::Object) {
        throw std::runtime_error("meta.json: missing object key \"counts\"");
    }
    Counts counts;
    for (const auto& member : recorded->object) {
        if (member.second.type != Json::Type::Number) {
            throw std::runtime_error("meta.json: count \"" + member.first +
                                     "\" is not a number");
        }
        counts[member.first] = std::stoll(member.second.text);
    }
    return counts;
}

void merge_into(Counts& target, const Counts& extra) {
    for (const auto& entry : extra) {
        target[entry.first] = entry.second;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "usage: corpus_stats_check [golden-dir]\n";
        return 2;
    }
    const std::filesystem::path golden =
        argc == 2 ? std::filesystem::path(argv[1])
                  : std::filesystem::path("native") / "golden";

    Counts derived;
    Counts recorded;
    try {
        merge_into(derived, count_transitions(golden / "transitions.json"));
        merge_into(derived, count_q_updates(golden / "q_updates.json"));
        merge_into(derived, count_mcts_oracle(golden / "mcts_oracle.json"));
        merge_into(derived, count_snake_corpus(golden / "snake_corpus.txt"));
        merge_into(derived,
                   count_policies_corpus(golden / "policies_corpus.txt"));
        merge_into(derived, count_mcts_cases(golden / "mcts_cases.txt",
                                             "mcts_searches",
                                             "mcts_root_children"));
        merge_into(derived, count_mcts_cases(golden / "mcts_wall_cases.txt",
                                             "mcts_wall_searches",
                                             "mcts_wall_root_children"));
        recorded = load_recorded_counts(golden / "meta.json");
    } catch (const std::exception& error) {
        std::cerr << "corpus_stats_check: " << error.what() << '\n';
        return 1;
    }

    std::set<std::string> keys;
    for (const auto& entry : derived) {
        keys.insert(entry.first);
    }
    for (const auto& entry : recorded) {
        keys.insert(entry.first);
    }

    std::vector<std::string> failures;
    for (const std::string& key : keys) {
        const auto got_it = derived.find(key);
        const auto want_it = recorded.find(key);
        const bool has_got = got_it != derived.end();
        const bool has_want = want_it != recorded.end();
        std::string status;
        if (!has_got) {
            failures.push_back(key + ": recorded " +
                               std::to_string(want_it->second) +
                               " but not derivable from the corpus files");
            status = "NOT DERIVED";
        } else if (!has_want) {
            failures.push_back(key + ": derived " +
                               std::to_string(got_it->second) +
                               " but absent from meta.json counts");
            status = "NOT RECORDED";
        } else if (got_it->second != want_it->second) {
            failures.push_back(key + ": derived " +
                               std::to_string(got_it->second) +
                               " != recorded " +
                               std::to_string(want_it->second));
            status = "MISMATCH";
        } else {
            status = "ok";
        }
        const std::string got_text =
            has_got ? std::to_string(got_it->second) : "None";
        const std::string want_text =
            has_want ? std::to_string(want_it->second) : "None";
        std::cout << std::left << std::setw(26) << key << " derived="
                  << std::right << std::setw(8) << got_text << " recorded="
                  << std::setw(8) << want_text << ' ' << status << '\n';
    }

    if (!failures.empty()) {
        for (const std::string& failure : failures) {
            std::cout << "FAIL: " << failure << '\n';
        }
        std::cout << "result: FAIL\n";
        return 1;
    }
    std::cout << "result: PASS (" << derived.size()
              << " counts recounted from the corpus files, all match "
                 "meta.json)\n";
    return 0;
}
