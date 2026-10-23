// DqnNet implementation (CPP-06d). See dqn_infer.hpp for the contract.
#include "narcade/dqn_infer.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace narcade {

namespace {

// ---------------------------------------------------------------------------
// Minimal JSON reader: just enough of RFC 8259 for runs/dqn_weights.json
// (objects, arrays, strings, numbers, true/false/null). Numbers are read
// with strtod; the export's doubles are exact decimal renderings of
// float32 values, so the double -> float conversion below is lossless.
// ---------------------------------------------------------------------------
struct JsonValue {
    enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };
    Type type = Type::kNull;
    bool boolean = false;
    double number = 0.0;
    std::string str;
    std::vector<JsonValue> array;
    std::vector<std::pair<std::string, JsonValue>> object;

    const JsonValue* find(const std::string& key) const {
        if (type != Type::kObject) {
            return nullptr;
        }
        for (const auto& kv : object) {
            if (kv.first == key) {
                return &kv.second;
            }
        }
        return nullptr;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : text_(text) {}

    JsonValue parse() {
        JsonValue v = parse_value();
        skip_ws();
        if (pos_ != text_.size()) {
            fail("trailing characters after top-level value");
        }
        return v;
    }

private:
    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("dqn weights JSON: " + msg + " at byte " +
                                 std::to_string(pos_));
    }

    void skip_ws() {
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    char peek() {
        skip_ws();
        if (pos_ >= text_.size()) {
            fail("unexpected end of input");
        }
        return text_[pos_];
    }

    void expect(char c) {
        if (peek() != c) {
            fail(std::string("expected '") + c + "'");
        }
        ++pos_;
    }

    JsonValue parse_value() {
        const char c = peek();
        if (c == '{') {
            return parse_object();
        }
        if (c == '[') {
            return parse_array();
        }
        if (c == '"') {
            JsonValue v;
            v.type = JsonValue::Type::kString;
            v.str = parse_string();
            return v;
        }
        if (c == 't' || c == 'f') {
            return parse_bool();
        }
        if (c == 'n') {
            if (text_.compare(pos_, 4, "null") != 0) {
                fail("invalid literal");
            }
            pos_ += 4;
            return JsonValue{};
        }
        return parse_number();
    }

    JsonValue parse_object() {
        JsonValue v;
        v.type = JsonValue::Type::kObject;
        expect('{');
        if (peek() == '}') {
            ++pos_;
            return v;
        }
        while (true) {
            if (peek() != '"') {
                fail("expected object key");
            }
            std::string key = parse_string();
            expect(':');
            JsonValue val = parse_value();
            v.object.emplace_back(std::move(key), std::move(val));
            const char c = peek();
            if (c == ',') {
                ++pos_;
                continue;
            }
            if (c == '}') {
                ++pos_;
                return v;
            }
            fail("expected ',' or '}' in object");
        }
    }

    JsonValue parse_array() {
        JsonValue v;
        v.type = JsonValue::Type::kArray;
        expect('[');
        if (peek() == ']') {
            ++pos_;
            return v;
        }
        while (true) {
            v.array.push_back(parse_value());
            const char c = peek();
            if (c == ',') {
                ++pos_;
                continue;
            }
            if (c == ']') {
                ++pos_;
                return v;
            }
            fail("expected ',' or ']' in array");
        }
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
            const char e = text_[pos_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    // Decode \uXXXX as UTF-8 (BMP only; the weight
                    // file's keys are plain ASCII, this is for
                    // robustness, not a feature).
                    if (pos_ + 4 > text_.size()) {
                        fail("truncated \\u escape");
                    }
                    unsigned code = 0;
                    for (int i = 0; i < 4; ++i) {
                        const char h = text_[pos_++];
                        code <<= 4;
                        if (h >= '0' && h <= '9') {
                            code += static_cast<unsigned>(h - '0');
                        } else if (h >= 'a' && h <= 'f') {
                            code += static_cast<unsigned>(h - 'a' + 10);
                        } else if (h >= 'A' && h <= 'F') {
                            code += static_cast<unsigned>(h - 'A' + 10);
                        } else {
                            fail("bad hex digit in \\u escape");
                        }
                    }
                    if (code < 0x80) {
                        out += static_cast<char>(code);
                    } else if (code < 0x800) {
                        out += static_cast<char>(0xC0 | (code >> 6));
                        out += static_cast<char>(0x80 | (code & 0x3F));
                    } else {
                        out += static_cast<char>(0xE0 | (code >> 12));
                        out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                        out += static_cast<char>(0x80 | (code & 0x3F));
                    }
                    break;
                }
                default:
                    fail("unknown escape");
            }
        }
    }

    JsonValue parse_bool() {
        JsonValue v;
        v.type = JsonValue::Type::kBool;
        if (text_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            v.boolean = true;
            return v;
        }
        if (text_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            v.boolean = false;
            return v;
        }
        fail("invalid literal");
    }

    JsonValue parse_number() {
        const char* begin = text_.c_str() + pos_;
        char* end = nullptr;
        const double v = std::strtod(begin, &end);
        if (end == begin) {
            fail("invalid number");
        }
        pos_ += static_cast<std::size_t>(end - begin);
        JsonValue out;
        out.type = JsonValue::Type::kNumber;
        out.number = v;
        return out;
    }

    const std::string& text_;
    std::size_t pos_ = 0;
};

const JsonValue& require(const JsonValue* v, const char* what) {
    if (v == nullptr) {
        throw std::runtime_error(std::string("dqn weights JSON: missing ") +
                                 what);
    }
    return *v;
}

std::vector<float> number_array(const JsonValue& v, const std::string& what) {
    if (v.type != JsonValue::Type::kArray) {
        throw std::runtime_error("dqn weights JSON: " + what +
                                 " is not an array");
    }
    std::vector<float> out;
    out.reserve(v.array.size());
    for (const JsonValue& e : v.array) {
        if (e.type != JsonValue::Type::kNumber) {
            throw std::runtime_error("dqn weights JSON: " + what +
                                     " contains a non-number");
        }
        out.push_back(static_cast<float>(e.number));
    }
    return out;
}

void check_shape(bool ok, const std::string& what) {
    if (!ok) {
        throw std::runtime_error("dqn weights: shape mismatch for " + what);
    }
}

}  // namespace

std::array<float, kDqnInputDim> encode_dqn(const std::vector<Cell>& body,
                                           Cell food, int heading) {
    if (body.empty()) {
        throw std::invalid_argument("encode_dqn: empty body");
    }
    if (heading < 0 || heading > 3) {
        throw std::invalid_argument("encode_dqn: heading out of range");
    }
    auto in_board = [](const Cell& c) {
        return c.r >= 0 && c.r < 12 && c.c >= 0 && c.c < 12;
    };
    for (const Cell& c : body) {
        if (!in_board(c)) {
            throw std::invalid_argument("encode_dqn: body cell off board");
        }
    }
    if (!in_board(food)) {
        throw std::invalid_argument("encode_dqn: food off board");
    }
    // scripts/train_dqn.py lines 38-44, in the same write order.
    std::array<float, kDqnInputDim> v{};
    const Cell head = body.back();
    v[static_cast<std::size_t>(head.r * 12 + head.c)] = 1.0f;
    for (std::size_t i = 0; i + 1 < body.size(); ++i) {
        v[144 + static_cast<std::size_t>(body[i].r * 12 + body[i].c)] = 1.0f;
    }
    v[288 + static_cast<std::size_t>(food.r * 12 + food.c)] = 1.0f;
    v[432 + static_cast<std::size_t>(heading)] = 1.0f;
    return v;
}

std::array<float, kDqnInputDim> encode_dqn(const SnakeEnv& env) {
    if (!env.food().has_value()) {
        throw std::runtime_error(
            "encode_dqn: env has no food (terminal-win state)");
    }
    const std::vector<Cell> body(env.body().begin(), env.body().end());
    return encode_dqn(body, *env.food(), env.heading());
}

DqnWeights load_dqn_weights(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("dqn weights: cannot open " + path);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    const JsonValue root = JsonParser(ss.str()).parse();
    if (root.type != JsonValue::Type::kObject) {
        throw std::runtime_error("dqn weights JSON: top level is not an "
                                 "object");
    }
    DqnWeights out;
    const JsonValue& dim = require(root.find("input_dim"), "input_dim");
    if (dim.type != JsonValue::Type::kNumber) {
        throw std::runtime_error("dqn weights JSON: input_dim not a number");
    }
    out.input_dim = static_cast<int>(dim.number);
    check_shape(out.input_dim == kDqnInputDim, "input_dim");

    const JsonValue& layers = require(root.find("layers"), "layers");
    if (layers.type != JsonValue::Type::kArray || layers.array.size() != 3) {
        throw std::runtime_error("dqn weights JSON: layers is not a "
                                 "3-element array");
    }
    // PyTorch nn.Linear layout, as exported: w is [out][in].
    constexpr int kOut[3] = {kDqnHidden1, kDqnHidden2, kDqnActions};
    constexpr int kIn[3] = {kDqnInputDim, kDqnHidden1, kDqnHidden2};
    for (int l = 0; l < 3; ++l) {
        const JsonValue& layer = layers.array[static_cast<std::size_t>(l)];
        const JsonValue& w = require(layer.find("w"), "layers[].w");
        const JsonValue& b = require(layer.find("b"), "layers[].b");
        if (w.type != JsonValue::Type::kArray) {
            throw std::runtime_error("dqn weights JSON: w is not an array");
        }
        DqnLayer dl;
        dl.w.reserve(w.array.size());
        for (const JsonValue& row : w.array) {
            dl.w.push_back(number_array(row, "w row"));
        }
        dl.b = number_array(b, "b");
        check_shape(static_cast<int>(dl.w.size()) == kOut[l],
                    "layer " + std::to_string(l) + " w rows");
        for (const std::vector<float>& row : dl.w) {
            check_shape(static_cast<int>(row.size()) == kIn[l],
                        "layer " + std::to_string(l) + " w cols");
        }
        check_shape(static_cast<int>(dl.b.size()) == kOut[l],
                    "layer " + std::to_string(l) + " b");
        out.layers[static_cast<std::size_t>(l)] = std::move(dl);
    }
    return out;
}

namespace {

template <std::size_t NIn, std::size_t NOut>
void linear(const DqnLayer& layer, const std::array<float, NIn>& x,
            std::array<float, NOut>& y, bool relu) {
    for (std::size_t j = 0; j < NOut; ++j) {
        const std::vector<float>& w = layer.w[j];
        float acc = layer.b[j];
        for (std::size_t i = 0; i < NIn; ++i) {
            acc += w[i] * x[i];
        }
        y[j] = relu && acc < 0.0f ? 0.0f : acc;
    }
}

}  // namespace

DqnActivations DqnNet::forward(
    const std::array<float, kDqnInputDim>& input) const {
    DqnActivations out;
    linear<kDqnInputDim, kDqnHidden1>(weights_.layers[0], input, out.h1,
                                      true);
    linear<kDqnHidden1, kDqnHidden2>(weights_.layers[1], out.h1, out.h2,
                                     true);
    linear<kDqnHidden2, kDqnActions>(weights_.layers[2], out.h2, out.logits,
                                     false);
    return out;
}

int DqnNet::act(const std::array<float, kDqnInputDim>& input) const {
    const DqnActivations out = forward(input);
    int best = 0;
    for (int a = 1; a < kDqnActions; ++a) {
        if (out.logits[static_cast<std::size_t>(a)] >
            out.logits[static_cast<std::size_t>(best)]) {
            best = a;
        }
    }
    return best;
}

}  // namespace narcade
