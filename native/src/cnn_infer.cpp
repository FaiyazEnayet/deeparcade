// CnnNet implementation. See cnn_infer.hpp for the contract.
#include "narcade/cnn_infer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <utility>

namespace narcade {

namespace {

// ---------------------------------------------------------------------------
// Minimal JSON reader: just enough of RFC 8259 for the DA8 checkpoint
// exports (objects, arrays, strings, numbers, true/false/null),
// following the reader in dqn_infer.cpp. Two extensions the CNN
// format needs: number tokens remember whether they were written as
// integers (the weights hash is computed over a canonical re-rendering
// in which integers and floats render differently, exactly as
// Python's json module round-trips them), and object lookup follows
// Python dict semantics (a duplicate key's last occurrence wins).
// ---------------------------------------------------------------------------
struct JsonValue {
    enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };
    Type type = Type::kNull;
    bool boolean = false;
    double number = 0.0;
    bool integer_token = false;
    std::string raw_token;
    std::string str;
    std::vector<JsonValue> array;
    std::vector<std::pair<std::string, JsonValue>> object;

    const JsonValue* find(const std::string& key) const {
        if (type != Type::kObject) {
            return nullptr;
        }
        const JsonValue* hit = nullptr;
        for (const auto& kv : object) {
            if (kv.first == key) {
                hit = &kv.second;
            }
        }
        return hit;
    }
};

[[noreturn]] void json_fail(const std::string& msg, std::size_t pos) {
    throw CnnCheckpointError("cnn checkpoint JSON: " + msg + " at byte " +
                             std::to_string(pos));
}

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : text_(text) {}

    JsonValue parse() {
        JsonValue v = parse_value();
        skip_ws();
        if (pos_ != text_.size()) {
            json_fail("trailing characters after top-level value", pos_);
        }
        return v;
    }

private:
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
            json_fail("unexpected end of input", pos_);
        }
        return text_[pos_];
    }

    void expect(char c) {
        if (peek() != c) {
            json_fail(std::string("expected '") + c + "'", pos_);
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
                json_fail("invalid literal", pos_);
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
                json_fail("expected object key", pos_);
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
            json_fail("expected ',' or '}' in object", pos_);
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
            json_fail("expected ',' or ']' in array", pos_);
        }
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (pos_ >= text_.size()) {
                json_fail("unterminated string", pos_);
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
                json_fail("unterminated escape", pos_);
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
                    if (pos_ + 4 > text_.size()) {
                        json_fail("truncated \\u escape", pos_);
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
                            json_fail("bad hex digit in \\u escape", pos_);
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
                    json_fail("unknown escape", pos_);
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
        json_fail("invalid literal", pos_);
    }

    JsonValue parse_number() {
        const std::size_t start = pos_;
        const char* begin = text_.c_str() + pos_;
        char* end = nullptr;
        const double v = std::strtod(begin, &end);
        if (end == begin) {
            json_fail("invalid number", pos_);
        }
        pos_ += static_cast<std::size_t>(end - begin);
        JsonValue out;
        out.type = JsonValue::Type::kNumber;
        out.number = v;
        out.raw_token = text_.substr(start, pos_ - start);
        out.integer_token =
            out.raw_token.find_first_of(".eE") == std::string::npos;
        return out;
    }

    const std::string& text_;
    std::size_t pos_ = 0;
};

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), one-shot over a byte string. Needed to verify
// the checkpoint's weights_sha256, which Python computes over the
// canonical JSON rendering of the layers payload.
// ---------------------------------------------------------------------------
std::uint32_t rotr(std::uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

std::string sha256_hex(const std::string& data) {
    static constexpr std::uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b,
        0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
        0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7,
        0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152,
        0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
        0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
        0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                          0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

    std::vector<std::uint8_t> msg(data.begin(), data.end());
    const std::uint64_t bit_len = static_cast<std::uint64_t>(msg.size()) * 8;
    msg.push_back(0x80);
    while (msg.size() % 64 != 56) {
        msg.push_back(0x00);
    }
    for (int i = 7; i >= 0; --i) {
        msg.push_back(static_cast<std::uint8_t>(bit_len >> (i * 8)));
    }

    for (std::size_t block = 0; block < msg.size(); block += 64) {
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            const std::size_t o = block + static_cast<std::size_t>(i) * 4;
            w[i] = (static_cast<std::uint32_t>(msg[o]) << 24) |
                   (static_cast<std::uint32_t>(msg[o + 1]) << 16) |
                   (static_cast<std::uint32_t>(msg[o + 2]) << 8) |
                   static_cast<std::uint32_t>(msg[o + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^
                                     (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^
                                     (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4],
                      f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 =
                hh + s1 + ch + k[i] + w[static_cast<std::size_t>(i)];
            const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = s0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a;
            a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (std::uint32_t word : h) {
        for (int shift = 28; shift >= 0; shift -= 4) {
            out += kHex[(word >> shift) & 0xF];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Canonical JSON rendering for the weights hash: a port of what
// narcade.models.cnn_weights_sha256 feeds to hashlib, namely
// json.dumps({"layers": layers}, sort_keys=True, separators=(",", ":"))
// over the parsed document. Integers render as written; floats render
// with Python's repr (shortest round-trip digits, CPython's fixed vs
// scientific choice), so the rendering matches byte for byte.
// ---------------------------------------------------------------------------
std::string python_float_repr(double v) {
    if (v == 0.0) {
        return std::signbit(v) ? "-0.0" : "0.0";
    }
    const bool neg = std::signbit(v);
    const double a = std::fabs(v);
    // Shortest significant-digit string that round-trips, with the
    // decimal exponent: value = D[0].D[1...] x 10^exp10.
    std::string digits;
    int exp10 = 0;
    for (int prec = 1; prec <= 17; ++prec) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.*e", prec - 1, a);
        if (std::strtod(buf, nullptr) == a) {
            const std::string s(buf);
            const std::size_t epos = s.find('e');
            exp10 = std::stoi(s.substr(epos + 1));
            for (char ch : s.substr(0, epos)) {
                if (ch != '.') {
                    digits += ch;
                }
            }
            break;
        }
    }
    // dtoa convention: value = digits x 10^(decpt - len(digits)).
    const int n = static_cast<int>(digits.size());
    const int decpt = exp10 + 1;
    std::string out;
    if (neg) {
        out += '-';
    }
    if (decpt <= -4 || decpt > 16) {
        // Scientific: one digit, optional fraction, signed exponent
        // of at least two digits.
        out += digits[0];
        if (n > 1) {
            out += '.';
            out += digits.substr(1);
        }
        out += 'e';
        const int shown = decpt - 1;
        out += shown < 0 ? '-' : '+';
        const int mag = shown < 0 ? -shown : shown;
        if (mag < 10) {
            out += '0';
        }
        out += std::to_string(mag);
        return out;
    }
    if (decpt <= 0) {
        out += "0.";
        out.append(static_cast<std::size_t>(-decpt), '0');
        out += digits;
    } else if (decpt >= n) {
        out += digits;
        out.append(static_cast<std::size_t>(decpt - n), '0');
        out += ".0";
    } else {
        out += digits.substr(0, static_cast<std::size_t>(decpt));
        out += '.';
        out += digits.substr(static_cast<std::size_t>(decpt));
    }
    return out;
}

void render_json_string(const std::string& s, std::string& out) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    out += '"';
}

void render_canonical(const JsonValue& v, std::string& out) {
    switch (v.type) {
        case JsonValue::Type::kNull:
            out += "null";
            break;
        case JsonValue::Type::kBool:
            out += v.boolean ? "true" : "false";
            break;
        case JsonValue::Type::kNumber:
            if (v.integer_token) {
                out += v.raw_token == "-0" ? "0" : v.raw_token;
            } else {
                out += python_float_repr(v.number);
            }
            break;
        case JsonValue::Type::kString:
            render_json_string(v.str, out);
            break;
        case JsonValue::Type::kArray: {
            out += '[';
            bool first = true;
            for (const JsonValue& e : v.array) {
                if (!first) {
                    out += ',';
                }
                first = false;
                render_canonical(e, out);
            }
            out += ']';
            break;
        }
        case JsonValue::Type::kObject: {
            // Sorted keys, duplicate keys collapsed last-wins, exactly
            // as json.dumps(sort_keys=True) renders the loaded dict.
            std::map<std::string, const JsonValue*> collapsed;
            for (const auto& kv : v.object) {
                collapsed[kv.first] = &kv.second;
            }
            out += '{';
            bool first = true;
            for (const auto& kv : collapsed) {
                if (!first) {
                    out += ',';
                }
                first = false;
                render_json_string(kv.first, out);
                out += ':';
                render_canonical(*kv.second, out);
            }
            out += '}';
            break;
        }
    }
}

std::string cnn_weights_sha256(const JsonValue& layers) {
    JsonValue root;
    root.type = JsonValue::Type::kObject;
    root.object.emplace_back("layers", layers);
    std::string canonical;
    render_canonical(root, canonical);
    return sha256_hex(canonical);
}

// ---------------------------------------------------------------------------
// Strict checkpoint loader: a port of
// narcade.models.validate_cnn_checkpoint_document plus the tensor
// extraction in load_cnn_checkpoint, in the same validation order.
// ---------------------------------------------------------------------------
const JsonValue& require_field(const JsonValue& obj, const char* key,
                               const char* what) {
    const JsonValue* v = obj.find(key);
    if (v == nullptr) {
        throw CnnCheckpointError(std::string("cnn checkpoint: missing ") +
                                 what);
    }
    return *v;
}

bool is_integral(const JsonValue& v) {
    return v.type == JsonValue::Type::kNumber && std::isfinite(v.number) &&
           std::floor(v.number) == v.number && v.number >= -2147483648.0 &&
           v.number <= 2147483647.0;
}

int field_int(const JsonValue& obj, const char* key, const char* what) {
    const JsonValue& v = require_field(obj, key, what);
    if (!is_integral(v)) {
        throw CnnCheckpointError(std::string("cnn checkpoint: ") + what +
                                 " is not an integer");
    }
    return static_cast<int>(v.number);
}

CnnConfig parse_config(const JsonValue& doc) {
    const JsonValue* raw = doc.find("config");
    if (raw == nullptr || raw->type != JsonValue::Type::kObject) {
        throw CnnCheckpointError("CNN config must be an object");
    }
    const JsonValue& obj = *raw;
    CnnConfig cfg;
    const JsonValue& channels = require_field(obj, "channels", "config.channels");
    if (channels.type != JsonValue::Type::kArray) {
        throw CnnCheckpointError("cnn checkpoint: config.channels is not "
                                 "an array");
    }
    cfg.channels.clear();
    for (const JsonValue& e : channels.array) {
        if (!is_integral(e)) {
            throw CnnCheckpointError("cnn checkpoint: config.channels holds "
                                     "a non-integer");
        }
        cfg.channels.push_back(static_cast<int>(e.number));
    }
    if (obj.find("kernel") != nullptr) {
        cfg.kernel = field_int(obj, "kernel", "config.kernel");
    }
    if (const JsonValue* residual = obj.find("residual")) {
        if (residual->type != JsonValue::Type::kBool) {
            throw CnnCheckpointError("cnn checkpoint: config.residual is "
                                     "not a boolean");
        }
        cfg.residual = residual->boolean;
    }
    if (obj.find("pool_size") != nullptr) {
        cfg.pool_size = field_int(obj, "pool_size", "config.pool_size");
    }
    if (obj.find("hidden") != nullptr) {
        cfg.hidden = field_int(obj, "hidden", "config.hidden");
    }
    // CNNConfig.__post_init__ validation.
    if (cfg.channels.empty()) {
        throw CnnCheckpointError("invalid CNN config: channels must be "
                                 "positive ints, got []");
    }
    for (int c : cfg.channels) {
        if (c <= 0) {
            throw CnnCheckpointError("invalid CNN config: channels must be "
                                     "positive ints");
        }
    }
    if (cfg.kernel <= 0 || cfg.kernel % 2 == 0) {
        throw CnnCheckpointError("invalid CNN config: kernel must be a "
                                 "positive odd int");
    }
    if (cfg.pool_size < 1 || cfg.pool_size > kCnnBoard) {
        throw CnnCheckpointError("invalid CNN config: pool_size must be "
                                 "in 1..12");
    }
    if (cfg.hidden <= 0) {
        throw CnnCheckpointError("invalid CNN config: hidden must be "
                                 "positive");
    }
    return cfg;
}

struct ExpectedLayer {
    std::string name;
    std::vector<std::int64_t> shape;
};

std::vector<ExpectedLayer> expected_layers(const CnnConfig& cfg) {
    std::vector<ExpectedLayer> out;
    int in_ch = kCnnChannels;
    for (std::size_t i = 0; i < cfg.channels.size(); ++i) {
        const std::int64_t ch = cfg.channels[i];
        const std::string base = "convs." + std::to_string(i);
        out.push_back({base + ".weight", {ch, in_ch, cfg.kernel, cfg.kernel}});
        out.push_back({base + ".bias", {ch}});
        in_ch = cfg.channels[i];
    }
    const std::int64_t last = cfg.channels.back();
    if (cfg.residual) {
        out.push_back({"res1.weight", {last, last, 3, 3}});
        out.push_back({"res1.bias", {last}});
        out.push_back({"res2.weight", {last, last, 3, 3}});
        out.push_back({"res2.bias", {last}});
    }
    const std::int64_t features = last * cfg.pool_size * cfg.pool_size;
    out.push_back({"head1.weight", {cfg.hidden, features + 4}});
    out.push_back({"head1.bias", {cfg.hidden}});
    out.push_back({"head2.weight", {kCnnActions, cfg.hidden}});
    out.push_back({"head2.bias", {kCnnActions}});
    return out;
}

// Recursive values walk: verifies the nested shape equals the
// declared shape (which also enforces rectangularity), every leaf is
// a finite number, and flattens row-major into `out`.
void flatten_values(const JsonValue& v,
                    const std::vector<std::int64_t>& shape, std::size_t depth,
                    const std::string& name, std::vector<double>& out) {
    if (depth == shape.size()) {
        if (v.type != JsonValue::Type::kNumber ||
            !std::isfinite(v.number)) {
            throw CnnCheckpointError("layer " + name +
                                     " holds a non-numeric or non-finite "
                                     "value");
        }
        out.push_back(v.number);
        return;
    }
    if (v.type != JsonValue::Type::kArray ||
        static_cast<std::int64_t>(v.array.size()) != shape[depth]) {
        throw CnnCheckpointError("layer " + name +
                                 " shape does not match the recorded "
                                 "config (nested values)");
    }
    for (const JsonValue& e : v.array) {
        flatten_values(e, shape, depth + 1, name, out);
    }
}

}  // namespace

CnnWeights load_cnn_weights(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cnn weights: cannot open " + path);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    const JsonValue doc = JsonParser(ss.str()).parse();
    if (doc.type != JsonValue::Type::kObject) {
        throw CnnCheckpointError("checkpoint document must be a JSON "
                                 "object");
    }
    const JsonValue* schema = doc.find("schema_version");
    if (schema == nullptr || schema->type != JsonValue::Type::kNumber ||
        schema->number != 1.0) {
        throw CnnCheckpointError(
            "unsupported CNN checkpoint schema_version; this code reads "
            "schema_version 1 only. Re-export with "
            "scripts/train_cnn_dqn.py");
    }
    const JsonValue* kind = doc.find("kind");
    if (kind == nullptr || kind->type != JsonValue::Type::kString ||
        kind->str != "deeparcade-cnn-dqn-checkpoint") {
        throw CnnCheckpointError(
            "checkpoint kind != expected 'deeparcade-cnn-dqn-checkpoint'");
    }
    const JsonValue* encoder = doc.find("encoder");
    if (encoder == nullptr || encoder->type != JsonValue::Type::kObject ||
        encoder->find("name") == nullptr ||
        encoder->find("name")->type != JsonValue::Type::kString ||
        encoder->find("name")->str != "encode_cnn_dqn" ||
        encoder->find("revision") == nullptr ||
        encoder->find("revision")->type != JsonValue::Type::kNumber ||
        encoder->find("revision")->number != 1.0) {
        throw CnnCheckpointError(
            "checkpoint encoder does not match the declared encoder "
            "encode_cnn_dqn revision 1");
    }
    const JsonValue* regime = doc.find("float_regime");
    if (regime == nullptr || regime->type != JsonValue::Type::kString ||
        regime->str != "float32") {
        throw CnnCheckpointError(
            "checkpoint float_regime != declared 'float32'");
    }

    CnnWeights out;
    out.config = parse_config(doc);
    const std::vector<ExpectedLayer> expected = expected_layers(out.config);

    const JsonValue& layers = require_field(doc, "layers", "layers");
    if (layers.type != JsonValue::Type::kArray) {
        throw CnnCheckpointError("checkpoint layers must be a list");
    }
    if (layers.array.size() != expected.size()) {
        throw CnnCheckpointError(
            "checkpoint layer names do not match the recorded config "
            "(layer count differs)");
    }
    std::vector<CnnTensor> tensors;
    for (std::size_t i = 0; i < layers.array.size(); ++i) {
        const JsonValue& layer = layers.array[i];
        if (layer.type != JsonValue::Type::kObject) {
            throw CnnCheckpointError("checkpoint layer is not an object");
        }
        const JsonValue* name = layer.find("name");
        if (name == nullptr || name->type != JsonValue::Type::kString ||
            name->str != expected[i].name) {
            throw CnnCheckpointError(
                "checkpoint layer names do not match the recorded "
                "config (position " +
                std::to_string(i) + ")");
        }
        // Exactly the keys 'name', 'shape', 'values' (distinct keys,
        // duplicate-collapsed as in Python).
        std::map<std::string, const JsonValue*> keys;
        for (const auto& kv : layer.object) {
            keys[kv.first] = &kv.second;
        }
        if (keys.size() != 3 || keys.count("name") == 0 ||
            keys.count("shape") == 0 || keys.count("values") == 0) {
            throw CnnCheckpointError("layer " + expected[i].name +
                                     " must hold exactly keys 'name', "
                                     "'shape', 'values'");
        }
        const JsonValue& shape = *keys["shape"];
        if (shape.type != JsonValue::Type::kArray ||
            shape.array.size() != expected[i].shape.size()) {
            throw CnnCheckpointError("layer " + expected[i].name +
                                     " shape != expected for the "
                                     "recorded config");
        }
        for (std::size_t d = 0; d < shape.array.size(); ++d) {
            if (!is_integral(shape.array[d]) ||
                static_cast<std::int64_t>(shape.array[d].number) !=
                    expected[i].shape[d]) {
                throw CnnCheckpointError("layer " + expected[i].name +
                                         " shape != expected for the "
                                         "recorded config");
            }
        }
        CnnTensor tensor;
        tensor.shape = expected[i].shape;
        flatten_values(*keys["values"], expected[i].shape, 0,
                       expected[i].name, tensor.values);
        tensors.push_back(std::move(tensor));
    }

    const std::string digest = cnn_weights_sha256(layers);
    const JsonValue* recorded = doc.find("weights_sha256");
    if (recorded == nullptr || recorded->type != JsonValue::Type::kString ||
        recorded->str != digest) {
        throw CnnCheckpointError(
            "checkpoint weights_sha256 does not match the weights "
            "payload; the document is corrupt or tampered");
    }
    out.weights_sha256 = digest;

    const JsonValue* metadata = doc.find("metadata");
    if (metadata == nullptr || metadata->type != JsonValue::Type::kObject) {
        throw CnnCheckpointError(
            "CNN checkpoint metadata must be an object with exactly "
            "keys 'training_args', 'data_key_range', 'code_commit'");
    }
    {
        std::map<std::string, const JsonValue*> keys;
        for (const auto& kv : metadata->object) {
            keys[kv.first] = &kv.second;
        }
        if (keys.size() != 3 || keys.count("training_args") == 0 ||
            keys.count("data_key_range") == 0 ||
            keys.count("code_commit") == 0) {
            throw CnnCheckpointError(
                "CNN checkpoint metadata must be an object with "
                "exactly keys 'training_args', 'data_key_range', "
                "'code_commit' (values may be null when unknown)");
        }
    }

    // Distribute the validated tensors into the net layout. The
    // layer order was verified against `expected`, so positions are
    // known: conv pairs, optional residual pairs, then the head.
    std::size_t pos = 0;
    for (std::size_t i = 0; i < out.config.channels.size(); ++i) {
        out.conv_weights.push_back(std::move(tensors[pos++]));
        out.conv_biases.push_back(std::move(tensors[pos++]));
    }
    if (out.config.residual) {
        out.res1_weight = std::move(tensors[pos++]);
        out.res1_bias = std::move(tensors[pos++]);
        out.res2_weight = std::move(tensors[pos++]);
        out.res2_bias = std::move(tensors[pos++]);
    }
    out.head1_weight = std::move(tensors[pos++]);
    out.head1_bias = std::move(tensors[pos++]);
    out.head2_weight = std::move(tensors[pos++]);
    out.head2_bias = std::move(tensors[pos++]);
    return out;
}

int CnnWeights::num_params() const {
    std::int64_t total = 0;
    auto add = [&total](const CnnTensor& t) { total += t.numel(); };
    for (const CnnTensor& t : conv_weights) add(t);
    for (const CnnTensor& t : conv_biases) add(t);
    if (config.residual) {
        add(res1_weight);
        add(res1_bias);
        add(res2_weight);
        add(res2_bias);
    }
    add(head1_weight);
    add(head1_bias);
    add(head2_weight);
    add(head2_bias);
    return static_cast<int>(total);
}

CnnEncoding encode_cnn(const std::vector<Cell>& body,
                       std::optional<Cell> food, int heading) {
    if (body.empty()) {
        throw std::invalid_argument("encode_cnn: empty body");
    }
    if (heading < 0 || heading > 3) {
        throw std::invalid_argument("encode_cnn: heading out of range");
    }
    auto in_board = [](const Cell& c) {
        return c.r >= 0 && c.r < kCnnBoard && c.c >= 0 && c.c < kCnnBoard;
    };
    for (const Cell& c : body) {
        if (!in_board(c)) {
            throw std::invalid_argument("encode_cnn: body cell off board");
        }
    }
    if (food.has_value() && !in_board(*food)) {
        throw std::invalid_argument("encode_cnn: food off board");
    }
    // narcade/models.py encode_cnn_dqn, in the same write order:
    // head plane from body[-1], body plane from body[:-1], food plane
    // (empty when there is no food), heading one-hot.
    CnnEncoding enc{};
    const Cell head = body.back();
    enc.spatial[static_cast<std::size_t>(head.r * kCnnBoard + head.c)] =
        1.0f;
    for (std::size_t i = 0; i + 1 < body.size(); ++i) {
        enc.spatial[144 + static_cast<std::size_t>(body[i].r * kCnnBoard +
                                                   body[i].c)] = 1.0f;
    }
    if (food.has_value()) {
        enc.spatial[288 + static_cast<std::size_t>(food->r * kCnnBoard +
                                                   food->c)] = 1.0f;
    }
    enc.heading[static_cast<std::size_t>(heading)] = 1.0f;
    return enc;
}

CnnEncoding encode_cnn(const SnakeEnv& env) {
    const std::vector<Cell> body(env.body().begin(), env.body().end());
    return encode_cnn(body, env.food(), env.heading());
}

namespace {

// Feature map in (C, H, W) row-major order, double precision.
using FeatureMap = std::vector<double>;

FeatureMap conv2d(const FeatureMap& x, int in_ch, int h, int w,
                  const CnnTensor& weight, const CnnTensor& bias,
                  int kernel) {
    const int out_ch = static_cast<int>(weight.shape[0]);
    const int pad = kernel / 2;
    FeatureMap out(static_cast<std::size_t>(out_ch) *
                   static_cast<std::size_t>(h) * static_cast<std::size_t>(w));
    const std::vector<double>& wv = weight.values;
    for (int oc = 0; oc < out_ch; ++oc) {
        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < w; ++c) {
                // Accumulation order (ic, kh, kw) matches
                // narcade.models._conv2d_numpy.
                double acc = bias.values[static_cast<std::size_t>(oc)];
                for (int ic = 0; ic < in_ch; ++ic) {
                    for (int kh = 0; kh < kernel; ++kh) {
                        const int sr = r + kh - pad;
                        if (sr < 0 || sr >= h) {
                            continue;
                        }
                        for (int kw = 0; kw < kernel; ++kw) {
                            const int sc = c + kw - pad;
                            if (sc < 0 || sc >= w) {
                                continue;
                            }
                            acc += wv[(static_cast<std::size_t>(
                                           (oc * in_ch + ic) * kernel +
                                           kh) *
                                       static_cast<std::size_t>(kernel)) +
                                      static_cast<std::size_t>(kw)] *
                                   x[(static_cast<std::size_t>(ic) *
                                          static_cast<std::size_t>(h) +
                                      static_cast<std::size_t>(sr)) *
                                         static_cast<std::size_t>(w) +
                                     static_cast<std::size_t>(sc)];
                        }
                    }
                }
                out[(static_cast<std::size_t>(oc) *
                         static_cast<std::size_t>(h) +
                     static_cast<std::size_t>(r)) *
                        static_cast<std::size_t>(w) +
                    static_cast<std::size_t>(c)] = acc;
            }
        }
    }
    return out;
}

// ReLU, then round to float32 (the declared parity regime: hidden
// activations are rounded to float32 after each ReLU, matching
// Float32Array storage in the browser mirror).
void relu_round(FeatureMap& x) {
    for (double& v : x) {
        v = static_cast<double>(static_cast<float>(v > 0.0 ? v : 0.0));
    }
}

// Adaptive average pooling with torch semantics: output cell (i, j)
// averages rows [floor(i*H/oh), ceil((i+1)*H/oh)) and the matching
// column window.
FeatureMap adaptive_avg_pool(const FeatureMap& x, int ch, int h, int w,
                             int pool) {
    FeatureMap out(static_cast<std::size_t>(ch) *
                   static_cast<std::size_t>(pool) *
                   static_cast<std::size_t>(pool));
    for (int c = 0; c < ch; ++c) {
        for (int i = 0; i < pool; ++i) {
            const int r0 = (i * h) / pool;
            const int r1 = ((i + 1) * h + pool - 1) / pool;
            for (int j = 0; j < pool; ++j) {
                const int c0 = (j * w) / pool;
                const int c1 = ((j + 1) * w + pool - 1) / pool;
                double sum = 0.0;
                for (int r = r0; r < r1; ++r) {
                    for (int cc = c0; cc < c1; ++cc) {
                        sum += x[(static_cast<std::size_t>(c) *
                                      static_cast<std::size_t>(h) +
                                  static_cast<std::size_t>(r)) *
                                     static_cast<std::size_t>(w) +
                                 static_cast<std::size_t>(cc)];
                    }
                }
                out[(static_cast<std::size_t>(c) *
                         static_cast<std::size_t>(pool) +
                     static_cast<std::size_t>(i)) *
                        static_cast<std::size_t>(pool) +
                    static_cast<std::size_t>(j)] =
                    sum / static_cast<double>((r1 - r0) * (c1 - c0));
            }
        }
    }
    return out;
}

}  // namespace

CnnActivations CnnNet::forward(const CnnEncoding& input) const {
    const CnnConfig& cfg = weights_.config;
    FeatureMap x(input.spatial.begin(), input.spatial.end());
    int in_ch = kCnnChannels;
    for (std::size_t i = 0; i < cfg.channels.size(); ++i) {
        x = conv2d(x, in_ch, kCnnBoard, kCnnBoard, weights_.conv_weights[i],
                   weights_.conv_biases[i], cfg.kernel);
        relu_round(x);
        in_ch = cfg.channels[i];
    }
    if (cfg.residual) {
        FeatureMap r = conv2d(x, in_ch, kCnnBoard, kCnnBoard,
                              weights_.res1_weight, weights_.res1_bias, 3);
        relu_round(r);
        r = conv2d(r, in_ch, kCnnBoard, kCnnBoard, weights_.res2_weight,
                   weights_.res2_bias, 3);
        for (std::size_t k = 0; k < x.size(); ++k) {
            const double sum = x[k] + r[k];
            x[k] = static_cast<double>(
                static_cast<float>(sum > 0.0 ? sum : 0.0));
        }
    }
    const FeatureMap pooled =
        adaptive_avg_pool(x, in_ch, kCnnBoard, kCnnBoard, cfg.pool_size);

    CnnActivations out;
    out.features = pooled;
    for (float hv : input.heading) {
        out.features.push_back(static_cast<double>(hv));
    }
    // head1: Linear(features + 4, hidden) + ReLU (rounded to float32).
    const std::vector<double>& hw = weights_.head1_weight.values;
    const int in_dim = static_cast<int>(out.features.size());
    std::vector<double> h1(static_cast<std::size_t>(cfg.hidden));
    for (int j = 0; j < cfg.hidden; ++j) {
        double acc = weights_.head1_bias.values[static_cast<std::size_t>(j)];
        for (int i = 0; i < in_dim; ++i) {
            acc += hw[static_cast<std::size_t>(j) *
                          static_cast<std::size_t>(in_dim) +
                      static_cast<std::size_t>(i)] *
                   out.features[static_cast<std::size_t>(i)];
        }
        h1[static_cast<std::size_t>(j)] =
            static_cast<double>(static_cast<float>(acc > 0.0 ? acc : 0.0));
    }
    // head2: Linear(hidden, 3); no rounding after the last layer.
    const std::vector<double>& ow = weights_.head2_weight.values;
    for (int a = 0; a < kCnnActions; ++a) {
        double acc = weights_.head2_bias.values[static_cast<std::size_t>(a)];
        for (int i = 0; i < cfg.hidden; ++i) {
            acc += ow[static_cast<std::size_t>(a) *
                          static_cast<std::size_t>(cfg.hidden) +
                      static_cast<std::size_t>(i)] *
                   h1[static_cast<std::size_t>(i)];
        }
        out.logits[static_cast<std::size_t>(a)] = acc;
    }
    return out;
}

int CnnNet::act(const CnnEncoding& input) const {
    const CnnActivations out = forward(input);
    int best = 0;
    for (int a = 1; a < kCnnActions; ++a) {
        if (out.logits[static_cast<std::size_t>(a)] >
            out.logits[static_cast<std::size_t>(best)]) {
            best = a;
        }
    }
    return best;
}

}  // namespace narcade
