// rng_trace_test: replay native/golden/rng_traces.txt against the
// CPython-compatible PythonRandom and require exact agreement.
//
// Trace format (see tools/native_reference.py):
//   SEED <n>        fresh Random(n)
//   R <hex64>       random() -> IEEE754 bit pattern, big-endian hex
//   B <k> <hex>     getrandbits(k)
//   G <n> <v>       randrange(n)
//   C <len> <idx>   choice(list(range(len))) == idx
//   I <v>           randint(0, 999999)
//   SNAPSEED <n>    fresh Random(n); 100 R lines precede the snapshot
//   SNAPSHOT        snapshot point; the ops that follow must be
//                   reproducible verbatim from the restored state
//
// Exit status is non-zero if any comparison mismatches.
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "narcade/rng.hpp"

namespace {

enum class Kind { kSeed, kSnapSeed, kSnapshot, kRandom, kBits, kRandRange, kChoice, kRandInt };

struct Op {
    Kind kind;
    std::int64_t arg = 0;      // seed, or k for B, or n for G, or len for C
    std::uint64_t expected = 0;  // expected value (bits pattern for R)
    int lineno = 0;
};

bool parse_u64(const std::string& s, std::uint64_t& out) {
    try {
        std::size_t pos = 0;
        out = std::stoull(s, &pos);
        return pos == s.size();
    } catch (...) {
        return false;
    }
}

bool parse_hex_u64(const std::string& s, std::uint64_t& out) {
    if (s.empty() || s.size() > 16) {
        return false;
    }
    std::uint64_t v = 0;
    for (char c : s) {
        v <<= 4;
        if (c >= '0' && c <= '9') {
            v |= static_cast<std::uint64_t>(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            v |= static_cast<std::uint64_t>(c - 'a' + 10);
        } else if (c >= 'A' && c <= 'F') {
            v |= static_cast<std::uint64_t>(c - 'A' + 10);
        } else {
            return false;
        }
    }
    out = v;
    return true;
}

bool parse_i64(const std::string& s, std::int64_t& out) {
    try {
        std::size_t pos = 0;
        out = std::stoll(s, &pos);
        return pos == s.size();
    } catch (...) {
        return false;
    }
}

bool parse_line(const std::string& line, int lineno, Op& op) {
    std::istringstream iss(line);
    std::string tag;
    if (!(iss >> tag)) {
        return false;
    }
    op.lineno = lineno;
    std::string x, y;
    if (tag == "SEED" || tag == "SNAPSEED") {
        if (!(iss >> x) || !parse_i64(x, op.arg)) return false;
        op.kind = (tag == "SEED") ? Kind::kSeed : Kind::kSnapSeed;
        return true;
    }
    if (tag == "SNAPSHOT") {
        op.kind = Kind::kSnapshot;
        return true;
    }
    if (tag == "R") {
        if (!(iss >> x) || !parse_hex_u64(x, op.expected)) return false;
        op.kind = Kind::kRandom;
        return true;
    }
    if (tag == "B") {
        if (!(iss >> x >> y) || !parse_i64(x, op.arg) ||
            !parse_hex_u64(y, op.expected)) {
            return false;
        }
        op.kind = Kind::kBits;
        return true;
    }
    if (tag == "G") {
        std::uint64_t n = 0;
        if (!(iss >> x >> y) || !parse_u64(x, n) || !parse_u64(y, op.expected)) {
            return false;
        }
        op.arg = static_cast<std::int64_t>(n);
        op.kind = Kind::kRandRange;
        return true;
    }
    if (tag == "C") {
        std::uint64_t n = 0;
        if (!(iss >> x >> y) || !parse_u64(x, n) || !parse_u64(y, op.expected)) {
            return false;
        }
        op.arg = static_cast<std::int64_t>(n);
        op.kind = Kind::kChoice;
        return true;
    }
    if (tag == "I") {
        if (!(iss >> x) || !parse_u64(x, op.expected)) return false;
        op.kind = Kind::kRandInt;
        return true;
    }
    return false;
}

// Execute one value-producing op; returns the produced value (for R,
// the IEEE754 bit pattern of the double).
std::uint64_t execute(const Op& op, narcade::PythonRandom& rng) {
    switch (op.kind) {
        case Kind::kRandom: {
            const double d = rng.random();
            std::uint64_t bits = 0;
            std::memcpy(&bits, &d, sizeof(bits));
            return bits;
        }
        case Kind::kBits:
            return rng.getrandbits(static_cast<int>(op.arg));
        case Kind::kRandRange:
            return rng.randrange(static_cast<std::uint64_t>(op.arg));
        case Kind::kChoice:
            return rng.choice_index(static_cast<std::uint64_t>(op.arg));
        case Kind::kRandInt:
            return static_cast<std::uint64_t>(rng.randint(0, 999999));
        default:
            return 0;
    }
}

bool is_value_op(Kind kind) {
    return kind == Kind::kRandom || kind == Kind::kBits ||
           kind == Kind::kRandRange || kind == Kind::kChoice ||
           kind == Kind::kRandInt;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string path =
        argc > 1 ? argv[1] : "native/golden/rng_traces.txt";
    std::ifstream in(path);
    if (!in) {
        std::cerr << "error: cannot open " << path << "\n";
        return 2;
    }

    std::vector<Op> ops;
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        Op op;
        if (!parse_line(line, lineno, op)) {
            std::cerr << "error: unparsable trace line " << lineno << ": "
                      << line << "\n";
            return 2;
        }
        ops.push_back(op);
    }

    std::uint64_t ops_checked = 0;
    std::uint64_t mismatches = 0;
    int first_mismatch_line = -1;
    auto record = [&](bool ok, int line_no) {
        ++ops_checked;
        if (!ok) {
            ++mismatches;
            if (first_mismatch_line < 0) {
                first_mismatch_line = line_no;
            }
        }
    };

    std::optional<narcade::PythonRandom> rng;
    std::size_t i = 0;
    while (i < ops.size()) {
        const Op& op = ops[i];
        if (op.kind == Kind::kSeed || op.kind == Kind::kSnapSeed) {
            rng.emplace(op.arg);
            ++i;
            continue;
        }
        if (op.kind == Kind::kSnapshot) {
            if (!rng.has_value()) {
                std::cerr << "error: SNAPSHOT before any seed (line "
                          << op.lineno << ")\n";
                return 2;
            }
            const narcade::PythonRandomState snapshot = rng->save();
            const std::size_t seg_begin = i + 1;
            // First pass: replay the segment, checking against the file
            // and recording produced values.
            std::vector<std::uint64_t> first_pass;
            first_pass.reserve(ops.size() - seg_begin);
            for (std::size_t j = seg_begin; j < ops.size(); ++j) {
                if (!is_value_op(ops[j].kind)) {
                    std::cerr << "error: non-value op inside snapshot "
                                 "segment (line "
                              << ops[j].lineno << ")\n";
                    return 2;
                }
                const std::uint64_t produced = execute(ops[j], *rng);
                first_pass.push_back(produced);
                record(produced == ops[j].expected, ops[j].lineno);
            }
            // Restore the snapshot and replay the same segment again;
            // every result must be identical to the first pass.
            rng->restore(snapshot);
            for (std::size_t j = seg_begin; j < ops.size(); ++j) {
                const std::uint64_t produced = execute(ops[j], *rng);
                record(produced == first_pass[j - seg_begin], ops[j].lineno);
            }
            i = ops.size();
            continue;
        }
        if (!rng.has_value()) {
            std::cerr << "error: value op before any seed (line "
                      << op.lineno << ")\n";
            return 2;
        }
        const std::uint64_t produced = execute(op, *rng);
        record(produced == op.expected, op.lineno);
        ++i;
    }

    std::cout << "ops_checked: " << ops_checked << "\n";
    std::cout << "mismatches: " << mismatches << "\n";
    if (mismatches > 0) {
        std::cout << "first_mismatch_line: " << first_mismatch_line << "\n";
        std::cout << "result: FAIL\n";
        return 1;
    }
    std::cout << "result: PASS\n";
    return 0;
}
