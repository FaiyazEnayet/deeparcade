// dqn_diff_runner: batch differential-test endpoint for the C++ DQN
// inference engine (CPP-06d). Driven by tools/native_dqn_check.py,
// which computes the PyTorch reference over the same states; every
// compared value must agree within the driver's reported tolerances
// (encodings bit-identically).
//
// Protocol (line-based text, stdin -> stdout). Floats travel as 8
// lowercase hex digits holding the IEEE754 binary32 bit pattern.
//
//   LOAD <path>
//     Loads runs/dqn_weights.json (shape-verified). Replies "OK" or
//     "ERROR <message>".
//   STATE <mode> <food_r> <food_c> <heading> <n_body>
//     then n_body lines "<r> <c>" (tail first, head last).
//     Replies:
//       "ENC <436 float hex>"     the encode_dqn encoding
//       "OUT <action> <l0> <l1> <l2>"   argmax + logits (float hex)
//       "H1 <128 float hex>"      only when mode == 1
//       "H2 <64 float hex>"       only when mode == 1
//     or "ERROR <message>".
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "narcade/dqn_infer.hpp"

namespace {

std::string float_hex(float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    char buf[9];
    std::snprintf(buf, sizeof(buf), "%08x", bits);
    return buf;
}

template <std::size_t N>
void print_floats(const char* tag, const std::array<float, N>& values) {
    std::cout << tag;
    for (float v : values) {
        std::cout << ' ' << float_hex(v);
    }
    std::cout << '\n';
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::unique_ptr<narcade::DqnNet> net;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream header(line);
        std::string tag;
        header >> tag;
        if (tag == "LOAD") {
            std::string path;
            header >> path;
            try {
                net = std::make_unique<narcade::DqnNet>(
                    narcade::load_dqn_weights(path));
                std::cout << "OK\n";
            } catch (const std::exception& e) {
                std::cout << "ERROR " << e.what() << "\n";
            }
            std::cout.flush();
        } else if (tag == "STATE") {
            int mode = 0;
            int fr = 0, fc = 0, heading = 0, n_body = 0;
            header >> mode >> fr >> fc >> heading >> n_body;
            std::vector<narcade::Cell> body;
            bool bad = false;
            for (int i = 0; i < n_body; ++i) {
                if (!std::getline(std::cin, line)) {
                    return 1;
                }
                std::istringstream cell_line(line);
                int r = 0, c = 0;
                if (!(cell_line >> r >> c)) {
                    bad = true;
                }
                body.push_back(narcade::Cell{r, c});
            }
            if (bad || !net) {
                std::cout << "ERROR "
                          << (bad ? "malformed body line"
                                  : "weights not loaded")
                          << "\n";
                std::cout.flush();
                continue;
            }
            try {
                const auto enc = narcade::encode_dqn(
                    body, narcade::Cell{fr, fc}, heading);
                const narcade::DqnActivations out = net->forward(enc);
                print_floats("ENC", enc);
                int best = 0;
                for (int a = 1; a < 3; ++a) {
                    if (out.logits[static_cast<std::size_t>(a)] >
                        out.logits[static_cast<std::size_t>(best)]) {
                        best = a;
                    }
                }
                std::cout << "OUT " << best;
                for (float v : out.logits) {
                    std::cout << ' ' << float_hex(v);
                }
                std::cout << '\n';
                if (mode == 1) {
                    print_floats("H1", out.h1);
                    print_floats("H2", out.h2);
                }
            } catch (const std::exception& e) {
                std::cout << "ERROR " << e.what() << "\n";
            }
            std::cout.flush();
        }
        // Unknown lines are ignored, mirroring the other runners.
    }
    return 0;
}
