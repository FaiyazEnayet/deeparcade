// cnn_diff_runner: batch differential-test endpoint for the C++ CNN
// DQN inference engine. Driven by tools/native_cnn_check.py, which
// computes the PyTorch reference (and the NumPy mirror) over the same
// states; encodings must agree bit-identically and logits within the
// driver's reported band.
//
// Protocol (line-based text, stdin -> stdout). Floats travel as 8
// lowercase hex digits holding the IEEE754 binary32 bit pattern; the
// native forward pass accumulates in float64 (the checkpoint's
// declared parity regime), so logits and features are cast to
// binary32 for the wire.
//
//   LOAD <path>
//     Loads a DA8 CNN checkpoint export (strict validation). Replies
//     "OK" or "ERROR <message>".
//   STATE <mode> <food_r> <food_c> <heading> <n_body>
//     food_r == -1 encodes "no food" (terminal win). Then n_body
//     lines "<r> <c>" (tail first, head last).
//     Replies:
//       "ENC <436 float hex>"     encode_cnn output: 432 spatial
//                                 (head, body, food planes) + 4
//                                 heading one-hot
//       "OUT <action> <l0> <l1> <l2>"   argmax + logits (float hex)
//       "FEAT <n float hex>"      only when mode == 1: pooled
//                                 features with heading concatenated
//                                 (the head1 input)
//     or "ERROR <message>".
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "narcade/cnn_infer.hpp"

namespace {

std::string float_hex(float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    char buf[9];
    std::snprintf(buf, sizeof(buf), "%08x", bits);
    return buf;
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::unique_ptr<narcade::CnnNet> net;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream header(line);
        std::string tag;
        header >> tag;
        if (tag == "LOAD") {
            std::string path;
            header >> path;
            try {
                net = std::make_unique<narcade::CnnNet>(
                    narcade::load_cnn_weights(path));
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
                std::optional<narcade::Cell> food;
                if (fr >= 0) {
                    food = narcade::Cell{fr, fc};
                }
                const narcade::CnnEncoding enc =
                    narcade::encode_cnn(body, food, heading);
                const narcade::CnnActivations out = net->forward(enc);
                std::cout << "ENC";
                for (float v : enc.spatial) {
                    std::cout << ' ' << float_hex(v);
                }
                for (float v : enc.heading) {
                    std::cout << ' ' << float_hex(v);
                }
                std::cout << '\n';
                int best = 0;
                for (int a = 1; a < 3; ++a) {
                    if (out.logits[static_cast<std::size_t>(a)] >
                        out.logits[static_cast<std::size_t>(best)]) {
                        best = a;
                    }
                }
                std::cout << "OUT " << best;
                for (double v : out.logits) {
                    std::cout << ' ' << float_hex(static_cast<float>(v));
                }
                std::cout << '\n';
                if (mode == 1) {
                    std::cout << "FEAT " << out.features.size();
                    for (double v : out.features) {
                        std::cout << ' '
                                  << float_hex(static_cast<float>(v));
                    }
                    std::cout << '\n';
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
