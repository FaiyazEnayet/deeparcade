// DqnNet: native float32 inference for the DeepArcade "Pro" DQN
// (CPP-06d). Inference ONLY: training stays in PyTorch
// (scripts/train_dqn.py, untouched), and this code consumes the real
// exported artifact runs/dqn_weights.json.
//
// The network is scripts/train_dqn.py's DQN: Linear(436,128) + ReLU,
// Linear(128,64) + ReLU, Linear(64,3). The exported JSON stores each
// layer's "w" in PyTorch Linear layout [out][in] (verified on load:
// 128x436, 64x128, 3x64, with matching biases).
//
// encode_dqn is a line-for-line port of scripts/train_dqn.py's
// encode_dqn (which must in turn match the browser's encodeDQN).
// Parity is gated on tools/native_dqn_check.py: bit-identical
// encodings on >= 10,000 harvested states, action agreement, and
// reported logit/activation differences against the PyTorch reference.
#ifndef NARCADE_DQN_INFER_HPP
#define NARCADE_DQN_INFER_HPP

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "narcade/snake.hpp"  // Cell, SnakeEnv

namespace narcade {

inline constexpr int kDqnInputDim = 436;  // 12*12*3 + 4 (INPUT_DIM)
inline constexpr int kDqnHidden1 = 128;
inline constexpr int kDqnHidden2 = 64;
inline constexpr int kDqnActions = 3;

// Port of encode_dqn(env) (scripts/train_dqn.py line 30). `body` is
// tail-first, head-last, exactly like env.body in Python; `food` is
// the food cell (Python would fail unpacking None on a full board, so
// a live food cell is a precondition); `heading` in 0..3. Layout:
// [0,144) head one-hot at r*12+c, [144,288) body (all but head),
// [288,432) food, [432,436) heading one-hot. The encoding is exact:
// every element is 0.0f or 1.0f, so parity is bit-identical or wrong.
std::array<float, kDqnInputDim> encode_dqn(const std::vector<Cell>& body,
                                           Cell food, int heading);

// Same, from a live env. Throws std::runtime_error if the env has no
// food (the terminal-win state), mirroring Python's unpack failure.
std::array<float, kDqnInputDim> encode_dqn(const SnakeEnv& env);

struct DqnLayer {
    // Row-major [out][in], exactly as stored in the export.
    std::vector<std::vector<float>> w;
    std::vector<float> b;
};

struct DqnWeights {
    int input_dim = 0;
    std::array<DqnLayer, 3> layers;
};

// Parse runs/dqn_weights.json (minimal JSON reader, no external
// dependencies) and verify the declared shapes: input_dim == 436;
// layer weights 128x436, 64x128, 3x64; biases 128, 64, 3. Throws
// std::runtime_error on any structural or shape mismatch.
DqnWeights load_dqn_weights(const std::string& path);

struct DqnActivations {
    std::array<float, kDqnHidden1> h1;      // post-ReLU layer 1
    std::array<float, kDqnHidden2> h2;      // post-ReLU layer 2
    std::array<float, kDqnActions> logits;  // layer 3 output (Q values)
};

class DqnNet {
public:
    explicit DqnNet(DqnWeights weights) : weights_(std::move(weights)) {}

    // Float32 forward pass in the network's layer order:
    // h1 = relu(W0 x + b0), h2 = relu(W1 h1 + b1), logits = W2 h2 + b2.
    // Accumulation is sequential in float, mirroring the mathematical
    // definition; PyTorch's backend may accumulate in a different
    // order, which is exactly the small float32 difference the check
    // driver measures and reports.
    DqnActivations forward(
        const std::array<float, kDqnInputDim>& input) const;

    // Argmax over the logits with first-max tie behavior (the rule of
    // torch.argmax / numpy.argmax used by the Python policy).
    int act(const std::array<float, kDqnInputDim>& input) const;

    const DqnWeights& weights() const { return weights_; }

private:
    DqnWeights weights_;
};

}  // namespace narcade

#endif  // NARCADE_DQN_INFER_HPP
