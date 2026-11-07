// CnnNet: native inference for the DeepArcade DA8 convolutional DQN.
// Inference ONLY: training stays in PyTorch (scripts/train_cnn_dqn.py,
// untouched), and this code consumes the committed exported artifacts
// runs/cnn_dqn_{plain,residual}_seed{7,8,9}_weights.json.
//
// The network is src/narcade/models.py's CNNDQN: a configurable 3x3
// convolution encoder over three spatial channels (head, body, food)
// on the 12x12 board, an optional residual block at the last channel
// width, adaptive average pooling, the heading one-hot concatenated
// as conditioning, and a two-layer head to three Q-values. The two
// committed variants are plain (channels [16, 32], 13,795 parameters)
// and residual (same channels plus the residual block, 32,291
// parameters); the loader accepts any config the Python exporter can
// record, with layer names and shapes derived from that config.
//
// Loader: a strict port of narcade.models.load_cnn_checkpoint. The
// document is validated before any tensor is built (schema_version 1,
// kind, encoder name/revision, float regime, config validity, exact
// layer names and shapes for the recorded config, every value a
// finite number, weights SHA-256 over the canonical layers payload,
// metadata key set) and any defect raises CnnCheckpointError, the
// counterpart of Python's CNNCheckpointError. One deliberate
// strengthening: nested layer values must be rectangular, where the
// Python loader's shape probe follows only the first branch and a
// ragged document is rejected later at tensor construction instead;
// both reject, this one at validation time.
//
// Float regime: the checkpoint format declares its parity regime as
// float64 (parity_float_regime in every export, matching the NumPy
// mirror narcade.models.cnn_logits_numpy): accumulate in double over
// the stored float32 values, rounding hidden activations to float32
// after each ReLU. The forward pass below mirrors cnn_logits_numpy's
// operation order, so native logits track the PyTorch policy to the
// same standard the MLP port is held to; tools/native_cnn_check.py
// measures and reports the actual differences against both the
// PyTorch policy and the NumPy mirror.
//
// encode_cnn is a port of narcade.models.encode_cnn_dqn. The encoding
// is exact (every element 0.0f or 1.0f), so encoding parity is
// bit-identical or wrong.
#ifndef NARCADE_CNN_INFER_HPP
#define NARCADE_CNN_INFER_HPP

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "narcade/snake.hpp"  // Cell, SnakeEnv

namespace narcade {

// Counterpart of narcade.models.CNNCheckpointError: the checkpoint
// document is malformed, unsupported, or corrupt. A std::runtime_error
// subclass, so existing catch (const std::exception&) call sites keep
// working while checkpoint failures stay distinguishable.
class CnnCheckpointError : public std::runtime_error {
public:
    explicit CnnCheckpointError(const std::string& msg)
        : std::runtime_error(msg) {}
};

inline constexpr int kCnnBoard = 12;
inline constexpr int kCnnChannels = 3;
inline constexpr int kCnnSpatialDim = kCnnChannels * kCnnBoard * kCnnBoard;
inline constexpr int kCnnHeadingDim = 4;
inline constexpr int kCnnActions = 3;

// Mirror of narcade.models.CNNConfig (defaults are the Python
// defaults CNNConfig.from_dict applies for absent keys).
struct CnnConfig {
    std::vector<int> channels = {16, 32};
    int kernel = 3;
    bool residual = false;
    int pool_size = 2;
    int hidden = 64;
};

// One checkpoint tensor: declared shape and row-major flat values.
// Values are held as double; every stored value is an exact float32
// (the export renders float32 weights as decimal text, and the parse
// recovers the exact value), so no precision is lost by the widening.
struct CnnTensor {
    std::vector<std::int64_t> shape;
    std::vector<double> values;

    std::int64_t numel() const {
        std::int64_t n = 1;
        for (std::int64_t d : shape) {
            n *= d;
        }
        return n;
    }
};

struct CnnWeights {
    CnnConfig config;
    std::string weights_sha256;
    // Encoder convolutions in layer order: convs.<i>.weight / .bias.
    std::vector<CnnTensor> conv_weights;
    std::vector<CnnTensor> conv_biases;
    // Residual block (present exactly when config.residual).
    CnnTensor res1_weight, res1_bias, res2_weight, res2_bias;
    CnnTensor head1_weight, head1_bias, head2_weight, head2_bias;

    int num_params() const;
};

// Strictly load and validate a DA8 CNN checkpoint export (see the
// header comment for the validation list). Throws CnnCheckpointError
// on any defect, before any tensor is built.
CnnWeights load_cnn_weights(const std::string& path);

// Port of narcade.models.encode_cnn_dqn. `body` is tail-first,
// head-last, exactly like env.body in Python; `food` is the food
// cell, or std::nullopt on a terminal win (the food plane then stays
// empty, exactly as in Python). Layout: spatial planes in channel
// order head, body, food, each plane cell index r*12 + c; heading
// one-hot of length 4.
struct CnnEncoding {
    std::array<float, kCnnSpatialDim> spatial;
    std::array<float, kCnnHeadingDim> heading;
};

CnnEncoding encode_cnn(const std::vector<Cell>& body,
                       std::optional<Cell> food, int heading);

// Same, from a live env.
CnnEncoding encode_cnn(const SnakeEnv& env);

struct CnnActivations {
    // Pooled encoder features with the heading one-hot concatenated
    // (the head1 input), length channels.back()*pool_size*pool_size+4.
    std::vector<double> features;
    std::array<double, kCnnActions> logits;  // head2 output (Q values)
};

class CnnNet {
public:
    explicit CnnNet(CnnWeights weights) : weights_(std::move(weights)) {}

    // Forward pass in the declared float64 parity regime (see the
    // header comment): double accumulation over the stored float32
    // values, hidden activations rounded to float32 after each ReLU,
    // mirroring narcade.models.cnn_logits_numpy's operation order.
    CnnActivations forward(const CnnEncoding& input) const;

    // Argmax over the logits with first-max tie behavior (the rule of
    // torch.argmax / numpy.argmax used by the Python policy).
    int act(const CnnEncoding& input) const;

    const CnnWeights& weights() const { return weights_; }

private:
    CnnWeights weights_;
};

}  // namespace narcade

#endif  // NARCADE_CNN_INFER_HPP
