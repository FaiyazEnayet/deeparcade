// PythonRandom: a bit-exact reimplementation of CPython's random.Random
// for integer seeds (Modules/_randommodule.c semantics).
//
// This exists so the native DeepArcade engine can consume random numbers
// draw-for-draw identically to the Python reference implementation. It is
// NOT std::mt19937(seed): CPython seeds MT19937 with init_by_array over
// the little-endian 32-bit words of abs(seed), and its random() /
// getrandbits() / _randbelow() draw algorithms differ from libstdc++'s
// distributions. Parity is gated on native/golden/rng_traces.txt.
//
// Supported domain (narrower than CPython's random.Random, by design):
// seeds are int64_t; getrandbits supports 0 <= k <= 64 and throws
// std::invalid_argument outside it (CPython returns arbitrary-precision
// ints); randbelow/randrange require n >= 1; randint(a, b) requires
// b >= a and is exact while (b - a) fits in int64_t — a span covering the
// whole int64 range is outside the supported domain because the width
// computation overflows before the draw. All in-repo callers use small
// ranges (board cells, action indices).
#ifndef NARCADE_RNG_HPP
#define NARCADE_RNG_HPP

#include <array>
#include <cstdint>

namespace narcade {

// Complete MT19937 state: the 624-word state vector plus the index of
// the next word to temper (624 means "regenerate the block on next draw").
struct PythonRandomState {
    std::array<std::uint32_t, 624> mt{};
    std::uint32_t index = 624;
};

class PythonRandom {
public:
    // Seed exactly like CPython random.Random(seed) for an integer seed:
    // key = little-endian 32-bit words of abs(seed); seed 0 uses key [0].
    explicit PythonRandom(std::int64_t seed);

    // CPython genrand_res53: ((a>>5)*67108864.0 + (b>>6)) * 2**-53 from
    // two successive uint32 outputs. Exact in IEEE754 double.
    double random();

    // CPython getrandbits(k). k == 0 returns 0. For k <= 32 a single
    // uint32 right-shifted by (32 - k). For k > 32 words are generated in
    // order and assembled little-endian (first word least significant),
    // with the final word right-shifted by the excess bits. This
    // implementation returns the value in a uint64_t, so k <= 64.
    std::uint64_t getrandbits(int k);

    // CPython _randbelow_with_getrandbits: rejection-sample
    // getrandbits(bit_length(n)) until the result is < n.
    std::uint64_t randbelow(std::uint64_t n);

    // CPython randrange(n) for n >= 1 (single-argument form).
    std::uint64_t randrange(std::uint64_t n) { return randbelow(n); }

    // CPython randint(a, b) == a + _randbelow(b - a + 1). Supported domain: see the header note above.
    std::int64_t randint(std::int64_t a, std::int64_t b);

    // Index chosen by CPython choice(seq): _randbelow(len(seq)).
    std::uint64_t choice_index(std::uint64_t len) { return randbelow(len); }

    // Snapshot / restore the complete generator state. A restored
    // instance reproduces the subsequent draws exactly.
    PythonRandomState save() const { return state_; }
    void restore(const PythonRandomState& state) { state_ = state; }

private:
    std::uint32_t next_uint32();
    void regenerate_block();

    PythonRandomState state_;
};

}  // namespace narcade

#endif  // NARCADE_RNG_HPP
