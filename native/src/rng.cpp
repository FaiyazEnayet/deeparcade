// Bit-exact port of CPython's MT19937-based Random core
// (Modules/_randommodule.c): init_genrand / init_by_array seeding,
// the standard twist + tempering generator, genrand_res53 for
// random(), and the getrandbits / _randbelow draw algorithms.
#include "narcade/rng.hpp"

#include <cstddef>
#include <stdexcept>

namespace narcade {

namespace {

constexpr std::uint32_t kN = 624;
constexpr std::uint32_t kM = 397;
constexpr std::uint32_t kMatrixA = 0x9908b0dfU;
constexpr std::uint32_t kUpperMask = 0x80000000U;
constexpr std::uint32_t kLowerMask = 0x7fffffffU;

// CPython init_genrand: linear recurrence state expansion.
void init_genrand(std::uint32_t mt[kN], std::uint32_t s) {
    mt[0] = s;
    for (std::uint32_t i = 1; i < kN; ++i) {
        mt[i] = 1812433253U * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i;
    }
}

// CPython init_by_array (MT19937 reference algorithm): seed the state
// from an array of 32-bit key words. All arithmetic wraps mod 2**32.
void init_by_array(std::uint32_t mt[kN], const std::uint32_t* init_key,
                   std::size_t key_length) {
    init_genrand(mt, 19650218U);
    std::size_t i = 1;
    std::size_t j = 0;
    std::size_t k = (static_cast<std::size_t>(kN) > key_length
                         ? static_cast<std::size_t>(kN)
                         : key_length);
    for (; k != 0; --k) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525U)) +
                init_key[j] + static_cast<std::uint32_t>(j);
        ++i;
        ++j;
        if (i >= kN) {
            mt[0] = mt[kN - 1];
            i = 1;
        }
        if (j >= key_length) {
            j = 0;
        }
    }
    for (k = kN - 1; k != 0; --k) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941U)) -
                static_cast<std::uint32_t>(i);
        ++i;
        if (i >= kN) {
            mt[0] = mt[kN - 1];
            i = 1;
        }
    }
    mt[0] = 0x80000000U;  // MSB is 1; assuring non-zero initial array.
}

}  // namespace

PythonRandom::PythonRandom(std::int64_t seed) {
    // CPython random_seed for an int argument: take abs(seed), split
    // into little-endian 32-bit words; seed 0 produces the key [0].
    std::uint64_t magnitude =
        seed < 0 ? static_cast<std::uint64_t>(-(seed + 1)) + 1
                 : static_cast<std::uint64_t>(seed);
    std::uint32_t key[sizeof(std::uint64_t) / sizeof(std::uint32_t)];
    std::size_t key_length = 0;
    if (magnitude == 0) {
        key[0] = 0;
        key_length = 1;
    } else {
        while (magnitude != 0) {
            key[key_length++] = static_cast<std::uint32_t>(magnitude & 0xffffffffU);
            magnitude >>= 32;
        }
    }
    init_by_array(state_.mt.data(), key, key_length);
    // init_genrand leaves the index at N, so the first draw regenerates
    // the block -- identical to CPython after seeding.
    state_.index = kN;
}

void PythonRandom::regenerate_block() {
    const auto& mt_const = state_.mt;
    auto* mt = state_.mt.data();
    std::uint32_t y;
    std::uint32_t kk;
    for (kk = 0; kk < kN - kM; ++kk) {
        y = (mt_const[kk] & kUpperMask) | (mt_const[kk + 1] & kLowerMask);
        mt[kk] = mt_const[kk + kM] ^ (y >> 1) ^ ((y & 1U) != 0 ? kMatrixA : 0U);
    }
    for (; kk < kN - 1; ++kk) {
        y = (mt_const[kk] & kUpperMask) | (mt_const[kk + 1] & kLowerMask);
        mt[kk] = mt_const[kk - (kN - kM)] ^ (y >> 1) ^
                 ((y & 1U) != 0 ? kMatrixA : 0U);
    }
    y = (mt_const[kN - 1] & kUpperMask) | (mt_const[0] & kLowerMask);
    mt[kN - 1] = mt_const[kM - 1] ^ (y >> 1) ^ ((y & 1U) != 0 ? kMatrixA : 0U);
    state_.index = 0;
}

std::uint32_t PythonRandom::next_uint32() {
    if (state_.index >= kN) {
        regenerate_block();
    }
    std::uint32_t y = state_.mt[state_.index++];
    // Tempering.
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}

double PythonRandom::random() {
    const double a = static_cast<double>(next_uint32() >> 5);
    const double b = static_cast<double>(next_uint32() >> 6);
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

std::uint64_t PythonRandom::getrandbits(int k) {
    if (k < 0) {
        throw std::invalid_argument("getrandbits: k must be non-negative");
    }
    if (k == 0) {
        return 0;
    }
    if (k > 64) {
        throw std::invalid_argument("getrandbits: k > 64 not supported");
    }
    if (k <= 32) {
        return next_uint32() >> (32 - k);
    }
    // Generate ceil(k/32) words in order; the first word is the least
    // significant and the final word is right-shifted by the excess.
    std::uint64_t result = 0;
    const int words = (k + 31) / 32;
    for (int i = 0; i < words; ++i) {
        std::uint32_t word = next_uint32();
        const int remaining = k - 32 * i;
        if (remaining < 32) {
            word >>= (32 - remaining);
        }
        result |= static_cast<std::uint64_t>(word) << (32 * i);
    }
    return result;
}

std::uint64_t PythonRandom::randbelow(std::uint64_t n) {
    if (n == 0) {
        throw std::invalid_argument("randbelow: n must be >= 1");
    }
    // bit_length(n) for n >= 1.
    int k = 0;
    for (std::uint64_t m = n; m != 0; m >>= 1) {
        ++k;
    }
    std::uint64_t r = getrandbits(k);
    while (r >= n) {
        r = getrandbits(k);
    }
    return r;
}

std::int64_t PythonRandom::randint(std::int64_t a, std::int64_t b) {
    if (b < a) {
        throw std::invalid_argument("randint: empty range");
    }
    const std::uint64_t width =
        static_cast<std::uint64_t>(b - a) + 1;
    return a + static_cast<std::int64_t>(randbelow(width));
}

}  // namespace narcade
