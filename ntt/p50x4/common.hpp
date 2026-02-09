#pragma once
// common.hpp - Constants, aligned memory, scalar double ops, twiddle builder
//
// Part of ntt::p50x4 - 4-prime ~50-bit NTT (double FMA Barrett)
//
// Types, bit ops, and integer mod arithmetic are provided by ../common.hpp.
// V4 SIMD primitives are provided by ../simd/v4.hpp.

#include "../common.hpp"
#include "../simd/v4.hpp"

#include <array>
#include <cmath>
#include <new>

namespace ntt { namespace p50x4 {

// ================================================================
// Constants
// ================================================================
static constexpr int LG_BLK_SZ = 8;
static constexpr std::size_t BLK_SZ = 256;
static constexpr int W2TAB_INIT = 12;
static constexpr int W2TAB_SIZE = 40;

static constexpr std::array<u64, 4> PRIMES = {
    519519244124161ULL, 750416685957121ULL,
    865865406873601ULL, 1096762848706561ULL
};

// ================================================================
// Aligned memory (4096-byte aligned, zero-filled doubles for FFT buffers)
// ================================================================
inline double* alloc_doubles(std::size_t count, std::size_t align = 4096) {
    std::size_t bytes = count * sizeof(double);
    bytes = (bytes + align - 1) / align * align;
    if (bytes == 0) bytes = align;
    void* p = ::operator new(bytes, std::align_val_t(align));
    std::memset(p, 0, bytes);
    return static_cast<double*>(p);
}

inline void free_doubles(void* p) {
    if (p) ::operator delete(p, std::align_val_t(4096));
}

// ================================================================
// Scalar double modular ops (for setup)
// ================================================================
inline double s_reduce_0n_to_pmhn(double a, double n) {
    return (a > 0.5 * n) ? (a - n) : a;
}

inline double s_reduce_pm1n_to_pmhn(double a, double n) {
    double h = 0.5 * n, t = a + n;
    if (a > h) return a - n;
    if (t < h) return t;
    return a;
}

inline double s_mulmod(double a, double b, double n, double ninv) {
    double h = a * b;
    double q = std::nearbyint(h * ninv);
    double l = std::fma(a, b, -h);
    return std::fma(-q, n, h) + l;
}

inline double s_reduce_pm1n(double a, double n, double ninv) {
    return std::fma(-std::nearbyint(a * ninv), n, a);
}

// ================================================================
// V4 twiddle builder (depends on scalar s_mulmod above)
// ================================================================
// V4 type and SIMD primitives are provided by ../simd/v4.hpp.

// Build V4 twiddle = {base, base*root, base*root^2, base*root^3}
// Sets step = v4_set1(root^4) for V4-stride advancement.
static inline V4 v4_build_tw(double base, double root,
                               double p, double pinv, V4& step) {
    double t1 = s_mulmod(base, root, p, pinv);
    double t2 = s_mulmod(t1, root, p, pinv);
    double t3 = s_mulmod(t2, root, p, pinv);
    double r2 = s_mulmod(root, root, p, pinv);
    double r4 = s_mulmod(r2, r2, p, pinv);
    step = v4_set1(r4);
    return _mm256_set_pd(t3, t2, t1, base);
}

}} // namespace ntt::p50x4
