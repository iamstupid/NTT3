#pragma once
#include "../common.hpp"

namespace ntt {

// ── u128 type for CRT ──
struct u128 {
    u64 lo, hi;

    friend NTT_FORCEINLINE u128 operator*(u32 a, u128 b) {
#if defined(_MSC_VER)
        u64 hi1;
        u64 lo1 = _umul128((u64)a, b.lo, &hi1);
        return {lo1, hi1 + (u64)a * b.hi};
#else
        unsigned __int128 full = (unsigned __int128)a * b.lo;
        return {(u64)full, (u64)(full >> 64) + (u64)a * b.hi};
#endif
    }

    friend NTT_FORCEINLINE u128 operator+(u128 a, u128 b) {
        u64 lo = a.lo + b.lo;
        u64 hi = a.hi + b.hi + (lo < a.lo);
        return {lo, hi};
    }

    friend NTT_FORCEINLINE u128 operator-(u128 a, u128 b) {
        u64 lo = a.lo - b.lo;
        u64 hi = a.hi - b.hi - (a.lo < b.lo);
        return {lo, hi};
    }

    friend NTT_FORCEINLINE bool operator>=(u128 a, u128 b) {
        return a.hi != b.hi ? a.hi > b.hi : a.lo >= b.lo;
    }
};

// ── Three primes ──
static constexpr u32 CRT_P0 = 880803841U;   // 105 * 2^23 + 1
static constexpr u32 CRT_P1 = 754974721U;   //  90 * 2^23 + 1
static constexpr u32 CRT_P2 = 377487361U;   //  45 * 2^23 + 1

// ── CRT lifting coefficients ──
// pi_i = (product / p_i) * inverse(product / p_i, p_i)
// These satisfy: pi_i ≡ 1 (mod p_i), pi_i ≡ 0 (mod p_j) for j≠i
static constexpr u128 CRT_PI0     = {0x3DCC500394E0000DULL, 0x00000000009BBB30ULL};
static constexpr u128 CRT_PI1     = {0xDA6D3FFCF3FFFFF5ULL, 0x0000000000CFA43FULL};
static constexpr u128 CRT_PI2     = {0x0B5EF00067200001ULL, 0x000000000033E910ULL};
static constexpr u128 CRT_PRODUCT = {0x11CC400078000001ULL, 0x0000000000CFA440ULL};

// Reciprocal of product.hi, slightly biased down for floor division
static constexpr double CRT_INV_HI = 0x1.3b9ee9fe0e109p-24;

// ── Per-coefficient CRT recovery ──
NTT_FORCEINLINE u128 crt_recover(u32 n0, u32 n1, u32 n2) {
    u128 sum = n0 * CRT_PI0 + n1 * CRT_PI1 + n2 * CRT_PI2;
    u32 q = (u32)((double)sum.hi * CRT_INV_HI);
    u128 r = sum - q * CRT_PRODUCT;
    if (r >= CRT_PRODUCT) r = r - CRT_PRODUCT;  // at most once
    return r;  // x in [0, product), <= 88 bit
}

// ── CRT + carry propagation for big integer multiplication ──
inline void crt_and_propagate(
    u32* out, idt len,
    const u32* r0, const u32* r1, const u32* r2)
{
    u64 carry = 0;
    for (idt i = 0; i < len; ++i) {
        u128 val = crt_recover(r0[i], r1[i], r2[i]);
        // Add carry
        u64 new_lo = val.lo + carry;
        u64 new_hi = val.hi + (new_lo < val.lo);
        out[i] = (u32)new_lo;
        carry = (new_lo >> 32) | (new_hi << 32);
    }
    // Handle remaining carry
    // For proper big-integer multiply the caller should ensure enough space
}

} // namespace ntt
