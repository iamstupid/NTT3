#pragma once
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

// ── Force-inline / hot function macros ──
#if defined(_MSC_VER)
  #define NTT_FORCEINLINE __forceinline
  #define NTT_HOT
  #define NTT_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
  #define NTT_FORCEINLINE [[gnu::always_inline]] inline
  #define NTT_HOT [[gnu::hot]]
  #define NTT_RESTRICT __restrict__
#else
  #define NTT_FORCEINLINE inline
  #define NTT_HOT
  #define NTT_RESTRICT
#endif

namespace ntt {

using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using i64 = int64_t;
using idt = std::size_t;

// ── Bit operations ──
// Constexpr-capable versions (for compile-time use in RootPlan etc.)
constexpr int ctz_constexpr(unsigned x) {
    int r = 0;
    while ((x & 1u) == 0) { x >>= 1; ++r; }
    return r;
}
constexpr int ctzll_constexpr(unsigned long long x) {
    int r = 0;
    while ((x & 1ull) == 0) { x >>= 1; ++r; }
    return r;
}
constexpr int clz_constexpr(unsigned x) {
    int r = 0;
    for (int i = 31; i >= 0; --i) {
        if (x & (1u << i)) return 31 - i;
    }
    return 32;
}
constexpr int lg_constexpr(unsigned x) {
    int r = 0;
    while (x > 1u) { x >>= 1; ++r; }
    return r;
}

// Runtime fast versions
#if defined(_MSC_VER)
  #include <intrin.h>
  inline int ntt_ctz(unsigned x)             { unsigned long r; _BitScanForward(&r, x); return (int)r; }
  inline int ntt_ctzll(unsigned long long x)  { unsigned long r; _BitScanForward64(&r, x); return (int)r; }
  inline int ntt_clz(unsigned x)             { unsigned long r; _BitScanReverse(&r, x); return 31 - (int)r; }
  inline int ntt_lg(unsigned x)              { unsigned long r; _BitScanReverse(&r, x); return (int)r; }
#else
  inline int ntt_ctz(unsigned x)             { return __builtin_ctz(x); }
  inline int ntt_ctzll(unsigned long long x)  { return __builtin_ctzll(x); }
  inline int ntt_clz(unsigned x)             { return __builtin_clz(x); }
  inline int ntt_lg(unsigned x)              { return 31 - __builtin_clz(x); }
#endif

// 64-bit log2 (floor), undefined for x == 0
#if defined(_MSC_VER)
  inline int ntt_lgll(unsigned long long x)  { unsigned long r; _BitScanReverse64(&r, x); return (int)r; }
#else
  inline int ntt_lgll(unsigned long long x)  { return 63 - __builtin_clzll(x); }
#endif

// Number of bits needed to represent x (x > 0)
inline int nbits_nz(idt x) {
    assert(x != 0);
    return ntt_lgll(static_cast<unsigned long long>(x)) + 1;
}

// 2^k as size_t
inline idt pow2(int k) { return idt{1} << k; }

// Ceiling division
inline idt cdiv(idt a, idt b) { return (a + b - 1) / b; }

// log2 for exact powers of 2
inline int log2_exact(idt n) {
    return ntt_ctzll(static_cast<unsigned long long>(n));
}

// ── Integer modular arithmetic (setup-only, not hot-path) ──
inline u64 mul_mod_u64(u64 a, u64 b, u64 m) {
#if defined(_MSC_VER) && !defined(__clang__)
    u64 hi = 0; u64 lo = _umul128(a, b, &hi);
    u64 rem = 0; (void)_udiv128(hi, lo, m, &rem); return rem;
#else
    return static_cast<u64>((static_cast<unsigned __int128>(a) * b) % m);
#endif
}

inline u64 mod_u128(u64 hi, u64 lo, u64 m) {
#if defined(_MSC_VER) && !defined(__clang__)
    u64 rem = 0; (void)_udiv128(hi, lo, m, &rem); return rem;
#else
    return static_cast<u64>(((static_cast<unsigned __int128>(hi) << 64) | lo) % m);
#endif
}

inline u64 pow_mod(u64 a, u64 e, u64 m) {
    u64 r = 1 % m; a %= m;
    while (e > 0) {
        if (e & 1) r = mul_mod_u64(r, a, m);
        a = mul_mod_u64(a, a, m); e >>= 1;
    }
    return r;
}

inline u64 inv_mod(u64 a, u64 p) { return pow_mod(a, p - 2, p); }
inline u64 sub_mod_u64(u64 a, u64 b, u64 m) { return (a >= b) ? (a - b) : (a + m - b); }
inline u64 add_mod_u64(u64 a, u64 b, u64 m) { u64 s = a + b; return (s >= m) ? (s - m) : s; }

inline u64 primitive_root(u64 p) {
    u64 n = p - 1;
    u64 fs[64]; int nf = 0;
    if ((n & 1) == 0) { fs[nf++] = 2; while ((n & 1) == 0) n >>= 1; }
    for (u64 d = 3; d * d <= n; d += 2)
        if (n % d == 0) { fs[nf++] = d; while (n % d == 0) n /= d; }
    if (n > 1) fs[nf++] = n;
    for (u64 g = 2; g < p; ++g) {
        bool ok = true;
        for (int i = 0; i < nf; ++i)
            if (pow_mod(g, (p - 1) / fs[i], p) == 1) { ok = false; break; }
        if (ok) return g;
    }
    return 0;
}

inline u64 n_revbin(u64 x, int bits) {
    u64 r = 0;
    for (int i = 0; i < bits; ++i) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

// ── Aligned allocation ──
#if defined(_MSC_VER)
  #include <malloc.h>
  template<class T, idt Align = 32>
  inline T* aligned_alloc_array(idt n) {
      return static_cast<T*>(_aligned_malloc(n * sizeof(T), Align));
  }
  template<class T, idt Align = 32>
  inline void aligned_free_array(T* p) {
      _aligned_free(p);
  }
#else
  #include <cstdlib>
  template<class T, idt Align = 32>
  inline T* aligned_alloc_array(idt n) {
      void* ptr = nullptr;
      posix_memalign(&ptr, Align, n * sizeof(T));
      return static_cast<T*>(ptr);
  }
  template<class T, idt Align = 32>
  inline void aligned_free_array(T* p) {
      std::free(p);
  }
#endif

// ── Ceil to power of 2 ──
inline idt ceil_pow2(idt x) {
    if (x < 2) return 1;
    return idt(1) << (ntt_lg((unsigned)(x - 1)) + 1);
}

// ── Ceil to smooth number {4,5,6}*2^n ──
// Max NTT size: 3*2^23 limbs (CRT overflow beyond this).
// Entries where mixed-radix sub_n < 4 vecs are omitted (twisted_conv needs batch-of-4).
// Minimum valid mixed-radix: 96 (m=3, sub_n=4 vecs), 160 (m=5, sub_n=4 vecs).
static constexpr idt SMOOTH_TABLE[] = {
    4,
    8,
    16,
    32,
    64, 96,
    128, 160, 192,
    256, 320, 384,
    512, 640, 768,
    1024, 1280, 1536,
    2048, 2560, 3072,
    4096, 5120, 6144,
    8192, 10240, 12288,
    16384, 20480, 24576,
    32768, 40960, 49152,
    65536, 81920, 98304,
    131072, 163840, 196608,
    262144, 327680, 393216,
    524288, 655360, 786432,
    1048576, 1310720, 1572864,
    2097152, 2621440, 3145728,
    4194304, 5242880, 6291456,
    8388608, 10485760, 12582912,
    16777216, 20971520, 25165824,
};
static constexpr int SMOOTH_TABLE_SIZE = sizeof(SMOOTH_TABLE) / sizeof(SMOOTH_TABLE[0]);

inline idt ceil_smooth(idt x) {
    // Binary search for smallest entry >= x
    int lo = 0, hi = SMOOTH_TABLE_SIZE;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (SMOOTH_TABLE[mid] < x) lo = mid + 1;
        else hi = mid;
    }
    return SMOOTH_TABLE[lo];  // lo == SMOOTH_TABLE_SIZE only if x > max (shouldn't happen)
}

// ── Constants ──
static constexpr int MAX_LOG = 26;
static constexpr int LOG_BLOCK = 6;  // cache-oblivious block = 2^6 = 64 Vecs
static constexpr idt BLOCK_SIZE = idt(1) << LOG_BLOCK;

} // namespace ntt
