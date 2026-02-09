#pragma once
// v4.hpp - AVX2 double-precision (4x f64) SIMD primitives
//
// Pure intrinsic wrappers for __m256d operations. Used by ntt::p50x4 engine.

#include "../common.hpp"
#include <immintrin.h>

namespace ntt {

using V4 = __m256d;

inline V4 v4_set1(double x)         { return _mm256_set1_pd(x); }
inline V4 v4_load(const double* p)   { return _mm256_load_pd(p); }
inline void v4_store(double* p, V4 v){ _mm256_store_pd(p, v); }
inline void v4_stream(double* p, V4 v){ _mm256_stream_pd(p, v); }
inline V4 v4_add(V4 a, V4 b)        { return _mm256_add_pd(a, b); }
inline V4 v4_sub(V4 a, V4 b)        { return _mm256_sub_pd(a, b); }
inline V4 v4_mul(V4 a, V4 b)        { return _mm256_mul_pd(a, b); }
inline V4 v4_neg(V4 a)              { return _mm256_xor_pd(a, _mm256_set1_pd(-0.0)); }
inline V4 v4_zero()                  { return _mm256_setzero_pd(); }
inline V4 v4_loadu(const double* p)  { return _mm256_loadu_pd(p); }
inline void v4_storeu(double* p, V4 v){ _mm256_storeu_pd(p, v); }

inline V4 v4_round(V4 x) {
    return _mm256_round_pd(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

// mulmod: result ~ a*b mod n, in [-n, n]
__forceinline V4 v4_mulmod(V4 a, V4 b, V4 n, V4 ninv) {
    V4 h = v4_mul(a, b);
    V4 q = v4_round(v4_mul(h, ninv));
    V4 l = _mm256_fmsub_pd(a, b, h);
    return v4_add(_mm256_fnmadd_pd(q, n, h), l);
}

// reduce_to_pm1n: bring to [-n, n]
__forceinline V4 v4_reduce_pm1n(V4 a, V4 n, V4 ninv) {
    return _mm256_fnmadd_pd(v4_round(v4_mul(a, ninv)), n, a);
}

// 4x4 transpose of vec4d registers
inline void v4_transpose(V4& y0, V4& y1, V4& y2, V4& y3,
                          V4  x0, V4  x1, V4  x2, V4  x3) {
    V4 t0 = _mm256_unpacklo_pd(x0, x1);
    V4 t1 = _mm256_unpackhi_pd(x0, x1);
    V4 t2 = _mm256_unpacklo_pd(x2, x3);
    V4 t3 = _mm256_unpackhi_pd(x2, x3);
    y0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    y1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    y2 = _mm256_permute2f128_pd(t0, t2, 0x31);
    y3 = _mm256_permute2f128_pd(t1, t3, 0x31);
}

// FLINT's unpack_lo_permute_0_2_1_3
inline V4 v4_unpack_lo_perm(V4 u, V4 v) {
    V4 a = _mm256_unpacklo_pd(u, v);
    return _mm256_permute4x64_pd(a, 0xD8);
}

inline V4 v4_unpack_hi_perm(V4 u, V4 v) {
    V4 a = _mm256_unpackhi_pd(u, v);
    return _mm256_permute4x64_pd(a, 0xD8);
}

inline V4 v4_set_d4(double a, double b, double c, double d) {
    return _mm256_setr_pd(a, b, c, d);
}

// Reverse 4 lanes: {a,b,c,d} -> {d,c,b,a}
inline V4 v4_reverse(V4 x) { return _mm256_permute4x64_pd(x, 0x1B); }

} // namespace ntt
