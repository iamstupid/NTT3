#pragma once
// pointmul.hpp - Frequency-domain pointwise multiply and scale
//
// Part of ntt::p50x4 - 4-prime ~50-bit NTT (double FMA Barrett)

#include "fft_ctx.hpp"

namespace ntt { namespace p50x4 {

// Point multiply in frequency domain
inline void point_mul(const FftCtx& Q, double* a, const double* b,
                          std::size_t len) {
    V4 n = v4_set1(Q.p), ninv = v4_set1(Q.pinv);
    std::size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        V4 va = v4_reduce_pm1n(v4_load(a + i), n, ninv);
        V4 vb = v4_reduce_pm1n(v4_load(b + i), n, ninv);
        v4_store(a + i, v4_mulmod(va, vb, n, ninv));
    }
    for (; i < len; ++i) {
        a[i] = s_mulmod(s_reduce_pm1n(a[i], Q.p, Q.pinv),
                         s_reduce_pm1n(b[i], Q.p, Q.pinv), Q.p, Q.pinv);
    }
}

// Scale by 1/N (for inverse FFT normalization)
inline void scale(const FftCtx& Q, double* d, std::size_t len, double inv_n) {
    V4 n = v4_set1(Q.p), ninv = v4_set1(Q.pinv);
    V4 vinv = v4_set1(inv_n);
    std::size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        V4 v = v4_reduce_pm1n(v4_load(d + i), n, ninv);
        v4_store(d + i, v4_mulmod(v, vinv, n, ninv));
    }
    for (; i < len; ++i) {
        d[i] = s_mulmod(s_reduce_pm1n(d[i], Q.p, Q.pinv), inv_n, Q.p, Q.pinv);
    }
}

}} // namespace ntt::p50x4
