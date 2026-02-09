#pragma once
// fft.hpp - Forward and inverse FFT: basecases, block recursion, entry points
//
// Part of ntt::p50x4 - 4-prime ~50-bit NTT (double FMA Barrett)

#include "fft_ctx.hpp"

namespace ntt { namespace p50x4 {

// ================================================================
// Forward FFT basecases (depth 0..8, j_is_zero variants)
// ================================================================

inline void fwd_length4_zero_j(double& x0, double& x1, double& x2, double& x3,
                                double n, double ninv, double e14) {
    double X0 = s_reduce_pm1n(x0, n, ninv);
    double X2 = s_reduce_pm1n(x2, n, ninv);
    double X3 = s_reduce_pm1n(x3, n, ninv);
    double Y0 = X0 + X2, Y1 = x1 + X3;
    double Y2 = X0 - X2, Y3 = x1 - X3;
    Y1 = s_reduce_pm1n(Y1, n, ninv);
    Y3 = s_mulmod(Y3, e14, n, ninv);
    x0 = Y0 + Y1; x1 = Y0 - Y1;
    x2 = Y2 + Y3; x3 = Y2 - Y3;
}

inline void fwd_length4_any_j(double& x0, double& x1, double& x2, double& x3,
                               double n, double ninv, double w2, double w, double iw) {
    double X0 = s_reduce_pm1n(x0, n, ninv);
    X0 = x0;
    X0 = s_reduce_pm1n(x0, n, ninv);
    double X2 = s_mulmod(x2, w2, n, ninv);
    double X3 = s_mulmod(x3, w2, n, ninv);
    double Y0 = X0 + X2, Y1 = x1 + X3;
    double Y2 = X0 - X2, Y3 = x1 - X3;
    Y1 = s_mulmod(Y1, w, n, ninv);
    Y3 = s_mulmod(Y3, iw, n, ninv);
    x0 = Y0 + Y1; x1 = Y0 - Y1;
    x2 = Y2 + Y3; x3 = Y2 - Y3;
}

// V4 versions for block processing
__forceinline void fwd_radix4_v4_j0(V4 n, V4 ninv, V4 iw,
                              double* X0, double* X1, double* X2, double* X3) {
    for (std::size_t i = 0; i < BLK_SZ; i += 4) {
        V4 x0 = v4_load(X0 + i); x0 = v4_reduce_pm1n(x0, n, ninv);
        V4 x1 = v4_load(X1 + i);
        V4 x2 = v4_load(X2 + i); x2 = v4_reduce_pm1n(x2, n, ninv);
        V4 x3 = v4_load(X3 + i); x3 = v4_reduce_pm1n(x3, n, ninv);
        V4 y0 = v4_add(x0, x2), y1 = v4_add(x1, x3);
        V4 y2 = v4_sub(x0, x2), y3 = v4_sub(x1, x3);
        y1 = v4_reduce_pm1n(y1, n, ninv);
        y3 = v4_mulmod(y3, iw, n, ninv);
        v4_store(X0 + i, v4_add(y0, y1));
        v4_store(X1 + i, v4_sub(y0, y1));
        v4_store(X2 + i, v4_add(y2, y3));
        v4_store(X3 + i, v4_sub(y2, y3));
    }
}

__forceinline void fwd_radix4_v4_jnz(V4 n, V4 ninv, V4 w, V4 w2, V4 iw,
                                double* X0, double* X1, double* X2, double* X3) {
    for (std::size_t i = 0; i < BLK_SZ; i += 4) {
        V4 x0 = v4_load(X0 + i); x0 = v4_reduce_pm1n(x0, n, ninv);
        V4 x1 = v4_load(X1 + i);
        V4 x2 = v4_load(X2 + i); x2 = v4_mulmod(x2, w2, n, ninv);
        V4 x3 = v4_load(X3 + i); x3 = v4_mulmod(x3, w2, n, ninv);
        V4 y0 = v4_add(x0, x2), y1 = v4_add(x1, x3);
        V4 y2 = v4_sub(x0, x2), y3 = v4_sub(x1, x3);
        y1 = v4_mulmod(y1, w, n, ninv);
        y3 = v4_mulmod(y3, iw, n, ninv);
        v4_store(X0 + i, v4_add(y0, y1));
        v4_store(X1 + i, v4_sub(y0, y1));
        v4_store(X2 + i, v4_add(y2, y3));
        v4_store(X3 + i, v4_sub(y2, y3));
    }
}

__forceinline void fwd_radix2_v4_j0(V4 n, V4 ninv, double* X0, double* X1) {
    for (std::size_t i = 0; i < BLK_SZ; i += 4) {
        V4 x0 = v4_load(X0 + i); x0 = v4_reduce_pm1n(x0, n, ninv);
        V4 x1 = v4_load(X1 + i); x1 = v4_reduce_pm1n(x1, n, ninv);
        v4_store(X0 + i, v4_add(x0, x1));
        v4_store(X1 + i, v4_sub(x0, x1));
    }
}

__forceinline void fwd_radix2_v4_jnz(V4 n, V4 ninv, V4 w, double* X0, double* X1) {
    for (std::size_t i = 0; i < BLK_SZ; i += 4) {
        V4 x0 = v4_load(X0 + i); x0 = v4_reduce_pm1n(x0, n, ninv);
        V4 x1 = v4_load(X1 + i); x1 = v4_mulmod(x1, w, n, ninv);
        v4_store(X0 + i, v4_add(x0, x1));
        v4_store(X1 + i, v4_sub(x0, x1));
    }
}

// ----------------------------------------------------------------
// Basecase depth 4 (16 points, vec4d + transpose) - template on J0
// ----------------------------------------------------------------
template <bool J0>
inline void fft_basecase_4(const FftCtx& Q, double* X, std::size_t j) {
    V4 n = v4_set1(Q.p), ninv = v4_set1(Q.pinv);
    V4 x0 = v4_load(X + 0), x1 = v4_load(X + 4);
    V4 x2 = v4_load(X + 8), x3 = v4_load(X + 12);
    int j_bits = 0; std::size_t j_r = 0;
    if constexpr (!J0) { j_bits = nbits_nz(j); j_r = j - pow2(j_bits - 1); }

    // Column FFTs
    {
        V4 X0 = v4_reduce_pm1n(x0, n, ninv);
        V4 X2, X3;
        if constexpr (J0) {
            X2 = v4_reduce_pm1n(x2, n, ninv);
            X3 = v4_reduce_pm1n(x3, n, ninv);
        } else {
            V4 cw2 = v4_set1(Q.w2tab[j_bits][j_r]);
            X2 = v4_mulmod(x2, cw2, n, ninv);
            X3 = v4_mulmod(x3, cw2, n, ninv);
        }
        V4 Y0 = v4_add(X0, X2), Y1 = v4_add(x1, X3);
        V4 Y2 = v4_sub(X0, X2), Y3 = v4_sub(x1, X3);
        if constexpr (J0) {
            Y1 = v4_reduce_pm1n(Y1, n, ninv);
            Y3 = v4_mulmod(Y3, v4_set1(Q.w2tab[1][0]), n, ninv);
        } else {
            Y1 = v4_mulmod(Y1, v4_set1(Q.w2tab[1 + j_bits][2 * j_r]), n, ninv);
            Y3 = v4_mulmod(Y3, v4_set1(Q.w2tab[1 + j_bits][2 * j_r + 1]), n, ninv);
        }
        x0 = v4_add(Y0, Y1); x1 = v4_sub(Y0, Y1);
        x2 = v4_add(Y2, Y3); x3 = v4_sub(Y2, Y3);
    }

    // Row twiddles
    V4 rw2, rw, riw;
    if constexpr (J0) {
        V4 u = v4_load(Q.w2tab[0]), v = v4_load(Q.w2tab[0] + 4);
        rw2 = u; rw = v4_unpack_lo_perm(u, v); riw = v4_unpack_hi_perm(u, v);
    } else {
        V4 u = v4_load(Q.w2tab[3 + j_bits] + 8 * j_r);
        V4 v = v4_load(Q.w2tab[3 + j_bits] + 8 * j_r + 4);
        rw2 = v4_load(Q.w2tab[2 + j_bits] + 4 * j_r);
        rw = v4_unpack_lo_perm(u, v); riw = v4_unpack_hi_perm(u, v);
    }

    v4_transpose(x0, x1, x2, x3, x0, x1, x2, x3);

    // Row FFTs
    {
        V4 X0 = v4_reduce_pm1n(x0, n, ninv);
        V4 X2 = v4_mulmod(x2, rw2, n, ninv);
        V4 X3 = v4_mulmod(x3, rw2, n, ninv);
        V4 Y0 = v4_add(X0, X2), Y1 = v4_add(x1, X3);
        V4 Y2 = v4_sub(X0, X2), Y3 = v4_sub(x1, X3);
        Y1 = v4_mulmod(Y1, rw, n, ninv);
        Y3 = v4_mulmod(Y3, riw, n, ninv);
        x0 = v4_add(Y0, Y1); x1 = v4_sub(Y0, Y1);
        x2 = v4_add(Y2, Y3); x3 = v4_sub(Y2, Y3);
    }

    v4_store(X + 0, x0); v4_store(X + 4, x1);
    v4_store(X + 8, x2); v4_store(X + 12, x3);
}

// ----------------------------------------------------------------
// Forward basecase depth 6..8: EXTEND via radix-4 - template on J0
// ----------------------------------------------------------------
inline void fft_basecase(const FftCtx& Q, double* X, int depth, std::size_t j);

template <bool J0>
inline void fft_basecase_extend(const FftCtx& Q, double* X, int m, std::size_t j) {
    std::size_t l = pow2(m - 2);
    V4 n = v4_set1(Q.p), ninv = v4_set1(Q.pinv);
    if constexpr (J0) {
        V4 iw = v4_set1(Q.w2tab[1][0]);
        for (std::size_t i = 0; i < l; i += 4) {
            V4 x0 = v4_reduce_pm1n(v4_load(X + 0*l + i), n, ninv);
            V4 x1 = v4_load(X + 1*l + i);
            V4 x2 = v4_reduce_pm1n(v4_load(X + 2*l + i), n, ninv);
            V4 x3 = v4_reduce_pm1n(v4_load(X + 3*l + i), n, ninv);
            V4 y0 = v4_add(x0, x2), y1 = v4_add(x1, x3);
            V4 y2 = v4_sub(x0, x2), y3 = v4_sub(x1, x3);
            y1 = v4_reduce_pm1n(y1, n, ninv);
            y3 = v4_mulmod(y3, iw, n, ninv);
            v4_store(X + 0*l + i, v4_add(y0, y1));
            v4_store(X + 1*l + i, v4_sub(y0, y1));
            v4_store(X + 2*l + i, v4_add(y2, y3));
            v4_store(X + 3*l + i, v4_sub(y2, y3));
        }
    } else {
        int j_bits = nbits_nz(j); std::size_t j_r = j - pow2(j_bits - 1);
        V4 w  = v4_set1(Q.w2tab[1 + j_bits][2 * j_r]);
        V4 w2 = v4_set1(Q.w2tab[j_bits][j_r]);
        V4 iw = v4_set1(Q.w2tab[1 + j_bits][2 * j_r + 1]);
        for (std::size_t i = 0; i < l; i += 4) {
            V4 x0 = v4_reduce_pm1n(v4_load(X + 0*l + i), n, ninv);
            V4 x1 = v4_load(X + 1*l + i);
            V4 x2 = v4_mulmod(v4_load(X + 2*l + i), w2, n, ninv);
            V4 x3 = v4_mulmod(v4_load(X + 3*l + i), w2, n, ninv);
            V4 y0 = v4_add(x0, x2), y1 = v4_add(x1, x3);
            V4 y2 = v4_sub(x0, x2), y3 = v4_sub(x1, x3);
            y1 = v4_mulmod(y1, w, n, ninv);
            y3 = v4_mulmod(y3, iw, n, ninv);
            v4_store(X + 0*l + i, v4_add(y0, y1));
            v4_store(X + 1*l + i, v4_sub(y0, y1));
            v4_store(X + 2*l + i, v4_add(y2, y3));
            v4_store(X + 3*l + i, v4_sub(y2, y3));
        }
    }
    fft_basecase(Q, X + 0*l, m - 2, 4*j + 0);
    fft_basecase(Q, X + 1*l, m - 2, 4*j + 1);
    fft_basecase(Q, X + 2*l, m - 2, 4*j + 2);
    fft_basecase(Q, X + 3*l, m - 2, 4*j + 3);
}

// Unified forward dispatch (depth >= 4 only; min transform is 256 = depth 8)
inline void fft_basecase(const FftCtx& Q, double* X, int depth, std::size_t j) {
    if (depth == 4) {
        if (j == 0) fft_basecase_4<true>(Q, X, 0);
        else        fft_basecase_4<false>(Q, X, j);
        return;
    }
    if (j == 0) fft_basecase_extend<true>(Q, X, depth, 0);
    else        fft_basecase_extend<false>(Q, X, depth, j);
}

// Entry for block-sized basecase (256 points)
inline void fft_base8(const FftCtx& Q, double* x, std::size_t j) {
    fft_basecase(Q, x, LG_BLK_SZ, j);
}

// ================================================================
// Forward FFT: block-level recursion (within-block twiddles)
// ================================================================
inline void fft_block(const FftCtx& Q, double* x,
                          std::size_t S, int k, std::size_t j) {
    if (k <= 0) return;

    std::size_t j_r = 0;
    int j_bits = 0;
    if (j != 0) { j_bits = nbits_nz(j); j_r = j - pow2(j_bits - 1); }

    if (k >= 2) {
        int k1 = 2, k2 = k - 2;
        std::size_t l2 = pow2(k2);
        V4 n = v4_set1(Q.p), ninv = v4_set1(Q.pinv);

        if (j == 0) {
            V4 iw = v4_set1(Q.w2tab[1][0]);
            for (std::size_t a = 0; a < l2; ++a) {
                double* X0 = x + BLK_SZ * (a*S + (S << k2) * 0);
                double* X1 = x + BLK_SZ * (a*S + (S << k2) * 1);
                double* X2 = x + BLK_SZ * (a*S + (S << k2) * 2);
                double* X3 = x + BLK_SZ * (a*S + (S << k2) * 3);
                fwd_radix4_v4_j0(n, ninv, iw, X0, X1, X2, X3);
            }
        } else {
            V4 w  = v4_set1(Q.w2tab[1 + j_bits][2 * j_r]);
            V4 w2 = v4_set1(Q.w2tab[j_bits][j_r]);
            V4 iw = v4_set1(Q.w2tab[1 + j_bits][2 * j_r + 1]);
            for (std::size_t a = 0; a < l2; ++a) {
                double* X0 = x + BLK_SZ * (a*S + (S << k2) * 0);
                double* X1 = x + BLK_SZ * (a*S + (S << k2) * 1);
                double* X2 = x + BLK_SZ * (a*S + (S << k2) * 2);
                double* X3 = x + BLK_SZ * (a*S + (S << k2) * 3);
                fwd_radix4_v4_jnz(n, ninv, w, w2, iw, X0, X1, X2, X3);
            }
        }

        if (l2 == 1) return;

        std::size_t l1 = pow2(k1);
        for (std::size_t b = 0; b < l1; ++b)
            fft_block(Q, x + BLK_SZ * ((b << k2) * S), S, k2, (j << k1) + b);
    } else {
        // k == 1: radix-2
        double* X0 = x + BLK_SZ * (S * 0);
        double* X1 = x + BLK_SZ * (S * 1);
        V4 n = v4_set1(Q.p), ninv = v4_set1(Q.pinv);
        if (j == 0) {
            fwd_radix2_v4_j0(n, ninv, X0, X1);
        } else {
            V4 w = v4_set1(Q.w2tab[j_bits][j_r]);
            fwd_radix2_v4_jnz(n, ninv, w, X0, X1);
        }
    }
}

// ================================================================
// Forward FFT: outer-level recursion (across blocks)
// ================================================================
inline void fft_internal(const FftCtx& Q, double* x,
                             std::size_t S, int k, std::size_t j) {
    if (k > 2) {
        int k1 = k / 2, k2 = k - k1;
        std::size_t l2 = pow2(k2);

        for (std::size_t a = 0; a < l2; ++a)
            fft_block(Q, x + BLK_SZ * (a * S), S << k2, k1, j);

        std::size_t l1 = pow2(k1);
        for (std::size_t b = 0; b < l1; ++b)
            fft_internal(Q, x + BLK_SZ * (b * (S << k2)), S, k2, (j << k1) + b);
        return;
    }

    if (k == 2) {
        fft_block(Q, x, S, 2, j);
        fft_base8(Q, x + BLK_SZ * (S * 0), 4 * j + 0);
        fft_base8(Q, x + BLK_SZ * (S * 1), 4 * j + 1);
        fft_base8(Q, x + BLK_SZ * (S * 2), 4 * j + 2);
        fft_base8(Q, x + BLK_SZ * (S * 3), 4 * j + 3);
    } else if (k == 1) {
        V4 n = v4_set1(Q.p), ninv = v4_set1(Q.pinv);
        double* X0 = x; double* X1 = x + BLK_SZ * S;
        if (j == 0) {
            fwd_radix2_v4_j0(n, ninv, X0, X1);
        } else {
            int jb = nbits_nz(j); std::size_t jr = j - pow2(jb - 1);
            V4 w = v4_set1(Q.w2tab[jb][jr]);
            fwd_radix2_v4_jnz(n, ninv, w, X0, X1);
        }
        fft_base8(Q, X0, 2 * j + 0);
        fft_base8(Q, X1, 2 * j + 1);
    } else {
        fft_base8(Q, x, j);
    }
}

// ================================================================
// Forward FFT entry point
// ================================================================
inline void fft(FftCtx& Q, double* d, int L) {
    if (L <= LG_BLK_SZ) {
        fft_basecase(Q, d, L, 0);
        return;
    }
    Q.fit_depth(L);
    fft_internal(Q, d, 1, L - LG_BLK_SZ, 0);
}

// ================================================================
// Inverse FFT: butterflies
// ================================================================
__forceinline void inv_radix4_v4_j0(V4 n, V4 ninv, V4 iW,
                              double* X0, double* X1, double* X2, double* X3) {
    for (std::size_t i = 0; i < BLK_SZ; i += 4) {
        V4 x0 = v4_load(X0 + i), x1 = v4_load(X1 + i);
        V4 x2 = v4_load(X2 + i), x3 = v4_load(X3 + i);
        V4 y0 = v4_add(x0, x1);
        V4 y1 = v4_sub(x0, x1);
        V4 y2 = v4_add(x2, x3);
        V4 y3 = v4_mulmod(v4_sub(x2, x3), iW, n, ninv);
        y0 = v4_reduce_pm1n(y0, n, ninv);
        y1 = v4_reduce_pm1n(y1, n, ninv);
        y2 = v4_reduce_pm1n(y2, n, ninv);
        v4_store(X0 + i, v4_add(y0, y2));
        v4_store(X1 + i, v4_add(y1, y3));
        v4_store(X2 + i, v4_sub(y0, y2));
        v4_store(X3 + i, v4_sub(y1, y3));
    }
}

__forceinline void inv_radix4_v4_jnz(V4 n, V4 ninv, V4 W, V4 W2, V4 iW,
                                double* X0, double* X1, double* X2, double* X3) {
    for (std::size_t i = 0; i < BLK_SZ; i += 4) {
        V4 x0 = v4_load(X0 + i), x1 = v4_load(X1 + i);
        V4 x2 = v4_load(X2 + i), x3 = v4_load(X3 + i);
        V4 y0 = v4_add(x0, x1);
        V4 y1 = v4_mulmod(v4_sub(x1, x0), W, n, ninv);
        V4 y2 = v4_add(x2, x3);
        V4 y3 = v4_mulmod(v4_sub(x3, x2), iW, n, ninv);
        y0 = v4_reduce_pm1n(y0, n, ninv);
        y2 = v4_reduce_pm1n(y2, n, ninv);
        V4 z0 = v4_add(y0, y2);
        V4 z2 = v4_mulmod(v4_sub(y0, y2), W2, n, ninv);
        v4_store(X0 + i, z0);
        v4_store(X2 + i, z2);
        v4_store(X1 + i, v4_add(y1, y3));
        v4_store(X3 + i, v4_mulmod(v4_sub(y1, y3), W2, n, ninv));
    }
}

__forceinline void inv_radix2_v4_j0(V4 n, V4 ninv, double* X0, double* X1) {
    for (std::size_t i = 0; i < BLK_SZ; i += 4) {
        V4 x0 = v4_load(X0 + i), x1 = v4_load(X1 + i);
        V4 y0 = v4_reduce_pm1n(v4_add(x0, x1), n, ninv);
        V4 y1 = v4_reduce_pm1n(v4_sub(x0, x1), n, ninv);
        v4_store(X0 + i, y0);
        v4_store(X1 + i, y1);
    }
}

__forceinline void inv_radix2_v4_jnz(V4 n, V4 ninv, V4 W, double* X0, double* X1) {
    for (std::size_t i = 0; i < BLK_SZ; i += 4) {
        V4 x0 = v4_load(X0 + i), x1 = v4_load(X1 + i);
        V4 y0 = v4_reduce_pm1n(v4_add(x0, x1), n, ninv);
        V4 y1 = v4_mulmod(v4_sub(x1, x0), W, n, ninv);
        v4_store(X0 + i, y0);
        v4_store(X1 + i, y1);
    }
}

// ================================================================
// Inverse FFT: block-level recursion (DIT: recurse first, butterfly after)
// ================================================================
inline void ifft_block(const FftCtx& Q, double* x,
                           std::size_t S, int k, std::size_t j) {
    if (k <= 0) return;

    std::size_t j_mr = 0;
    int j_bits = 0;
    if (j != 0) { j_bits = nbits_nz(j); j_mr = pow2(j_bits) - 1 - j; }

    if (k >= 2) {
        int k1 = 2, k2 = k - 2;
        std::size_t l2 = pow2(k2);

        if (l2 > 1) {
            std::size_t l1 = pow2(k1);
            for (std::size_t b = 0; b < l1; ++b)
                ifft_block(Q, x + BLK_SZ * ((b << k2) * S), S, k2, (j << k1) + b);
        }

        V4 n = v4_set1(Q.p), ninv = v4_set1(Q.pinv);
        if (j == 0) {
            V4 iW = v4_neg(v4_set1(Q.w2tab[1][0]));
            for (std::size_t a = 0; a < l2; ++a) {
                double* X0 = x + BLK_SZ * (a*S + (S << k2) * 0);
                double* X1 = x + BLK_SZ * (a*S + (S << k2) * 1);
                double* X2 = x + BLK_SZ * (a*S + (S << k2) * 2);
                double* X3 = x + BLK_SZ * (a*S + (S << k2) * 3);
                inv_radix4_v4_j0(n, ninv, iW, X0, X1, X2, X3);
            }
        } else {
            V4 W  = v4_set1(Q.w2tab[1 + j_bits][2 * j_mr + 1]);
            V4 W2 = v4_neg(v4_set1(Q.w2tab[j_bits][j_mr]));
            V4 iW = v4_set1(Q.w2tab[1 + j_bits][2 * j_mr]);
            for (std::size_t a = 0; a < l2; ++a) {
                double* X0 = x + BLK_SZ * (a*S + (S << k2) * 0);
                double* X1 = x + BLK_SZ * (a*S + (S << k2) * 1);
                double* X2 = x + BLK_SZ * (a*S + (S << k2) * 2);
                double* X3 = x + BLK_SZ * (a*S + (S << k2) * 3);
                inv_radix4_v4_jnz(n, ninv, W, W2, iW, X0, X1, X2, X3);
            }
        }
    } else {
        double* X0 = x + BLK_SZ * (S * 0);
        double* X1 = x + BLK_SZ * (S * 1);
        V4 n = v4_set1(Q.p), ninv = v4_set1(Q.pinv);
        if (j == 0) {
            inv_radix2_v4_j0(n, ninv, X0, X1);
        } else {
            V4 W = v4_set1(Q.w2tab[j_bits][j_mr]);
            inv_radix2_v4_jnz(n, ninv, W, X0, X1);
        }
    }
}

// ================================================================
// Inverse FFT basecases (depth 4..8)
// ================================================================

template <bool J0>
inline void ifft_basecase_4(const FftCtx& Q, double* X, std::size_t j) {
    V4 n = v4_set1(Q.p), ninv = v4_set1(Q.pinv);
    V4 x0 = v4_load(X + 0), x1 = v4_load(X + 4);
    V4 x2 = v4_load(X + 8), x3 = v4_load(X + 12);
    int j_bits = 0; std::size_t j_r = 0;
    if constexpr (!J0) { j_bits = nbits_nz(j); j_r = j - pow2(j_bits - 1); }

    // Inverse row butterflies
    V4 rW, riW, rW2;
    if constexpr (J0) {
        rW  = v4_set_d4(-1.0, Q.w2tab[2][1], Q.w2tab[3][3], Q.w2tab[3][1]);
        riW = v4_set_d4(Q.w2tab[1][0], Q.w2tab[2][0], Q.w2tab[3][2], Q.w2tab[3][0]);
        rW2 = v4_set_d4(1.0, -Q.w2tab[1][0], -Q.w2tab[2][1], -Q.w2tab[2][0]);
    } else {
        std::size_t mj = pow2(j_bits - 1) - 1 - j_r;
        V4 mirr_rw2 = v4_load(Q.w2tab[2 + j_bits] + 4 * mj);
        V4 mu = v4_load(Q.w2tab[3 + j_bits] + 8 * mj);
        V4 mv = v4_load(Q.w2tab[3 + j_bits] + 8 * mj + 4);
        rW  = v4_reverse(v4_unpack_hi_perm(mu, mv));
        riW = v4_reverse(v4_unpack_lo_perm(mu, mv));
        rW2 = v4_neg(v4_reverse(mirr_rw2));
    }
    {
        V4 y0 = v4_reduce_pm1n(v4_add(x0, x1), n, ninv);
        V4 y1 = v4_mulmod(v4_sub(x1, x0), rW, n, ninv);
        V4 y2 = v4_reduce_pm1n(v4_add(x2, x3), n, ninv);
        V4 y3 = v4_mulmod(v4_sub(x3, x2), riW, n, ninv);
        V4 z0 = v4_add(y0, y2);
        V4 z2 = v4_mulmod(v4_sub(y0, y2), rW2, n, ninv);
        x0 = z0; x1 = v4_add(y1, y3);
        x2 = z2; x3 = v4_mulmod(v4_sub(y1, y3), rW2, n, ninv);
    }

    v4_transpose(x0, x1, x2, x3, x0, x1, x2, x3);

    // Inverse column butterfly
    if constexpr (J0) {
        V4 iW_col = v4_neg(v4_set1(Q.w2tab[1][0]));
        V4 y0 = v4_reduce_pm1n(v4_add(x0, x1), n, ninv);
        V4 y1 = v4_reduce_pm1n(v4_sub(x0, x1), n, ninv);
        V4 y2 = v4_reduce_pm1n(v4_add(x2, x3), n, ninv);
        V4 y3 = v4_mulmod(v4_sub(x2, x3), iW_col, n, ninv);
        x0 = v4_add(y0, y2); x1 = v4_add(y1, y3);
        x2 = v4_sub(y0, y2); x3 = v4_sub(y1, y3);
    } else {
        std::size_t nb = pow2(j_bits);
        V4 cW  = v4_set1(Q.w2tab[1 + j_bits][nb - 1 - 2 * j_r]);
        V4 ciW = v4_set1(Q.w2tab[1 + j_bits][nb - 2 - 2 * j_r]);
        V4 cW2 = v4_neg(v4_set1(Q.w2tab[j_bits][pow2(j_bits - 1) - 1 - j_r]));
        V4 y0 = v4_reduce_pm1n(v4_add(x0, x1), n, ninv);
        V4 y1 = v4_mulmod(v4_sub(x1, x0), cW, n, ninv);
        V4 y2 = v4_reduce_pm1n(v4_add(x2, x3), n, ninv);
        V4 y3 = v4_mulmod(v4_sub(x3, x2), ciW, n, ninv);
        V4 z0 = v4_add(y0, y2);
        V4 z2 = v4_mulmod(v4_sub(y0, y2), cW2, n, ninv);
        x0 = z0; x1 = v4_add(y1, y3);
        x2 = z2; x3 = v4_mulmod(v4_sub(y1, y3), cW2, n, ninv);
    }

    v4_store(X + 0, x0); v4_store(X + 4, x1);
    v4_store(X + 8, x2); v4_store(X + 12, x3);
}

// Inverse basecase depth 6..8: EXTEND via radix-4 (DIT)
inline void ifft_basecase(const FftCtx& Q, double* X, int depth, std::size_t j);

template <bool J0>
inline void ifft_basecase_extend(const FftCtx& Q, double* X, int m, std::size_t j) {
    std::size_t l = pow2(m - 2);
    // DIT: sub-IDFTs first
    ifft_basecase(Q, X + 0*l, m - 2, 4*j + 0);
    ifft_basecase(Q, X + 1*l, m - 2, 4*j + 1);
    ifft_basecase(Q, X + 2*l, m - 2, 4*j + 2);
    ifft_basecase(Q, X + 3*l, m - 2, 4*j + 3);

    V4 n = v4_set1(Q.p), ninv = v4_set1(Q.pinv);
    if constexpr (J0) {
        V4 iW = v4_neg(v4_set1(Q.w2tab[1][0]));
        for (std::size_t i = 0; i < l; i += 4) {
            V4 x0 = v4_load(X + 0*l + i), x1 = v4_load(X + 1*l + i);
            V4 x2 = v4_load(X + 2*l + i), x3 = v4_load(X + 3*l + i);
            V4 y0 = v4_reduce_pm1n(v4_add(x0, x1), n, ninv);
            V4 y1 = v4_reduce_pm1n(v4_sub(x0, x1), n, ninv);
            V4 y2 = v4_reduce_pm1n(v4_add(x2, x3), n, ninv);
            V4 y3 = v4_mulmod(v4_sub(x2, x3), iW, n, ninv);
            v4_store(X + 0*l + i, v4_add(y0, y2));
            v4_store(X + 1*l + i, v4_add(y1, y3));
            v4_store(X + 2*l + i, v4_sub(y0, y2));
            v4_store(X + 3*l + i, v4_sub(y1, y3));
        }
    } else {
        int j_bits = nbits_nz(j); std::size_t j_r = j - pow2(j_bits - 1);
        std::size_t nb = pow2(j_bits);
        V4 W  = v4_set1(Q.w2tab[1 + j_bits][nb - 1 - 2 * j_r]);
        V4 iW = v4_set1(Q.w2tab[1 + j_bits][nb - 2 - 2 * j_r]);
        V4 W2 = v4_neg(v4_set1(Q.w2tab[j_bits][(nb >> 1) - 1 - j_r]));
        for (std::size_t i = 0; i < l; i += 4) {
            V4 x0 = v4_load(X + 0*l + i), x1 = v4_load(X + 1*l + i);
            V4 x2 = v4_load(X + 2*l + i), x3 = v4_load(X + 3*l + i);
            V4 y0 = v4_reduce_pm1n(v4_add(x0, x1), n, ninv);
            V4 y1 = v4_mulmod(v4_sub(x1, x0), W, n, ninv);
            V4 y2 = v4_reduce_pm1n(v4_add(x2, x3), n, ninv);
            V4 y3 = v4_mulmod(v4_sub(x3, x2), iW, n, ninv);
            V4 z0 = v4_add(y0, y2);
            V4 z2 = v4_mulmod(v4_sub(y0, y2), W2, n, ninv);
            v4_store(X + 0*l + i, z0);
            v4_store(X + 2*l + i, z2);
            v4_store(X + 1*l + i, v4_add(y1, y3));
            v4_store(X + 3*l + i, v4_mulmod(v4_sub(y1, y3), W2, n, ninv));
        }
    }
}

// Unified inverse dispatch (depth >= 4 only)
inline void ifft_basecase(const FftCtx& Q, double* X, int depth, std::size_t j) {
    if (depth == 4) {
        if (j == 0) ifft_basecase_4<true>(Q, X, 0);
        else        ifft_basecase_4<false>(Q, X, j);
        return;
    }
    if (j == 0) ifft_basecase_extend<true>(Q, X, depth, 0);
    else        ifft_basecase_extend<false>(Q, X, depth, j);
}

inline void ifft_base8(const FftCtx& Q, double* x, std::size_t j) {
    ifft_basecase(Q, x, LG_BLK_SZ, j);
}

inline void ifft_internal(const FftCtx& Q, double* x,
                              std::size_t S, int k, std::size_t j) {
    if (k > 2) {
        int k1 = k / 2, k2 = k - k1;

        std::size_t l1 = pow2(k1);
        for (std::size_t b = 0; b < l1; ++b)
            ifft_internal(Q, x + BLK_SZ * (b * (S << k2)), S, k2, (j << k1) + b);

        std::size_t l2 = pow2(k2);
        for (std::size_t a = 0; a < l2; ++a)
            ifft_block(Q, x + BLK_SZ * (a * S), S << k2, k1, j);
        return;
    }

    if (k == 2) {
        ifft_base8(Q, x + BLK_SZ * (S * 0), 4 * j + 0);
        ifft_base8(Q, x + BLK_SZ * (S * 1), 4 * j + 1);
        ifft_base8(Q, x + BLK_SZ * (S * 2), 4 * j + 2);
        ifft_base8(Q, x + BLK_SZ * (S * 3), 4 * j + 3);
        ifft_block(Q, x, S, 2, j);
    } else if (k == 1) {
        double* X0 = x; double* X1 = x + BLK_SZ * S;
        ifft_base8(Q, X0, 2 * j + 0);
        ifft_base8(Q, X1, 2 * j + 1);
        V4 n = v4_set1(Q.p), ninv = v4_set1(Q.pinv);
        if (j == 0) {
            inv_radix2_v4_j0(n, ninv, X0, X1);
        } else {
            int jb = nbits_nz(j); std::size_t jmr = pow2(jb) - 1 - j;
            V4 W = v4_set1(Q.w2tab[jb][jmr]);
            inv_radix2_v4_jnz(n, ninv, W, X0, X1);
        }
    } else {
        ifft_base8(Q, x, j);
    }
}

// ================================================================
// Inverse FFT entry point
// ================================================================
inline void ifft(FftCtx& Q, double* d, int L) {
    if (L <= LG_BLK_SZ) {
        ifft_basecase(Q, d, L, 0);
        return;
    }
    Q.fit_depth(L);
    ifft_internal(Q, d, 1, L - LG_BLK_SZ, 0);
}

}} // namespace ntt::p50x4
