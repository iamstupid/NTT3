#pragma once
// bailey.hpp - Bailey's 4-step FFT for large power-of-2 transforms
//
// Part of ntt::p50x4 - 4-prime ~50-bit NTT (double FMA Barrett)

#include "fft.hpp"

namespace ntt { namespace p50x4 {

// Threshold: use Bailey for L >= BAILEY_MIN_L
static constexpr int BAILEY_MIN_L = 27;

// Cache-oblivious out-of-place transpose with AVX2 4x4 micro-kernels.
static constexpr std::size_t TRANSPOSE_TILE = 64;

// 4x4 AVX2 micro-kernel: transpose src[r:r+4][c:c+4] -> dst[c:c+4][r:r+4]
static __forceinline void transpose_4x4_kernel(
        double* __restrict dst, const double* __restrict src,
        std::size_t R, std::size_t C,
        std::size_t r, std::size_t c) {
    V4 x0 = v4_load(src + r * C + c);
    V4 x1 = v4_load(src + (r + 1) * C + c);
    V4 x2 = v4_load(src + (r + 2) * C + c);
    V4 x3 = v4_load(src + (r + 3) * C + c);
    V4 y0, y1, y2, y3;
    v4_transpose(y0, y1, y2, y3, x0, x1, x2, x3);
    v4_stream(dst + c * R + r, y0);
    v4_stream(dst + (c + 1) * R + r, y1);
    v4_stream(dst + (c + 2) * R + r, y2);
    v4_stream(dst + (c + 3) * R + r, y3);
}

// Base-case tile: column-outer order keeps only 4 destination write streams active.
static inline void transpose_tile(
        double* __restrict dst, const double* __restrict src,
        std::size_t R, std::size_t C,
        std::size_t r0, std::size_t c0,
        std::size_t rn, std::size_t cn) {
    for (std::size_t cc = c0; cc < c0 + cn; cc += 4) {
        for (std::size_t rr = r0; rr < r0 + rn; rr += 4) {
            transpose_4x4_kernel(dst, src, R, C, rr, cc);
        }
    }
}

// Recursive cache-oblivious splitter.
static void transpose_rec(
        double* __restrict dst, const double* __restrict src,
        std::size_t R, std::size_t C,
        std::size_t r0, std::size_t c0,
        std::size_t rn, std::size_t cn) {
    if (rn <= TRANSPOSE_TILE && cn <= TRANSPOSE_TILE) {
        transpose_tile(dst, src, R, C, r0, c0, rn, cn);
        return;
    }
    if (rn >= cn) {
        std::size_t half = rn >> 1;
        transpose_rec(dst, src, R, C, r0, c0, half, cn);
        transpose_rec(dst, src, R, C, r0 + half, c0, rn - half, cn);
    } else {
        std::size_t half = cn >> 1;
        transpose_rec(dst, src, R, C, r0, c0, rn, half);
        transpose_rec(dst, src, R, C, r0, c0 + half, rn, cn - half);
    }
}

inline void bailey_transpose(double* dst, const double* src,
                              std::size_t R, std::size_t C) {
    transpose_rec(dst, src, R, C, 0, 0, R, C);
    _mm_sfence();
}

// Apply twiddle factors: data[r][c] *= omega_N^{r*c}
inline void bailey_twiddle_fwd(const FftCtx& Q, double* data,
                                std::size_t R, std::size_t C, double omega_N) {
    V4 n = v4_set1(Q.p), ninv = v4_set1(Q.pinv);
    double omega_r = 1.0;

    for (std::size_t r = 1; r < R; ++r) {
        omega_r = s_mulmod(omega_r, omega_N, Q.p, Q.pinv);
        double step = omega_r;
        double* row = data + r * C;

        V4 tw_step4;
        V4 tw0 = v4_build_tw(1.0, step, Q.p, Q.pinv, tw_step4);
        V4 tw1 = v4_mulmod(tw0, tw_step4, n, ninv);
        V4 tw2 = v4_mulmod(tw1, tw_step4, n, ninv);
        V4 tw3 = v4_mulmod(tw2, tw_step4, n, ninv);
        V4 tw_step8 = v4_mulmod(tw_step4, tw_step4, n, ninv);
        V4 tw_step16 = v4_mulmod(tw_step8, tw_step8, n, ninv);

        std::size_t c = 0;
        for (; c + 15 < C; c += 16) {
            v4_store(row + c,      v4_mulmod(v4_load(row + c),      tw0, n, ninv));
            v4_store(row + c + 4,  v4_mulmod(v4_load(row + c + 4),  tw1, n, ninv));
            v4_store(row + c + 8,  v4_mulmod(v4_load(row + c + 8),  tw2, n, ninv));
            v4_store(row + c + 12, v4_mulmod(v4_load(row + c + 12), tw3, n, ninv));
            tw0 = v4_mulmod(tw0, tw_step16, n, ninv);
            tw1 = v4_mulmod(tw1, tw_step16, n, ninv);
            tw2 = v4_mulmod(tw2, tw_step16, n, ninv);
            tw3 = v4_mulmod(tw3, tw_step16, n, ninv);
        }
        V4 tw = tw0;
        for (; c + 3 < C; c += 4) {
            v4_store(row + c, v4_mulmod(v4_load(row + c), tw, n, ninv));
            tw = v4_mulmod(tw, tw_step4, n, ninv);
        }
        double running = _mm256_cvtsd_f64(tw);
        for (; c < C; ++c) {
            row[c] = s_mulmod(row[c], running, Q.p, Q.pinv);
            running = s_mulmod(running, step, Q.p, Q.pinv);
        }
    }
}

// Inverse twiddle: data[r][c] *= omega_N^{-r*c}
inline void bailey_twiddle_inv(const FftCtx& Q, double* data,
                                std::size_t R, std::size_t C, double omega_Ni) {
    V4 n = v4_set1(Q.p), ninv = v4_set1(Q.pinv);
    double omega_r = 1.0;

    for (std::size_t r = 1; r < R; ++r) {
        omega_r = s_mulmod(omega_r, omega_Ni, Q.p, Q.pinv);
        double step = omega_r;
        double* row = data + r * C;

        V4 tw_step4;
        V4 tw0 = v4_build_tw(1.0, step, Q.p, Q.pinv, tw_step4);
        V4 tw1 = v4_mulmod(tw0, tw_step4, n, ninv);
        V4 tw2 = v4_mulmod(tw1, tw_step4, n, ninv);
        V4 tw3 = v4_mulmod(tw2, tw_step4, n, ninv);
        V4 tw_step8 = v4_mulmod(tw_step4, tw_step4, n, ninv);
        V4 tw_step16 = v4_mulmod(tw_step8, tw_step8, n, ninv);

        std::size_t c = 0;
        for (; c + 15 < C; c += 16) {
            v4_store(row + c,      v4_mulmod(v4_load(row + c),      tw0, n, ninv));
            v4_store(row + c + 4,  v4_mulmod(v4_load(row + c + 4),  tw1, n, ninv));
            v4_store(row + c + 8,  v4_mulmod(v4_load(row + c + 8),  tw2, n, ninv));
            v4_store(row + c + 12, v4_mulmod(v4_load(row + c + 12), tw3, n, ninv));
            tw0 = v4_mulmod(tw0, tw_step16, n, ninv);
            tw1 = v4_mulmod(tw1, tw_step16, n, ninv);
            tw2 = v4_mulmod(tw2, tw_step16, n, ninv);
            tw3 = v4_mulmod(tw3, tw_step16, n, ninv);
        }
        V4 tw = tw0;
        for (; c + 3 < C; c += 4) {
            v4_store(row + c, v4_mulmod(v4_load(row + c), tw, n, ninv));
            tw = v4_mulmod(tw, tw_step4, n, ninv);
        }
        double running = _mm256_cvtsd_f64(tw);
        for (; c < C; ++c) {
            row[c] = s_mulmod(row[c], running, Q.p, Q.pinv);
            running = s_mulmod(running, step, Q.p, Q.pinv);
        }
    }
}

// Bailey 4-step forward FFT for N = 2^L
inline void fft_bailey(FftCtx& Q, double* d, int L) {
    assert(L >= 16 && "Bailey requires L >= 16 (sub-FFTs must be >= 256)");
    int L1 = L / 2;
    int L2 = L - L1;
    std::size_t R = pow2(L1);
    std::size_t C = pow2(L2);
    std::size_t N = R * C;

    Q.fit_depth(L);

    double omega_N = s_reduce_0n_to_pmhn(
        static_cast<double>(pow_mod(Q.prim_root, (Q.prime - 1) / N, Q.prime)), Q.p);

    double* tmp = Q.ensure_bailey_tmp(N);

    // Step 1: C-point FFTs on each of R rows
    for (std::size_t r = 0; r < R; ++r)
        fft(Q, d + r * C, L2);

    // Step 2: Multiply by twiddle factors
    bailey_twiddle_fwd(Q, d, R, C, omega_N);

    // Step 3: Transpose R*C -> C*R
    bailey_transpose(tmp, d, R, C);

    // Step 4: R-point FFTs on each of C rows
    for (std::size_t c = 0; c < C; ++c)
        fft(Q, tmp + c * R, L1);

    // Step 5: Transpose back C*R -> R*C
    bailey_transpose(d, tmp, C, R);
}

// Bailey 4-step inverse FFT for N = 2^L
inline void ifft_bailey(FftCtx& Q, double* d, int L) {
    assert(L >= 16 && "Bailey requires L >= 16 (sub-FFTs must be >= 256)");
    int L1 = L / 2;
    int L2 = L - L1;
    std::size_t R = pow2(L1);
    std::size_t C = pow2(L2);
    std::size_t N = R * C;

    Q.fit_depth(L);

    double omega_Ni = s_reduce_0n_to_pmhn(
        static_cast<double>(pow_mod(Q.prim_root, Q.prime - 1 - (Q.prime - 1) / N, Q.prime)), Q.p);

    double* tmp = Q.ensure_bailey_tmp(N);

    // Step 1: Transpose R*C -> C*R
    bailey_transpose(tmp, d, R, C);

    // Step 2: R-point IFFTs on each of C rows
    for (std::size_t c = 0; c < C; ++c)
        ifft(Q, tmp + c * R, L1);

    // Step 3: Transpose back C*R -> R*C
    bailey_transpose(d, tmp, C, R);

    // Step 4: Inverse twiddle
    bailey_twiddle_inv(Q, d, R, C, omega_Ni);

    // Step 5: C-point IFFTs on each of R rows
    for (std::size_t r = 0; r < R; ++r)
        ifft(Q, d + r * C, L2);
}

}} // namespace ntt::p50x4
