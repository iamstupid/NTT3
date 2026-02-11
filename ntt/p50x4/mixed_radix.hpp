#pragma once
// mixed_radix.hpp - Radix-3/5 outer passes, ceil_ntt_size, mixed-radix dispatch
//
// Part of ntt::p50x4 - 4-prime ~50-bit NTT (double FMA Barrett)

#include "bailey.hpp"
#include "pointmul.hpp"

namespace ntt { namespace p50x4 {

// ================================================================
// Radix-3 outer DIF pass (forward): N = 3 * sub_n, sub_n = 2^k
// ================================================================
inline void radix3_dif_pass(const FftCtx& Q, double* f, std::size_t N) {
    std::size_t sub_n = N / 3;
    int k = log2_exact(sub_n);
    V4 n = v4_set1(Q.p), ninv = v4_set1(Q.pinv);
    V4 vneghalf = v4_set1(Q.neg_half_d);
    V4 vj3half = v4_set1(Q.j3_half_d);

    double tw_root = Q.tw3_roots_d[k];
    double tw2_root = s_mulmod(tw_root, tw_root, Q.p, Q.pinv);

    V4 tw_step, tw2_step;
    V4 tw_v  = v4_build_tw(1.0, tw_root,  Q.p, Q.pinv, tw_step);
    V4 tw2_v = v4_build_tw(1.0, tw2_root, Q.p, Q.pinv, tw2_step);

    std::size_t j = 0;
    for (; j + 3 < sub_n; j += 4) {
        V4 a = v4_load(f + j);
        V4 b = v4_load(f + sub_n + j);
        V4 c = v4_load(f + 2 * sub_n + j);

        V4 s = v4_add(b, c);
        V4 d = v4_sub(b, c);
        V4 f0 = v4_add(a, s);

        V4 hs = v4_mulmod(s, vneghalf, n, ninv);
        V4 jd = v4_mulmod(d, vj3half, n, ninv);
        V4 ahs = v4_add(v4_reduce_pm1n(a, n, ninv), hs);

        v4_store(f + j, f0);
        v4_store(f + sub_n + j, v4_mulmod(v4_add(ahs, jd), tw_v, n, ninv));
        v4_store(f + 2 * sub_n + j, v4_mulmod(v4_sub(ahs, jd), tw2_v, n, ninv));

        tw_v  = v4_mulmod(tw_v,  tw_step,  n, ninv);
        tw2_v = v4_mulmod(tw2_v, tw2_step, n, ninv);
    }

    double tw = _mm256_cvtsd_f64(tw_v);
    double tw2 = _mm256_cvtsd_f64(tw2_v);
    for (; j < sub_n; ++j) {
        double a = f[j], b = f[sub_n + j], c = f[2 * sub_n + j];
        double s = b + c, d = b - c;
        double f0 = a + s;
        double hs = s_mulmod(s, Q.neg_half_d, Q.p, Q.pinv);
        double jd = s_mulmod(d, Q.j3_half_d, Q.p, Q.pinv);
        double ahs = s_reduce_pm1n(a, Q.p, Q.pinv) + hs;
        f[j] = f0;
        f[sub_n + j] = s_mulmod(ahs + jd, tw, Q.p, Q.pinv);
        f[2 * sub_n + j] = s_mulmod(ahs - jd, tw2, Q.p, Q.pinv);
        tw = s_mulmod(tw, tw_root, Q.p, Q.pinv);
        tw2 = s_mulmod(tw2, tw2_root, Q.p, Q.pinv);
    }
}

// ================================================================
// Radix-3 outer DIT pass (inverse): fused with 1/3 scale
// ================================================================
inline void radix3_dit_pass(const FftCtx& Q, double* f, std::size_t N) {
    std::size_t sub_n = N / 3;
    int k = log2_exact(sub_n);
    V4 n = v4_set1(Q.p), ninv = v4_set1(Q.pinv);
    V4 vinv3 = v4_set1(Q.inv3_d);
    V4 vneghalf = v4_set1(Q.neg_half_d);
    V4 vj3half = v4_set1(Q.j3_half_d);

    double tw_inv_root = Q.tw3i_roots_d[k];
    double tw2_inv_root = s_mulmod(tw_inv_root, tw_inv_root, Q.p, Q.pinv);

    V4 tw1_step, tw2_step;
    V4 tw1_v = v4_build_tw(Q.inv3_d, tw_inv_root,  Q.p, Q.pinv, tw1_step);
    V4 tw2_v = v4_build_tw(Q.inv3_d, tw2_inv_root, Q.p, Q.pinv, tw2_step);

    std::size_t j = 0;
    for (; j + 3 < sub_n; j += 4) {
        V4 fa = v4_mulmod(v4_load(f + j), vinv3, n, ninv);
        V4 fb = v4_mulmod(v4_load(f + sub_n + j), tw1_v, n, ninv);
        V4 fc = v4_mulmod(v4_load(f + 2 * sub_n + j), tw2_v, n, ninv);

        V4 s = v4_add(fb, fc);
        V4 d = v4_sub(fb, fc);

        V4 hs = v4_mulmod(s, vneghalf, n, ninv);
        V4 jd = v4_mulmod(d, vj3half, n, ninv);
        V4 ahs = v4_add(v4_reduce_pm1n(fa, n, ninv), hs);

        v4_store(f + j, v4_reduce_pm1n(v4_add(fa, s), n, ninv));
        v4_store(f + sub_n + j, v4_reduce_pm1n(v4_sub(ahs, jd), n, ninv));
        v4_store(f + 2 * sub_n + j, v4_reduce_pm1n(v4_add(ahs, jd), n, ninv));

        tw1_v = v4_mulmod(tw1_v, tw1_step, n, ninv);
        tw2_v = v4_mulmod(tw2_v, tw2_step, n, ninv);
    }

    double tw1_s = _mm256_cvtsd_f64(tw1_v);
    double tw2_s = _mm256_cvtsd_f64(tw2_v);
    for (; j < sub_n; ++j) {
        double fa = s_mulmod(f[j], Q.inv3_d, Q.p, Q.pinv);
        double fb = s_mulmod(f[sub_n + j], tw1_s, Q.p, Q.pinv);
        double fc = s_mulmod(f[2 * sub_n + j], tw2_s, Q.p, Q.pinv);
        double s = fb + fc, d = fb - fc;
        double hs = s_mulmod(s, Q.neg_half_d, Q.p, Q.pinv);
        double jd = s_mulmod(d, Q.j3_half_d, Q.p, Q.pinv);
        double ahs = s_reduce_pm1n(fa, Q.p, Q.pinv) + hs;
        f[j] = s_reduce_pm1n(fa + s, Q.p, Q.pinv);
        f[sub_n + j] = s_reduce_pm1n(ahs - jd, Q.p, Q.pinv);
        f[2 * sub_n + j] = s_reduce_pm1n(ahs + jd, Q.p, Q.pinv);
        tw1_s = s_mulmod(tw1_s, tw_inv_root, Q.p, Q.pinv);
        tw2_s = s_mulmod(tw2_s, tw2_inv_root, Q.p, Q.pinv);
    }
}

// ================================================================
// Radix-5 outer DIF pass (forward): Karatsuba-style 6-mul butterfly
// ================================================================
inline void radix5_dif_pass(const FftCtx& Q, double* f, std::size_t N) {
    std::size_t sub_n = N / 5;
    int k = log2_exact(sub_n);
    V4 n = v4_set1(Q.p), ninv = v4_set1(Q.pinv);
    V4 vc1h = v4_set1(Q.c1h_d), vc2h = v4_set1(Q.c2h_d), vc12h = v4_set1(Q.c12h_d);
    V4 vj1h = v4_set1(Q.j1h_d), vj2h = v4_set1(Q.j2h_d), vj12s = v4_set1(Q.j12s_d);

    double tw_root = Q.tw5_roots_d[k];
    double tw2_r = s_mulmod(tw_root, tw_root, Q.p, Q.pinv);
    double tw3_r = s_mulmod(tw2_r, tw_root, Q.p, Q.pinv);
    double tw4_r = s_mulmod(tw2_r, tw2_r, Q.p, Q.pinv);

    V4 tw1_step, tw2_step, tw3_step, tw4_step;
    V4 tw1_v = v4_build_tw(1.0, tw_root, Q.p, Q.pinv, tw1_step);
    V4 tw2_v = v4_build_tw(1.0, tw2_r,   Q.p, Q.pinv, tw2_step);
    V4 tw3_v = v4_build_tw(1.0, tw3_r,   Q.p, Q.pinv, tw3_step);
    V4 tw4_v = v4_build_tw(1.0, tw4_r,   Q.p, Q.pinv, tw4_step);

    std::size_t j = 0;
    for (; j + 3 < sub_n; j += 4) {
        V4 a = v4_load(f + j);
        V4 b = v4_load(f + sub_n + j);
        V4 c = v4_load(f + 2*sub_n + j);
        V4 d = v4_load(f + 3*sub_n + j);
        V4 e = v4_load(f + 4*sub_n + j);

        V4 s1 = v4_add(b, e), t1 = v4_sub(b, e);
        V4 s2 = v4_add(c, d), t2 = v4_sub(c, d);
        V4 f0 = v4_add(a, v4_add(s1, s2));

        V4 p1 = v4_mulmod(s1, vc1h, n, ninv);
        V4 p2 = v4_mulmod(s2, vc2h, n, ninv);
        V4 p3 = v4_mulmod(v4_add(s1, s2), vc12h, n, ninv);
        V4 pp = v4_add(p1, p2);
        V4 ra = v4_reduce_pm1n(a, n, ninv);
        V4 alpha = v4_add(ra, pp);
        V4 gamma = v4_add(ra, v4_reduce_pm1n(v4_sub(p3, pp), n, ninv));

        V4 q1 = v4_mulmod(t1, vj1h, n, ninv);
        V4 q2 = v4_mulmod(t2, vj2h, n, ninv);
        V4 q3 = v4_mulmod(v4_sub(t1, t2), vj12s, n, ninv);
        V4 beta  = v4_add(q1, q2);
        V4 delta = v4_add(v4_reduce_pm1n(v4_sub(q3, q1), n, ninv), q2);

        v4_store(f + j, f0);
        v4_store(f + sub_n + j,   v4_mulmod(v4_add(alpha, beta),  tw1_v, n, ninv));
        v4_store(f + 2*sub_n + j, v4_mulmod(v4_add(gamma, delta), tw2_v, n, ninv));
        v4_store(f + 3*sub_n + j, v4_mulmod(v4_sub(gamma, delta), tw3_v, n, ninv));
        v4_store(f + 4*sub_n + j, v4_mulmod(v4_sub(alpha, beta),  tw4_v, n, ninv));

        tw1_v = v4_mulmod(tw1_v, tw1_step, n, ninv);
        tw2_v = v4_mulmod(tw2_v, tw2_step, n, ninv);
        tw3_v = v4_mulmod(tw3_v, tw3_step, n, ninv);
        tw4_v = v4_mulmod(tw4_v, tw4_step, n, ninv);
    }

    double tw1 = _mm256_cvtsd_f64(tw1_v), tw2 = _mm256_cvtsd_f64(tw2_v);
    double tw3 = _mm256_cvtsd_f64(tw3_v), tw4 = _mm256_cvtsd_f64(tw4_v);
    for (; j < sub_n; ++j) {
        double a = f[j], b = f[sub_n+j], c = f[2*sub_n+j], d = f[3*sub_n+j], e = f[4*sub_n+j];
        double s1 = b + e, t1 = b - e;
        double s2 = c + d, t2 = c - d;
        double f0 = a + s1 + s2;
        double p1 = s_mulmod(s1, Q.c1h_d, Q.p, Q.pinv);
        double p2 = s_mulmod(s2, Q.c2h_d, Q.p, Q.pinv);
        double p3 = s_mulmod(s1 + s2, Q.c12h_d, Q.p, Q.pinv);
        double pp = p1 + p2;
        double alpha = s_reduce_pm1n(a, Q.p, Q.pinv) + pp;
        double gamma = s_reduce_pm1n(a, Q.p, Q.pinv) + s_reduce_pm1n(p3 - pp, Q.p, Q.pinv);
        double q1 = s_mulmod(t1, Q.j1h_d, Q.p, Q.pinv);
        double q2 = s_mulmod(t2, Q.j2h_d, Q.p, Q.pinv);
        double q3 = s_mulmod(t1 - t2, Q.j12s_d, Q.p, Q.pinv);
        double beta = q1 + q2;
        double delta = s_reduce_pm1n(q3 - q1, Q.p, Q.pinv) + q2;
        f[j] = f0;
        f[sub_n+j] = s_mulmod(alpha + beta, tw1, Q.p, Q.pinv);
        f[2*sub_n+j] = s_mulmod(gamma + delta, tw2, Q.p, Q.pinv);
        f[3*sub_n+j] = s_mulmod(gamma - delta, tw3, Q.p, Q.pinv);
        f[4*sub_n+j] = s_mulmod(alpha - beta, tw4, Q.p, Q.pinv);
        tw1 = s_mulmod(tw1, tw_root, Q.p, Q.pinv);
        tw2 = s_mulmod(tw2, tw2_r, Q.p, Q.pinv);
        tw3 = s_mulmod(tw3, tw3_r, Q.p, Q.pinv);
        tw4 = s_mulmod(tw4, tw4_r, Q.p, Q.pinv);
    }
}

// ================================================================
// Radix-5 outer DIT pass (inverse): fused with 1/5 scale
// ================================================================
inline void radix5_dit_pass(const FftCtx& Q, double* f, std::size_t N) {
    std::size_t sub_n = N / 5;
    int k = log2_exact(sub_n);
    V4 n = v4_set1(Q.p), ninv = v4_set1(Q.pinv);
    V4 vinv5 = v4_set1(Q.inv5_d);
    V4 vc1h = v4_set1(Q.c1h_d), vc2h = v4_set1(Q.c2h_d), vc12h = v4_set1(Q.c12h_d);
    V4 vj1h = v4_set1(Q.j1h_d), vj2h = v4_set1(Q.j2h_d), vj12s = v4_set1(Q.j12s_d);

    double tw_inv_root = Q.tw5i_roots_d[k];
    double tw2i = s_mulmod(tw_inv_root, tw_inv_root, Q.p, Q.pinv);
    double tw3i = s_mulmod(tw2i, tw_inv_root, Q.p, Q.pinv);
    double tw4i = s_mulmod(tw2i, tw2i, Q.p, Q.pinv);

    V4 tw1_step, tw2_step, tw3_step, tw4_step;
    V4 tw1_v = v4_build_tw(Q.inv5_d, tw_inv_root, Q.p, Q.pinv, tw1_step);
    V4 tw2_v = v4_build_tw(Q.inv5_d, tw2i,        Q.p, Q.pinv, tw2_step);
    V4 tw3_v = v4_build_tw(Q.inv5_d, tw3i,        Q.p, Q.pinv, tw3_step);
    V4 tw4_v = v4_build_tw(Q.inv5_d, tw4i,        Q.p, Q.pinv, tw4_step);

    std::size_t j = 0;
    for (; j + 3 < sub_n; j += 4) {
        V4 fa = v4_mulmod(v4_load(f + j), vinv5, n, ninv);
        V4 fb = v4_mulmod(v4_load(f + sub_n + j),   tw1_v, n, ninv);
        V4 fc = v4_mulmod(v4_load(f + 2*sub_n + j), tw2_v, n, ninv);
        V4 fd = v4_mulmod(v4_load(f + 3*sub_n + j), tw3_v, n, ninv);
        V4 fe = v4_mulmod(v4_load(f + 4*sub_n + j), tw4_v, n, ninv);

        V4 s1 = v4_add(fb, fe), t1 = v4_sub(fb, fe);
        V4 s2 = v4_add(fc, fd), t2 = v4_sub(fc, fd);
        V4 f0 = v4_add(fa, v4_add(s1, s2));

        V4 p1 = v4_mulmod(s1, vc1h, n, ninv);
        V4 p2 = v4_mulmod(s2, vc2h, n, ninv);
        V4 p3 = v4_mulmod(v4_add(s1, s2), vc12h, n, ninv);
        V4 pp = v4_add(p1, p2);
        V4 rfa = v4_reduce_pm1n(fa, n, ninv);
        V4 alpha = v4_add(rfa, pp);
        V4 gamma = v4_add(rfa, v4_reduce_pm1n(v4_sub(p3, pp), n, ninv));

        V4 q1 = v4_mulmod(t1, vj1h, n, ninv);
        V4 q2 = v4_mulmod(t2, vj2h, n, ninv);
        V4 q3 = v4_mulmod(v4_sub(t1, t2), vj12s, n, ninv);
        V4 beta  = v4_add(q1, q2);
        V4 delta = v4_add(v4_reduce_pm1n(v4_sub(q3, q1), n, ninv), q2);

        v4_store(f + j,             v4_reduce_pm1n(f0, n, ninv));
        v4_store(f + sub_n + j,     v4_reduce_pm1n(v4_sub(alpha, beta), n, ninv));
        v4_store(f + 2*sub_n + j,   v4_reduce_pm1n(v4_sub(gamma, delta), n, ninv));
        v4_store(f + 3*sub_n + j,   v4_reduce_pm1n(v4_add(gamma, delta), n, ninv));
        v4_store(f + 4*sub_n + j,   v4_reduce_pm1n(v4_add(alpha, beta), n, ninv));

        tw1_v = v4_mulmod(tw1_v, tw1_step, n, ninv);
        tw2_v = v4_mulmod(tw2_v, tw2_step, n, ninv);
        tw3_v = v4_mulmod(tw3_v, tw3_step, n, ninv);
        tw4_v = v4_mulmod(tw4_v, tw4_step, n, ninv);
    }

    double tws1 = _mm256_cvtsd_f64(tw1_v), tws2 = _mm256_cvtsd_f64(tw2_v);
    double tws3 = _mm256_cvtsd_f64(tw3_v), tws4 = _mm256_cvtsd_f64(tw4_v);
    for (; j < sub_n; ++j) {
        double fa = s_mulmod(f[j], Q.inv5_d, Q.p, Q.pinv);
        double fb = s_mulmod(f[sub_n+j], tws1, Q.p, Q.pinv);
        double fc = s_mulmod(f[2*sub_n+j], tws2, Q.p, Q.pinv);
        double fd = s_mulmod(f[3*sub_n+j], tws3, Q.p, Q.pinv);
        double fe = s_mulmod(f[4*sub_n+j], tws4, Q.p, Q.pinv);
        double s1 = fb + fe, t1 = fb - fe;
        double s2 = fc + fd, t2 = fc - fd;
        double f0 = fa + s1 + s2;
        double p1 = s_mulmod(s1, Q.c1h_d, Q.p, Q.pinv);
        double p2 = s_mulmod(s2, Q.c2h_d, Q.p, Q.pinv);
        double p3 = s_mulmod(s1 + s2, Q.c12h_d, Q.p, Q.pinv);
        double pp = p1 + p2;
        double alpha = s_reduce_pm1n(fa, Q.p, Q.pinv) + pp;
        double gamma = s_reduce_pm1n(fa, Q.p, Q.pinv) + s_reduce_pm1n(p3 - pp, Q.p, Q.pinv);
        double q1 = s_mulmod(t1, Q.j1h_d, Q.p, Q.pinv);
        double q2 = s_mulmod(t2, Q.j2h_d, Q.p, Q.pinv);
        double q3 = s_mulmod(t1 - t2, Q.j12s_d, Q.p, Q.pinv);
        double beta = q1 + q2;
        double delta = s_reduce_pm1n(q3 - q1, Q.p, Q.pinv) + q2;
        f[j] = s_reduce_pm1n(f0, Q.p, Q.pinv);
        f[sub_n+j] = s_reduce_pm1n(alpha - beta, Q.p, Q.pinv);
        f[2*sub_n+j] = s_reduce_pm1n(gamma - delta, Q.p, Q.pinv);
        f[3*sub_n+j] = s_reduce_pm1n(gamma + delta, Q.p, Q.pinv);
        f[4*sub_n+j] = s_reduce_pm1n(alpha + beta, Q.p, Q.pinv);
        tws1 = s_mulmod(tws1, tw_inv_root, Q.p, Q.pinv);
        tws2 = s_mulmod(tws2, tw2i, Q.p, Q.pinv);
        tws3 = s_mulmod(tws3, tw3i, Q.p, Q.pinv);
        tws4 = s_mulmod(tws4, tw4i, Q.p, Q.pinv);
    }
}

// ================================================================
// Mixed-radix FFT dispatch
// ================================================================

// Find smallest NTT-friendly size >= x from {2^k, 3*2^k, 5*2^k}.
// For mixed-radix sizes, require the 2^k factor >= BLK_SZ so sub-FFTs are >= 256.
inline std::size_t ceil_ntt_size(std::size_t x) {
    if (x <= 1) return 1;
    std::size_t p2 = ceil_pow2(x);
    std::size_t best = p2;
    {
        std::size_t base = (x + 2) / 3;
        std::size_t sub = ceil_pow2(base);
        if (sub < BLK_SZ) sub = BLK_SZ;
        std::size_t s = 3 * sub;
        if (s >= x && s < best) best = s;
    }
    {
        std::size_t base = (x + 4) / 5;
        std::size_t sub = ceil_pow2(base);
        if (sub < BLK_SZ) sub = BLK_SZ;
        std::size_t s = 5 * sub;
        if (s >= x && s < best) best = s;
    }
    return best;
}

// Decompose N = m * 2^k where m in {1, 3, 5}
inline void ntt_factor(std::size_t N, std::size_t& m, int& k) {
    k = 0;
    std::size_t tmp = N;
    while ((tmp & 1) == 0) { ++k; tmp >>= 1; }
    m = tmp;
}

// Power-of-2 FFT with optional Bailey for large sizes
inline void fft_auto(FftCtx& Q, double* d, int L) {
    if (L >= BAILEY_MIN_L)
        fft_bailey(Q, d, L);
    else
        fft(Q, d, L);
}

// Power-of-2 IFFT with optional Bailey for large sizes
inline void ifft_auto(FftCtx& Q, double* d, int L) {
    if (L >= BAILEY_MIN_L)
        ifft_bailey(Q, d, L);
    else
        ifft(Q, d, L);
}

// Forward mixed-radix FFT
inline void fft_mixed(FftCtx& Q, double* d, std::size_t N) {
    std::size_t m; int k;
    ntt_factor(N, m, k);

    if (m == 1) {
        fft_auto(Q, d, k);
        return;
    }

    if (m == 3) radix3_dif_pass(Q, d, N);
    else        radix5_dif_pass(Q, d, N);

    std::size_t sub_n = std::size_t{1} << k;
    for (std::size_t i = 0; i < m; ++i)
        fft_auto(Q, d + i * sub_n, k);
}

// Inverse mixed-radix FFT
inline void ifft_mixed(FftCtx& Q, double* d, std::size_t N) {
    std::size_t m; int k;
    ntt_factor(N, m, k);

    if (m == 1) {
        ifft_auto(Q, d, k);
        return;
    }

    std::size_t sub_n = std::size_t{1} << k;
    for (std::size_t i = 0; i < m; ++i)
        ifft_auto(Q, d + i * sub_n, k);

    if (m == 3) radix3_dit_pass(Q, d, N);
    else        radix5_dit_pass(Q, d, N);
}

// Scale for mixed-radix inverse (only the 1/2^k part; 1/m is fused in DIT pass)
inline void scale_mixed(const FftCtx& Q, double* d, std::size_t len,
                            std::size_t N) {
    std::size_t m; int k;
    ntt_factor(N, m, k);

    if (m == 1) {
        u64 inv_N = inv_mod(N % Q.prime, Q.prime);
        scale(Q, d, len, static_cast<double>(inv_N));
    } else {
        u64 sub_n = std::size_t{1} << k;
        u64 inv_sub = inv_mod(sub_n % Q.prime, Q.prime);
        scale(Q, d, len, static_cast<double>(inv_sub));
    }
}

}} // namespace ntt::p50x4
