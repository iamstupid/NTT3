#pragma once
// fft_ctx.hpp - Per-prime FFT context with split w2tab twiddle tables
//
// Part of ntt::p50x4 - 4-prime ~50-bit NTT (double FMA Barrett)

#include "common.hpp"

namespace ntt { namespace p50x4 {

struct FftCtx {
    double p = 0.0;
    double pinv = 0.0;
    u64 prime = 0;
    u64 prim_root = 0;
    unsigned int w2tab_depth = 0;
    double* w2tab[W2TAB_SIZE] = {};

    // Radix-3 constants (FP, balanced representation in [-p/2, p/2])
    double neg_half_d = 0.0;     // -1/2 mod p
    double j3_half_d = 0.0;     // (omega_3 - omega_3^2)/2 mod p
    double inv3_d = 0.0;        // 1/3 mod p

    // Radix-5 constants
    double inv5_d = 0.0;        // 1/5 mod p
    double c1h_d = 0.0;         // (omega_5 + omega_5^4)/2 mod p
    double c2h_d = 0.0;         // (omega_5^2 + omega_5^3)/2 mod p
    double j1h_d = 0.0;         // (omega_5 - omega_5^4)/2 mod p
    double j2h_d = 0.0;         // (omega_5^2 - omega_5^3)/2 mod p
    double c12h_d = 0.0;        // c1h + c2h mod p
    double j12s_d = 0.0;        // j1h + j2h mod p

    // Outer-pass twiddle roots: tw3_roots_d[k] = omega_{3*2^k}, etc.
    static constexpr int MAX_TW = 42;
    double tw3_roots_d[MAX_TW] = {};
    double tw3i_roots_d[MAX_TW] = {};
    double tw5_roots_d[MAX_TW] = {};
    double tw5i_roots_d[MAX_TW] = {};

    void init(u64 pp) {
        prime = pp;
        p = static_cast<double>(pp);
        pinv = 1.0 / p;
        prim_root = primitive_root(pp);

        // Allocate initial tables (consecutive storage for first W2TAB_INIT tables)
        std::size_t N = pow2(W2TAB_INIT - 1);
        double* t = alloc_doubles(N);
        w2tab[0] = t;
        t[0] = 1.0;

        std::size_t l = 1;
        for (unsigned int k = 1; k < W2TAB_INIT; ++k, l *= 2) {
            u64 ww = pow_mod(prim_root, (prime - 1) >> (k + 1), prime);
            double w = s_reduce_0n_to_pmhn(static_cast<double>(ww), p);
            double* curr = t + l;
            w2tab[k] = curr;
            for (std::size_t i = 0; i < l; ++i) {
                double x = t[i];
                x = s_mulmod(x, w, p, pinv);
                x = s_reduce_pm1n_to_pmhn(x, p);
                curr[i] = x;
            }
        }
        w2tab_depth = W2TAB_INIT;
        for (unsigned int k = W2TAB_INIT; k < W2TAB_SIZE; ++k)
            w2tab[k] = nullptr;

        // Radix-3/5 constants
        u64 half_u = inv_mod(2, pp);
        u64 neg_half_u = pp - half_u;
        neg_half_d = s_reduce_0n_to_pmhn(static_cast<double>(neg_half_u), p);

        inv3_d = s_reduce_0n_to_pmhn(static_cast<double>(inv_mod(3, pp)), p);
        inv5_d = s_reduce_0n_to_pmhn(static_cast<double>(inv_mod(5, pp)), p);

        // omega_3 and derived
        u64 w3 = pow_mod(prim_root, (pp - 1) / 3, pp);
        u64 w3sq = mul_mod_u64(w3, w3, pp);
        u64 j3_u = sub_mod_u64(w3, w3sq, pp);
        u64 j3h_u = mul_mod_u64(j3_u, half_u, pp);
        j3_half_d = s_reduce_0n_to_pmhn(static_cast<double>(j3h_u), p);

        // omega_5 and derived
        u64 w5_1 = pow_mod(prim_root, (pp - 1) / 5, pp);
        u64 w5_2 = mul_mod_u64(w5_1, w5_1, pp);
        u64 w5_3 = mul_mod_u64(w5_2, w5_1, pp);
        u64 w5_4 = mul_mod_u64(w5_3, w5_1, pp);

        u64 s14 = add_mod_u64(w5_1, w5_4, pp);
        u64 s23 = add_mod_u64(w5_2, w5_3, pp);
        u64 d14 = sub_mod_u64(w5_1, w5_4, pp);
        u64 d23 = sub_mod_u64(w5_2, w5_3, pp);

        c1h_d = s_reduce_0n_to_pmhn(static_cast<double>(mul_mod_u64(s14, half_u, pp)), p);
        c2h_d = s_reduce_0n_to_pmhn(static_cast<double>(mul_mod_u64(s23, half_u, pp)), p);
        j1h_d = s_reduce_0n_to_pmhn(static_cast<double>(mul_mod_u64(d14, half_u, pp)), p);
        j2h_d = s_reduce_0n_to_pmhn(static_cast<double>(mul_mod_u64(d23, half_u, pp)), p);

        u64 c12h_u = add_mod_u64(mul_mod_u64(s14, half_u, pp), mul_mod_u64(s23, half_u, pp), pp);
        u64 j12s_u = add_mod_u64(mul_mod_u64(d14, half_u, pp), mul_mod_u64(d23, half_u, pp), pp);
        c12h_d = s_reduce_0n_to_pmhn(static_cast<double>(c12h_u), p);
        j12s_d = s_reduce_0n_to_pmhn(static_cast<double>(j12s_u), p);

        // Twiddle roots for outer radix-3/5 passes
        int max_k2 = 0;
        { u64 tmp = pp - 1; while ((tmp & 1) == 0) { ++max_k2; tmp >>= 1; } }

        for (int k = 0; k <= max_k2 && k < MAX_TW; ++k) {
            u64 exp3 = (pp - 1) / (3ULL * (1ULL << k));
            tw3_roots_d[k] = s_reduce_0n_to_pmhn(
                static_cast<double>(pow_mod(prim_root, exp3, pp)), p);
            tw3i_roots_d[k] = s_reduce_0n_to_pmhn(
                static_cast<double>(pow_mod(prim_root, pp - 1 - exp3, pp)), p);

            u64 exp5 = (pp - 1) / (5ULL * (1ULL << k));
            tw5_roots_d[k] = s_reduce_0n_to_pmhn(
                static_cast<double>(pow_mod(prim_root, exp5, pp)), p);
            tw5i_roots_d[k] = s_reduce_0n_to_pmhn(
                static_cast<double>(pow_mod(prim_root, pp - 1 - exp5, pp)), p);
        }
    }

    void fit_depth(unsigned int depth) {
        while (w2tab_depth < depth) {
            unsigned int k = w2tab_depth;
            u64 ww = pow_mod(prim_root, (prime - 1) >> (k + 1), prime);
            double w = s_reduce_0n_to_pmhn(static_cast<double>(ww), p);

            std::size_t N = pow2(k - 1);
            double* curr = alloc_doubles(N);
            w2tab[k] = curr;

            double* src = w2tab[0];
            std::size_t off = 0;
            std::size_t slen = pow2(W2TAB_INIT - 1);

            for (unsigned int j = W2TAB_INIT - 1; j < k; ++j) {
                std::size_t tlen = (j == static_cast<unsigned>(W2TAB_INIT - 1))
                                   ? slen : pow2(j - 1);
                for (std::size_t i = 0; i < tlen; ++i) {
                    double x = src[i];
                    x = s_mulmod(x, w, p, pinv);
                    x = s_reduce_pm1n_to_pmhn(x, p);
                    curr[off + i] = x;
                }
                off += tlen;
                src = w2tab[j + 1];
            }
            w2tab_depth = k + 1;
        }
    }

    // Reusable Bailey workspace (avoids repeated alloc+memset)
    double* bailey_tmp = nullptr;
    std::size_t bailey_tmp_cap = 0;

    double* ensure_bailey_tmp(std::size_t n) {
        if (bailey_tmp_cap >= n) return bailey_tmp;
        if (bailey_tmp) free_doubles(bailey_tmp);
        bailey_tmp = alloc_doubles(n);
        bailey_tmp_cap = n;
        return bailey_tmp;
    }

    void clear() {
        if (bailey_tmp) { free_doubles(bailey_tmp); bailey_tmp = nullptr; bailey_tmp_cap = 0; }
        if (w2tab[0]) { free_doubles(w2tab[0]); w2tab[0] = nullptr; }
        for (int k = W2TAB_INIT; k < W2TAB_SIZE; ++k)
            if (w2tab[k]) { free_doubles(w2tab[k]); w2tab[k] = nullptr; }
    }

    // Forward twiddle lookup: w[2*j]
    double w2_fwd(std::size_t j) const {
        if (j == 0) return w2tab[0][0];
        int jb = nbits_nz(j);
        return w2tab[jb][j - pow2(jb - 1)];
    }

    // Inverse twiddle lookup: uses mirrored index
    double w2_inv(std::size_t j) const {
        if (j == 0) return w2tab[0][0];
        int jb = nbits_nz(j);
        std::size_t jmr = pow2(jb) - 1 - j;
        return w2tab[jb][jmr];
    }
};

}} // namespace ntt::p50x4
