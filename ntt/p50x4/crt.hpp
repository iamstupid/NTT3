#pragma once
// crt.hpp - CRT reconstruction: 4 primes, SIMD Garner + scalar Horner
//
// Part of ntt::p50x4 - 4-prime ~50-bit NTT (double FMA Barrett)
//
// Phase 1 (FP port, AVX2):   4 coefficients x Garner chain   ->  mixed-radix digits
// Phase 2 (INT port, scalar): Horner evaluation               ->  4-limb integer
// Phase 3 (INT port, scalar): shift + add to output           ->  accumulate into z[]

#include "common.hpp"

namespace ntt { namespace p50x4 {

// ================================================================
// CRT vector primitives
// ================================================================
// v4_mulmod and v4_reduce_pm1n are provided by common.hpp / simd/v4.hpp.

static inline V4 crt_v4_to_unsigned(V4 x, V4 p)
{
    V4 mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LT_OQ);
    return _mm256_add_pd(x, _mm256_and_pd(mask, p));
}

static inline __m256i crt_v4_double_to_u64(V4 x)
{
    const V4 magic = _mm256_set1_pd(4503599627370496.0); // 2^52
    V4 biased = _mm256_add_pd(
        _mm256_round_pd(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
        magic);
    return _mm256_sub_epi64(
        _mm256_castpd_si256(biased),
        _mm256_castpd_si256(magic));
}

// ================================================================
// CRT context with precomputed constants
// ================================================================
struct CrtCtx {
    u64 p[4];

    // Phase 1: per-prime broadcast vectors
    __m256d vp[4];
    __m256d vpinv[4];

    // Phase 1: Garner inverses c_ij = p_i^{-1} mod p_j, signed double
    __m256d vc01, vc02, vc03;
    __m256d vc12, vc13;
    __m256d vc23;

    void init() {
        p[0] = PRIMES[0];
        p[1] = PRIMES[1];
        p[2] = PRIMES[2];
        p[3] = PRIMES[3];

        for (int i = 0; i < 4; i++) {
            double pd = static_cast<double>(p[i]);
            vp[i]    = _mm256_set1_pd(pd);
            vpinv[i] = _mm256_set1_pd(1.0 / pd);
        }

        double pd[4];
        for (int i = 0; i < 4; i++) pd[i] = static_cast<double>(p[i]);

        auto signed_dbl = [](u64 x, double pp) -> double {
            double v = static_cast<double>(x);
            return (v > pp * 0.5) ? (v - pp) : v;
        };

        auto modinv = [](u64 a, u64 m) -> u64 {
            std::int64_t g0 = static_cast<std::int64_t>(m);
            std::int64_t g1 = static_cast<std::int64_t>(a % m);
            std::int64_t u0 = 0, u1 = 1, tmp;
            while (g1) {
                std::int64_t q = g0 / g1;
                tmp = g0 - q * g1; g0 = g1; g1 = tmp;
                tmp = u0 - q * u1; u0 = u1; u1 = tmp;
            }
            return static_cast<u64>((u0 % static_cast<std::int64_t>(m) +
                                     static_cast<std::int64_t>(m)) %
                                    static_cast<std::int64_t>(m));
        };

        vc01 = _mm256_set1_pd(signed_dbl(modinv(p[0], p[1]), pd[1]));
        vc02 = _mm256_set1_pd(signed_dbl(modinv(p[0], p[2]), pd[2]));
        vc03 = _mm256_set1_pd(signed_dbl(modinv(p[0], p[3]), pd[3]));
        vc12 = _mm256_set1_pd(signed_dbl(modinv(p[1], p[2]), pd[2]));
        vc13 = _mm256_set1_pd(signed_dbl(modinv(p[1], p[3]), pd[3]));
        vc23 = _mm256_set1_pd(signed_dbl(modinv(p[2], p[3]), pd[3]));
    }
};

// ================================================================
// Phase 1: SIMD Garner (FP port, 4 coefficients parallel)
// ================================================================

// reduce + round + to_unsigned: snap to exact integer in [0, p)
#define GARNER_UINT(x, p, pinv) \
    crt_v4_to_unsigned( \
        _mm256_round_pd( \
            v4_reduce_pm1n((x), (p), (pinv)), \
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), \
        (p))

static inline void garner_phase1(
    const CrtCtx* C,
    const double* d0, const double* d1,
    const double* d2, const double* d3,
    std::size_t idx,
    u64 v0[4], u64 v1[4],
    u64 v2[4], u64 v3[4])
{
    __m256d r0 = _mm256_loadu_pd(d0 + idx);
    __m256d r1 = _mm256_loadu_pd(d1 + idx);
    __m256d r2 = _mm256_loadu_pd(d2 + idx);
    __m256d r3 = _mm256_loadu_pd(d3 + idx);

    __m256d fv0 = GARNER_UINT(r0, C->vp[0], C->vpinv[0]);

    __m256d fv1 = GARNER_UINT(
        v4_mulmod(_mm256_sub_pd(r1, fv0), C->vc01, C->vp[1], C->vpinv[1]),
        C->vp[1], C->vpinv[1]);

    __m256d t = GARNER_UINT(
        v4_mulmod(_mm256_sub_pd(r2, fv0), C->vc02, C->vp[2], C->vpinv[2]),
        C->vp[2], C->vpinv[2]);
    __m256d fv2 = GARNER_UINT(
        v4_mulmod(_mm256_sub_pd(t, fv1), C->vc12, C->vp[2], C->vpinv[2]),
        C->vp[2], C->vpinv[2]);

    t = GARNER_UINT(
        v4_mulmod(_mm256_sub_pd(r3, fv0), C->vc03, C->vp[3], C->vpinv[3]),
        C->vp[3], C->vpinv[3]);
    t = GARNER_UINT(
        v4_mulmod(_mm256_sub_pd(t, fv1), C->vc13, C->vp[3], C->vpinv[3]),
        C->vp[3], C->vpinv[3]);
    __m256d fv3 = GARNER_UINT(
        v4_mulmod(_mm256_sub_pd(t, fv2), C->vc23, C->vp[3], C->vpinv[3]),
        C->vp[3], C->vpinv[3]);

    _mm256_storeu_si256((__m256i*)v0, crt_v4_double_to_u64(fv0));
    _mm256_storeu_si256((__m256i*)v1, crt_v4_double_to_u64(fv1));
    _mm256_storeu_si256((__m256i*)v2, crt_v4_double_to_u64(fv2));
    _mm256_storeu_si256((__m256i*)v3, crt_v4_double_to_u64(fv3));
}

#undef GARNER_UINT

// ================================================================
// Phase 2: Scalar Horner (INT port, per coefficient)
// ================================================================
static inline void horner_phase2(
    u64 a0, u64 a1, u64 a2, u64 a3,
    u64 p0, u64 p1, u64 p2,
    u64 out[4])
{
#if defined(__SIZEOF_INT128__)
    __uint128_t w;
    u64 x0, x1, x2, c;

    w = (__uint128_t)a3 * p2 + a2;
    x0 = (u64)w; x1 = (u64)(w >> 64);

    w = (__uint128_t)x0 * p1 + a1;
    x0 = (u64)w; c = (u64)(w >> 64);
    w = (__uint128_t)x1 * p1 + c;
    x1 = (u64)w; x2 = (u64)(w >> 64);

    w = (__uint128_t)x0 * p0 + a0;
    out[0] = (u64)w; c = (u64)(w >> 64);
    w = (__uint128_t)x1 * p0 + c;
    out[1] = (u64)w; c = (u64)(w >> 64);
    w = (__uint128_t)x2 * p0 + c;
    out[2] = (u64)w; out[3] = (u64)(w >> 64);
#else
    u64 hi, lo, c;
    unsigned char cf;

    lo = _umul128(a3, p2, &hi);
    cf = _addcarry_u64(0, lo, a2, &lo);
    _addcarry_u64(cf, hi, 0, &hi);
    u64 x0 = lo, x1 = hi;

    lo = _umul128(x0, p1, &hi);
    cf = _addcarry_u64(0, lo, a1, &lo);
    _addcarry_u64(cf, hi, 0, &hi);
    x0 = lo; c = hi;

    lo = _umul128(x1, p1, &hi);
    cf = _addcarry_u64(0, lo, c, &lo);
    _addcarry_u64(cf, hi, 0, &hi);
    x1 = lo;
    u64 x2 = hi;

    lo = _umul128(x0, p0, &hi);
    cf = _addcarry_u64(0, lo, a0, &lo);
    _addcarry_u64(cf, hi, 0, &hi);
    out[0] = lo; c = hi;

    lo = _umul128(x1, p0, &hi);
    cf = _addcarry_u64(0, lo, c, &lo);
    _addcarry_u64(cf, hi, 0, &hi);
    out[1] = lo; c = hi;

    lo = _umul128(x2, p0, &hi);
    cf = _addcarry_u64(0, lo, c, &lo);
    _addcarry_u64(cf, hi, 0, &hi);
    out[2] = lo;
    out[3] = hi;
#endif
}

// ================================================================
// Phase 3: Shift + accumulate to output
// ================================================================

static inline void accum_shifted(
    u64* buf, unsigned off, const u64 x[4], unsigned shift)
{
    unsigned char cf;

    if (shift == 0) {
        cf = 0;
        cf = _addcarry_u64(cf, buf[off+0], x[0], &buf[off+0]);
        cf = _addcarry_u64(cf, buf[off+1], x[1], &buf[off+1]);
        cf = _addcarry_u64(cf, buf[off+2], x[2], &buf[off+2]);
        cf = _addcarry_u64(cf, buf[off+3], x[3], &buf[off+3]);
        buf[off+4] += cf;
    } else {
        unsigned rs = 64 - shift;
        u64 s0 =  x[0] << shift;
        u64 s1 = (x[1] << shift) | (x[0] >> rs);
        u64 s2 = (x[2] << shift) | (x[1] >> rs);
        u64 s3 = (x[3] << shift) | (x[2] >> rs);
        u64 s4 =                    x[3] >> rs;

        cf = 0;
        cf = _addcarry_u64(cf, buf[off+0], s0, &buf[off+0]);
        cf = _addcarry_u64(cf, buf[off+1], s1, &buf[off+1]);
        cf = _addcarry_u64(cf, buf[off+2], s2, &buf[off+2]);
        cf = _addcarry_u64(cf, buf[off+3], s3, &buf[off+3]);
        cf = _addcarry_u64(cf, buf[off+4], s4, &buf[off+4]);
        buf[off+5] += cf;
    }
}

static inline void flush_to_output(
    u64* z, std::size_t zn,
    const u64* buf, std::size_t len,
    std::size_t base)
{
    unsigned char cf = 0;
    std::size_t i;
    for (i = 0; i < len && base + i < zn; i++)
        cf = _addcarry_u64(cf, z[base + i], buf[i], &z[base + i]);
    for (; cf && base + i < zn; i++)
        cf = _addcarry_u64(cf, z[base + i], 0, &z[base + i]);
}

// ================================================================
// Main CRT loop
// ================================================================
static void crt_reconstruct(
    const CrtCtx* C,
    u64* z, std::size_t zn,
    const double* d0, const double* d1,
    const double* d2, const double* d3,
    std::size_t ncoeffs)
{
    u64 p0 = C->p[0], p1 = C->p[1], p2 = C->p[2];
    std::size_t ngroups = ncoeffs / 4;

    std::memset(z, 0, zn * sizeof(u64));

    for (std::size_t g = 0; g < ngroups; g++)
    {
        u64 a0[4], a1[4], a2[4], a3[4];
        garner_phase1(C, d0, d1, d2, d3, g * 4, a0, a1, a2, a3);

        u64 buf[9] = {0};

        for (int j = 0; j < 4; j++) {
            u64 x[4];
            horner_phase2(a0[j], a1[j], a2[j], a3[j], p0, p1, p2, x);
            accum_shifted(buf, static_cast<unsigned>(j), x, static_cast<unsigned>(j * 16));
        }

        flush_to_output(z, zn, buf, 9, g * 5);
    }

    // Tail: remaining 0-3 coefficients
    std::size_t rem = ncoeffs - ngroups * 4;
    if (rem > 0)
    {
        std::size_t base = ngroups * 4;

        double r0[4]={0}, r1[4]={0}, r2[4]={0}, r3[4]={0};
        for (std::size_t j = 0; j < rem; j++) {
            r0[j] = d0[base + j];
            r1[j] = d1[base + j];
            r2[j] = d2[base + j];
            r3[j] = d3[base + j];
        }

        u64 a0[4], a1[4], a2[4], a3[4];
        garner_phase1(C, r0, r1, r2, r3, 0, a0, a1, a2, a3);

        for (std::size_t j = 0; j < rem; j++) {
            u64 x[4];
            horner_phase2(a0[j], a1[j], a2[j], a3[j], p0, p1, p2, x);

            std::size_t bit_off = (base + j) * 80;
            std::size_t loff    = bit_off / 64;
            unsigned shift = static_cast<unsigned>(bit_off % 64);

            if (shift == 0) {
                unsigned char cf = 0;
                for (int k = 0; k < 4 && loff + static_cast<std::size_t>(k) < zn; k++)
                    cf = _addcarry_u64(cf, z[loff+k], x[k], &z[loff+k]);
                for (std::size_t k = 4; cf && loff + k < zn; k++)
                    cf = _addcarry_u64(cf, z[loff+k], 0, &z[loff+k]);
            } else {
                unsigned rs = 64 - shift;
                u64 s[5] = {
                    x[0] << shift,
                    (x[1] << shift) | (x[0] >> rs),
                    (x[2] << shift) | (x[1] >> rs),
                    (x[3] << shift) | (x[2] >> rs),
                    x[3] >> rs
                };
                unsigned char cf = 0;
                for (int k = 0; k < 5 && loff + static_cast<std::size_t>(k) < zn; k++)
                    cf = _addcarry_u64(cf, z[loff+k], s[k], &z[loff+k]);
                for (std::size_t k = 5; cf && loff + k < zn; k++)
                    cf = _addcarry_u64(cf, z[loff+k], 0, &z[loff+k]);
            }
        }
    }
}

}} // namespace ntt::p50x4
