#pragma once
// multiply.hpp - Ntt4 class, 80-bit coefficient extraction, top-level multiply
//
// Part of ntt::p50x4 - 4-prime ~50-bit NTT (double FMA Barrett)

#include "mixed_radix.hpp"
#include "crt.hpp"

namespace ntt { namespace p50x4 {

// ================================================================
// 80-bit coefficient conversion utilities
// ================================================================

static constexpr double MAGIC = 6755399441055744.0; // 3 * 2^51
static constexpr long long MAGIC_BITS = 0x4338000000000000LL;

inline u64 double_to_u64_mod(double x, double p) {
    double tmp = x + MAGIC;
    long long r;
    std::memcpy(&r, &tmp, 8);
    r -= MAGIC_BITS;
    long long pp = static_cast<long long>(p);
    if (r < 0) r += pp;
    if (r >= pp) r -= pp;
    return static_cast<u64>(r);
}

// Trunc index: compensate for 4x4 transpose in basecases
inline std::size_t trunc_index(int L, std::size_t i) {
    if (L >= 4)
        i = (i & ~std::size_t{15}) | ((i >> 2) & 3) | ((i & 3) << 2);
    return i;
}

// Number of 80-bit coefficients from n_limbs u64 words
inline std::size_t n_coeffs_80(std::size_t n_limbs) {
    return (n_limbs * 4 + 4) / 5;
}

// SIMD extraction: 5 limbs -> 4 x (lo48, hi32) as __m256d.
inline void extract_4x80_simd(const u64* a, std::size_t base, V4& lo48, V4& hi32) {
    const __m256i kOffsets    = _mm256_set_epi64x(48, 32, 16, 0);
    const __m256i kInvOffsets = _mm256_set_epi64x(16, 32, 48, 64);
    const __m256i kMask48i    = _mm256_set1_epi64x(0xFFFFFFFFFFFFULL);
    const __m256i kMask32i    = _mm256_set1_epi64x(0xFFFFFFFFULL);
    const __m256i MAGIC_I     = _mm256_set1_epi64x(0x4330000000000000ULL);
    const V4      MAGIC_D     = _mm256_set1_pd(4503599627370496.0);

    __m256i lo_limbs = _mm256_loadu_si256((const __m256i*)(a + base));
    __m256i hi_limbs = _mm256_loadu_si256((const __m256i*)(a + base + 1));

    __m256i part1   = _mm256_srlv_epi64(lo_limbs, kOffsets);
    __m256i part2   = _mm256_sllv_epi64(hi_limbs, kInvOffsets);
    __m256i comb_lo = _mm256_or_si256(part1, part2);
    __m256i comb_hi = _mm256_srlv_epi64(hi_limbs, kOffsets);

    __m256i lo_i = _mm256_and_si256(comb_lo, kMask48i);
    __m256i hi_i = _mm256_and_si256(
        _mm256_or_si256(
            _mm256_srli_epi64(comb_lo, 48),
            _mm256_slli_epi64(comb_hi, 16)
        ),
        kMask32i);

    lo48 = _mm256_sub_pd(
        _mm256_castsi256_pd(_mm256_or_si256(lo_i, MAGIC_I)), MAGIC_D);
    hi32 = _mm256_sub_pd(
        _mm256_castsi256_pd(_mm256_or_si256(hi_i, MAGIC_I)), MAGIC_D);
}

// SIMD modular reduction: 4 coefficients x 1 prime.
__forceinline V4 reduce_4x1p(V4 lo, V4 hi, V4 p, V4 pinv, V4 t48) {
    V4 h = v4_mul(hi, t48);
    V4 q = v4_round(v4_mul(h, pinv));
    V4 l = _mm256_fmsub_pd(hi, t48, h);
    V4 t = v4_add(_mm256_fnmadd_pd(q, p, h), l);

    V4 r = v4_add(t, lo);
    V4 q2 = v4_round(v4_mul(r, pinv));
    return _mm256_fnmadd_pd(q2, p, r);
}

// Single-prime conversion.
inline void convert_80bit_to_double(double* out, const u64* limbs, std::size_t n_limbs,
                                     const FftCtx& Q) {
    std::size_t n_coeffs = n_coeffs_80(n_limbs);
    if (n_coeffs == 0) return;

    double T48 = s_reduce_0n_to_pmhn(
        static_cast<double>(pow_mod(2, 48, Q.prime)), Q.p);
    V4 vp = v4_set1(Q.p), vpinv = v4_set1(Q.pinv), vt48 = v4_set1(T48);

    std::size_t ngroups = n_limbs / 5;
    for (std::size_t g = 0; g < ngroups; g++) {
        V4 lo, hi;
        extract_4x80_simd(limbs, g * 5, lo, hi);
        v4_store(out + g * 4, reduce_4x1p(lo, hi, vp, vpinv, vt48));
    }

    std::size_t tail_start = ngroups * 4;
    if (tail_start < n_coeffs) {
        u64 pad[5] = {};
        std::size_t rem = n_limbs - ngroups * 5;
        for (std::size_t i = 0; i < rem; i++)
            pad[i] = limbs[ngroups * 5 + i];

        V4 lo, hi;
        extract_4x80_simd(pad, 0, lo, hi);
        V4 result = reduce_4x1p(lo, hi, vp, vpinv, vt48);

        double tmp[4];
        _mm256_storeu_pd(tmp, result);
        std::size_t tail_count = n_coeffs - tail_start;
        for (std::size_t i = 0; i < tail_count; i++)
            out[tail_start + i] = tmp[i];
    }
}

// Multi-prime conversion: extract once, reduce for all 4 primes.
inline void convert_80bit_all_primes(double* out[4], const u64* limbs, std::size_t n_limbs,
                                      const FftCtx ctx[4]) {
    std::size_t n_coeffs = n_coeffs_80(n_limbs);
    if (n_coeffs == 0) return;

    V4 vp[4], vpinv[4], vt48[4];
    for (int i = 0; i < 4; i++) {
        vp[i]   = v4_set1(ctx[i].p);
        vpinv[i] = v4_set1(ctx[i].pinv);
        double T48 = s_reduce_0n_to_pmhn(
            static_cast<double>(pow_mod(2, 48, ctx[i].prime)), ctx[i].p);
        vt48[i] = v4_set1(T48);
    }

    std::size_t ngroups = n_limbs / 5;
    for (std::size_t g = 0; g < ngroups; g++) {
        V4 lo, hi;
        extract_4x80_simd(limbs, g * 5, lo, hi);
        for (int pi = 0; pi < 4; pi++)
            v4_store(out[pi] + g * 4, reduce_4x1p(lo, hi, vp[pi], vpinv[pi], vt48[pi]));
    }

    std::size_t tail_start = ngroups * 4;
    if (tail_start < n_coeffs) {
        u64 pad[5] = {};
        std::size_t rem = n_limbs - ngroups * 5;
        for (std::size_t i = 0; i < rem; i++)
            pad[i] = limbs[ngroups * 5 + i];

        V4 lo, hi;
        extract_4x80_simd(pad, 0, lo, hi);
        std::size_t tail_count = n_coeffs - tail_start;
        for (int pi = 0; pi < 4; pi++) {
            V4 result = reduce_4x1p(lo, hi, vp[pi], vpinv[pi], vt48[pi]);
            double tmp[4];
            _mm256_storeu_pd(tmp, result);
            for (std::size_t j = 0; j < tail_count; j++)
                out[pi][tail_start + j] = tmp[j];
        }
    }
}

// ================================================================
// Ntt4: 4-prime NTT multiply engine
// ================================================================
class Ntt4 {
public:
    Ntt4() {
        for (int i = 0; i < 4; ++i)
            ctx_[i].init(PRIMES[i]);
        crt_.init();
    }

    ~Ntt4() {
        for (int i = 0; i < 4; ++i)
            ctx_[i].clear();
    }

    // Thread-local singleton (lazy init on first use)
    static Ntt4& instance() {
        static thread_local Ntt4 inst;
        return inst;
    }

    // Big-integer multiply: a[0..na) * b[0..nb) -> out[0..out_len)
    // Uses 80-bit coefficient extraction + 4-prime NTT + SIMD Garner CRT.
    void multiply(u64* out, std::size_t out_len,
                  const u64* a, std::size_t na,
                  const u64* b, std::size_t nb) {
        if (na == 0 || nb == 0) {
            std::memset(out, 0, out_len * sizeof(u64));
            return;
        }

        std::size_t nca = n_coeffs_80(na);
        std::size_t ncb = n_coeffs_80(nb);
        std::size_t conv_len = nca + ncb - 1;
        std::size_t N = ceil_ntt_size(conv_len);
        if (N < BLK_SZ) N = BLK_SZ;

        double* fa[4];
        for (int i = 0; i < 4; i++)
            fa[i] = alloc_doubles(N);

        convert_80bit_all_primes(fa, a, na, ctx_);

        // Single shared fb buffer, reused per prime
        double* fb = alloc_doubles(N);
        std::size_t nb_coeffs = n_coeffs_80(nb);

        for (int pi = 0; pi < 4; ++pi) {
            auto& Q = ctx_[pi];

            convert_80bit_to_double(fb, b, nb, Q);
            std::memset(fb + nb_coeffs, 0, (N - nb_coeffs) * sizeof(double));
            fft_mixed(Q, fa[pi], N);
            fft_mixed(Q, fb, N);
            point_mul(Q, fa[pi], fb, N);
            ifft_mixed(Q, fa[pi], N);

            scale_mixed(Q, fa[pi], conv_len, N);
        }
        free_doubles(fb);

        // CRT reconstruct
        std::size_t zn = (80 * conv_len + 256 + 63) / 64;
        // Use a temporary buffer for CRT output, then copy to out
        std::vector<u64> z(zn, 0);
        crt_reconstruct(&crt_, z.data(), z.size(),
                        fa[0], fa[1], fa[2], fa[3], conv_len);

        for (int i = 0; i < 4; i++) free_doubles(fa[i]);

        // Copy to output, trimming to actual product size
        std::size_t product_len = na + nb;
        std::size_t copy_len = (std::min)({product_len, out_len, z.size()});
        std::memcpy(out, z.data(), copy_len * sizeof(u64));
        if (copy_len < out_len)
            std::memset(out + copy_len, 0, (out_len - copy_len) * sizeof(u64));
    }

    const FftCtx* contexts() const { return ctx_; }
    const CrtCtx* crt() const { return &crt_; }

private:
    FftCtx ctx_[4];
    CrtCtx crt_;
};

}} // namespace ntt::p50x4
