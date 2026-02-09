#pragma once
#include "../common.hpp"
#include "mont_scalar.hpp"
#include "mont_vec.hpp"
#include "../simd/avx2.hpp"
#include <array>

namespace ntt {

// Cyclic convolution at SIMD width (8-point for AVX2).
// Direct port of __conv8, __conv8_4, __vec_cvdt8 from ref.cpp
template<typename B>
struct CyclicConv;

template<>
struct CyclicConv<Avx2> {
    using Vec = __m256i;

    // Full twisted convolution over n Vecs.
    // Port of __vec_cvdt8 from ref.cpp.
    // Iterates with ruler-sequence twiddle updates, calling conv8_batch4 in groups of 4.
    static void twisted_conv(Vec* f, Vec* g, idt n,
                              const MontScalar& ms, u32 img, const u32* RT3,
                              u32 rr_init = 0)
    {
        u32 RR = rr_init ? rr_init : ms.one;
        const u32 mod2_ = ms.mod2;
        const Vec vNiv = _mm256_set1_epi32((i32)ms.niv);
        const Vec vMod = _mm256_set1_epi32((i32)ms.mod);
        const Vec vMod2 = _mm256_set1_epi32((i32)mod2_);

        for (idt i = 0; i < n; i += 4) {
            const u32 RRi = ms.mul(RR, img);
            conv8_batch4(
                (u32*)(f + i), (u32*)(g + i),
                {RR, mod2_ - RR, RRi, mod2_ - RRi},
                vNiv, vMod, vMod2);
            RR = ms.mul(RR, RT3[ntt_ctzll(i + 4)]);
        }
    }

    // Single 8-point twisted convolution
    // f = fx * f * g mod (x^8 - ww)
    // Direct port of __conv8
    NTT_FORCEINLINE static void conv8(
        Vec* f, const Vec* g,
        Vec ww, Vec fx,
        Vec Niv, Vec Mod, Vec Mod2)
    {
        const Vec raa = _mm256_load_si256(f);
        const Vec rbb = _mm256_load_si256(g);

        Vec taa = _mm256_min_epu32(raa, _mm256_sub_epi32(raa, Mod2));
        Vec bb = shrk(mul_bsm(rbb, fx, Niv, Mod), Mod);
        Vec aw = shrk(mul_bsm(taa, ww, Niv, Mod), Mod);
        Vec aa = shrk(taa, Mod);
        Vec awa = _mm256_permute2x128_si256(aa, aw, 3);

        Vec b0 = _mm256_permute4x64_epi64(bb, 0x00);
        Vec b1 = _mm256_shuffle_epi32(b0, 0xb1);
        Vec aw7 = _mm256_alignr_epi8(aa, awa, 12);

        Vec res00 = _mm256_mul_epu32(aa, b0);
        Vec res01 = _mm256_mul_epu32(_mm256_srli_epi64(aa, 32), b0);
        Vec res10 = _mm256_mul_epu32(aw7, b1);
        Vec res11 = _mm256_mul_epu32(aa, b1);

        Vec b2 = _mm256_permute4x64_epi64(bb, 0x55);
        Vec b3 = _mm256_shuffle_epi32(b2, 0xb1);
        Vec aw6 = _mm256_alignr_epi8(aa, awa, 8);
        Vec aw5 = _mm256_alignr_epi8(aa, awa, 4);

        res00 = _mm256_add_epi64(res00, _mm256_mul_epu32(aw6, b2));
        res01 = _mm256_add_epi64(res01, _mm256_mul_epu32(aw7, b2));
        res10 = _mm256_add_epi64(res10, _mm256_mul_epu32(aw5, b3));
        res11 = _mm256_add_epi64(res11, _mm256_mul_epu32(aw6, b3));

        Vec b4 = _mm256_permute4x64_epi64(bb, 0xaa);
        Vec b5 = _mm256_shuffle_epi32(b4, 0xb1);
        Vec aw3 = _mm256_alignr_epi8(awa, aw, 12);

        res00 = _mm256_add_epi64(res00, _mm256_mul_epu32(awa, b4));
        res01 = _mm256_add_epi64(res01, _mm256_mul_epu32(aw5, b4));
        res10 = _mm256_add_epi64(res10, _mm256_mul_epu32(aw3, b5));
        res11 = _mm256_add_epi64(res11, _mm256_mul_epu32(awa, b5));

        Vec b6 = _mm256_permute4x64_epi64(bb, 0xff);
        Vec b7 = _mm256_shuffle_epi32(b6, 0xb1);
        Vec aw2 = _mm256_alignr_epi8(awa, aw, 8);
        Vec aw1 = _mm256_alignr_epi8(awa, aw, 4);

        res00 = _mm256_add_epi64(res00, _mm256_mul_epu32(aw2, b6));
        res01 = _mm256_add_epi64(res01, _mm256_mul_epu32(aw3, b6));
        res10 = _mm256_add_epi64(res10, _mm256_mul_epu32(aw1, b7));
        res11 = _mm256_add_epi64(res11, _mm256_mul_epu32(aw2, b7));

        res00 = _mm256_add_epi64(res00, res10);
        res01 = _mm256_add_epi64(res01, res11);

        Vec reduced = mont_reduce_raw(res00, res01, Niv, Mod);
        _mm256_store_si256(f, shrk(reduced, Mod2));
    }

    // Batch of 4 twisted convolutions
    // Direct port of __conv8_4
    static void conv8_batch4(
        u32* NTT_RESTRICT f, u32* NTT_RESTRICT g,
        std::array<u32, 4> ww,
        Vec Niv, Vec Mod, Vec Mod2)
    {
        alignas(64) u32 awa[4][16];
        alignas(64) Vec res0[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(),
                                    _mm256_setzero_si256(), _mm256_setzero_si256()};
        alignas(64) Vec res1[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(),
                                    _mm256_setzero_si256(), _mm256_setzero_si256()};

        for (int i = 0; i < 4; ++i) {
            Vec gg = _mm256_load_si256((const Vec*)(g + i * 8));
            gg = shrk(shrk(gg, Mod2), Mod);
            _mm256_store_si256((Vec*)(g + i * 8), gg);

            Vec ff = _mm256_load_si256((const Vec*)(f + i * 8));
            ff = shrk(ff, Mod2);
            Vec wv = _mm256_set1_epi32((i32)ww[i]);
            Vec ffw = shrk(mul_bsm(ff, wv, Niv, Mod), Mod);
            ff = shrk(ff, Mod);
            _mm256_store_si256((Vec*)(awa[i]), ffw);
            _mm256_store_si256((Vec*)(awa[i] + 8), ff);
        }

        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 4; ++j) {
                Vec bi = _mm256_set1_epi32((i32)g[j * 8 + i]);
                Vec aj = _mm256_loadu_si256((const Vec*)(awa[j] + 8 - i));
                Vec aj2 = _mm256_srli_epi64(aj, 32);
                res0[j] = _mm256_add_epi64(res0[j], _mm256_mul_epu32(bi, aj));
                res1[j] = _mm256_add_epi64(res1[j], _mm256_mul_epu32(bi, aj2));
            }
        }

        for (int i = 0; i < 4; ++i) {
            Vec reduced = mont_reduce_raw(res0[i], res1[i], Niv, Mod);
            _mm256_store_si256((Vec*)(f + i * 8), shrk(reduced, Mod2));
        }
    }

private:
    NTT_FORCEINLINE static Vec shrk(Vec x, Vec M) {
        return _mm256_min_epu32(x, _mm256_sub_epi32(x, M));
    }

    NTT_FORCEINLINE static Vec mul_bsm(Vec a, Vec b, Vec niv, Vec mod) {
        Vec even = _mm256_mul_epu32(a, b);
        Vec odd = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), b);
        return mont_reduce_raw(even, odd, niv, mod);
    }

    NTT_FORCEINLINE static Vec mont_reduce_raw(Vec even, Vec odd, Vec niv, Vec mod) {
        Vec ce = _mm256_mul_epu32(even, niv);
        Vec co = _mm256_mul_epu32(odd, niv);
        ce = _mm256_mul_epu32(ce, mod);
        co = _mm256_mul_epu32(co, mod);
        return _mm256_blend_epi32(
            _mm256_srli_epi64(_mm256_add_epi64(even, ce), 32),
            _mm256_add_epi64(odd, co),
            0xaa);
    }
};

} // namespace ntt
