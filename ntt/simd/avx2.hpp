#pragma once
#include "../common.hpp"

namespace ntt {

struct Avx2 {
    using Vec = __m256i;
    static constexpr int LANES = 8;
    static constexpr int LOG_LANES = 3;
    static constexpr int ALIGN = 32;

    NTT_FORCEINLINE static Vec load(const void* p) {
        return _mm256_load_si256((const Vec*)p);
    }
    NTT_FORCEINLINE static Vec loadu(const void* p) {
        return _mm256_loadu_si256((const Vec*)p);
    }
    NTT_FORCEINLINE static void store(void* p, Vec x) {
        _mm256_store_si256((Vec*)p, x);
    }
    NTT_FORCEINLINE static Vec broadcast(u32 x) {
        return _mm256_set1_epi32((i32)x);
    }
    NTT_FORCEINLINE static Vec add32(Vec a, Vec b) {
        return _mm256_add_epi32(a, b);
    }
    NTT_FORCEINLINE static Vec sub32(Vec a, Vec b) {
        return _mm256_sub_epi32(a, b);
    }
    NTT_FORCEINLINE static Vec min32(Vec a, Vec b) {
        return _mm256_min_epu32(a, b);
    }
    NTT_FORCEINLINE static Vec mullo32(Vec a, Vec b) {
        return _mm256_mullo_epi32(a, b);
    }
    // Even-lane 32x32->64 widening multiply
    NTT_FORCEINLINE static Vec mul64(Vec a, Vec b) {
        return _mm256_mul_epu32(a, b);
    }
    NTT_FORCEINLINE static Vec srl64(Vec a, int imm) {
        return _mm256_srli_epi64(a, imm);
    }
    NTT_FORCEINLINE static Vec add64(Vec a, Vec b) {
        return _mm256_add_epi64(a, b);
    }
    NTT_FORCEINLINE static Vec blend_0xaa(Vec a, Vec b) {
        return _mm256_blend_epi32(a, b, 0xaa);
    }
    NTT_FORCEINLINE static Vec permutevar(Vec a, Vec idx) {
        return _mm256_permutevar8x32_epi32(a, idx);
    }
    NTT_FORCEINLINE static Vec shuffle_AAAA(Vec a) {
        return _mm256_shuffle_epi32(a, 0x00);
    }
    NTT_FORCEINLINE static Vec shuffle_BBBB(Vec a) {
        return _mm256_shuffle_epi32(a, 0x55);
    }
    NTT_FORCEINLINE static Vec shuffle_CCCC(Vec a) {
        return _mm256_shuffle_epi32(a, 0xaa);
    }
    NTT_FORCEINLINE static Vec shuffle_DDDD(Vec a) {
        return _mm256_shuffle_epi32(a, 0xff);
    }
    NTT_FORCEINLINE static Vec shuffle_CDAB(Vec a) {
        return _mm256_shuffle_epi32(a, 0xb1);  // _MM_PERM_CDAB = 2,3,0,1 = 0xb1
    }
    NTT_FORCEINLINE static Vec alignr4(Vec hi, Vec lo) {
        return _mm256_alignr_epi8(hi, lo, 4);
    }
    NTT_FORCEINLINE static Vec alignr8(Vec hi, Vec lo) {
        return _mm256_alignr_epi8(hi, lo, 8);
    }
    NTT_FORCEINLINE static Vec alignr12(Vec hi, Vec lo) {
        return _mm256_alignr_epi8(hi, lo, 12);
    }
    NTT_FORCEINLINE static Vec permute2f128_03(Vec a, Vec b) {
        return _mm256_permute2x128_si256(a, b, 0x03);
    }
    NTT_FORCEINLINE static Vec permute4x64_00(Vec a) {
        return _mm256_permute4x64_epi64(a, 0x00);
    }
    NTT_FORCEINLINE static Vec permute4x64_55(Vec a) {
        return _mm256_permute4x64_epi64(a, 0x55);
    }
    NTT_FORCEINLINE static Vec permute4x64_aa(Vec a) {
        return _mm256_permute4x64_epi64(a, 0xaa);
    }
    NTT_FORCEINLINE static Vec permute4x64_ff(Vec a) {
        return _mm256_permute4x64_epi64(a, 0xff);
    }
    NTT_FORCEINLINE static Vec setr(u32 a, u32 b, u32 c, u32 d,
                                     u32 e, u32 f, u32 g, u32 h) {
        return _mm256_setr_epi32((i32)a,(i32)b,(i32)c,(i32)d,
                                 (i32)e,(i32)f,(i32)g,(i32)h);
    }
    NTT_FORCEINLINE static Vec zero() {
        return _mm256_setzero_si256();
    }
};

} // namespace ntt
