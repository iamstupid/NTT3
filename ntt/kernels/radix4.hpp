#pragma once
#include "../common.hpp"
#include "../mont/mont_vec.hpp"

namespace ntt {

template<typename B>
struct Radix4Kernel {
    using Vec = typename B::Vec;
    using M = MontVec<B>;

    // DIF: pure butterfly (r=0 path, no twiddle)
    // Inputs in [0, 2M), outputs: p0 [0,2M), p1/p2/p3 [0,4M)
    NTT_FORCEINLINE static void dif_butterfly_notw(
        Vec* p0, Vec* p1, Vec* p2, Vec* p3,
        idt L, const M& m)
    {
        for (idt i = 0; i < L; ++i) {
            const Vec f0 = B::load(p0 + i);
            const Vec f1 = B::load(p1 + i);
            const Vec f2 = B::load(p2 + i);
            const Vec f3 = B::load(p3 + i);

            const Vec g1 = m.add2(f1, f3);
            const Vec g3 = m.mul_by_img(m.lazy_sub(f1, f3));
            const Vec g0 = m.add2(f0, f2);
            const Vec g2 = m.sub2(f0, f2);

            B::store(p0 + i, m.add2(g0, g1));
            B::store(p1 + i, m.lazy_sub(g0, g1));
            B::store(p2 + i, m.lazy_add(g2, g3));
            B::store(p3 + i, m.lazy_sub(g2, g3));
        }
    }

    // DIF: twiddle-fused butterfly (blocked loop)
    // Inputs may be in [0, 4M), outputs in [0, 4M)
    NTT_FORCEINLINE static void dif_butterfly(
        Vec* p0, Vec* p1, Vec* p2, Vec* p3,
        idt L,
        Vec r1, Vec r1_niv,
        Vec r2, Vec r2_niv,
        Vec nr3, Vec nr3_niv,
        const M& m)
    {
        for (idt j = 0; j < L; ++j) {
            const Vec f0 = B::load(p0 + j);
            const Vec f1 = B::load(p1 + j);
            const Vec f2 = B::load(p2 + j);
            const Vec f3 = B::load(p3 + j);

            const Vec g1 = m.mont_mul_precomp(f1, r1, r1_niv);
            const Vec ng3 = m.mont_mul_precomp(f3, nr3, nr3_niv);
            const Vec g2 = m.mont_mul_precomp(f2, r2, r2_niv);
            const Vec g0 = m.shrink2(f0);

            const Vec h3 = m.mul_by_img(m.lazy_add(g1, ng3));
            const Vec h1 = m.sub2(g1, ng3);
            const Vec h0 = m.add2(g0, g2);
            const Vec h2 = m.sub2(g0, g2);

            B::store(p0 + j, m.lazy_add(h0, h1));
            B::store(p1 + j, m.lazy_sub(h0, h1));
            B::store(p2 + j, m.lazy_add(h2, h3));
            B::store(p3 + j, m.lazy_sub(h2, h3));
        }
    }

    // DIT: innermost layer (L=1) with scale (N^{-1}) fusion
    // Inputs may be in [0, 4M), outputs in [0, 2M)
    NTT_FORCEINLINE static void dit_butterfly_scale(
        Vec* p0, Vec* p1, Vec* p2, Vec* p3,
        Vec r1, Vec r2, Vec r3,
        Vec fx, Vec fx_niv,
        const M& m)
    {
        const Vec f0 = B::load(p0);
        const Vec f1 = B::load(p1);
        const Vec f2 = B::load(p2);
        const Vec f3 = B::load(p3);

        const Vec g3 = m.mul_by_img(m.lazy_sub(f3, f2));
        const Vec g2 = m.add2(f2, f3);
        const Vec g0 = m.add2(f0, f1);
        const Vec g1 = m.sub2(f0, f1);

        const Vec h2 = m.lazy_sub(g0, g2);
        const Vec h3 = m.lazy_sub(g1, g3);
        const Vec h0 = m.lazy_add(g0, g2);
        const Vec h1 = m.lazy_add(g1, g3);

        B::store(p0, m.mont_mul_precomp(h0, fx, fx_niv));
        B::store(p1, m.mont_mul_bsm(h1, r1));
        B::store(p2, m.mont_mul_bsm(h2, r2));
        B::store(p3, m.mont_mul_bsm(h3, r3));
    }

    // DIT: twiddle-fused butterfly (blocked loop, inner layers)
    // Inputs may be in [0, 4M), outputs in [0, 2M)
    NTT_FORCEINLINE static void dit_butterfly(
        Vec* p0, Vec* p1, Vec* p2, Vec* p3,
        idt L,
        Vec r1, Vec r1_niv,
        Vec r2, Vec r2_niv,
        Vec r3, Vec r3_niv,
        const M& m)
    {
        for (idt j = 0; j < L; ++j) {
            const Vec f0 = B::load(p0 + j);
            const Vec f1 = B::load(p1 + j);
            const Vec f2 = B::load(p2 + j);
            const Vec f3 = B::load(p3 + j);

            const Vec g3 = m.mul_by_img(m.lazy_sub(f3, f2));
            const Vec g2 = m.add2(f2, f3);
            const Vec g0 = m.add2(f0, f1);
            const Vec g1 = m.sub2(f0, f1);

            const Vec h2 = m.lazy_sub(g0, g2);
            const Vec h3 = m.lazy_sub(g1, g3);
            const Vec h0 = m.lazy_add(g0, g2);
            const Vec h1 = m.lazy_add(g1, g3);

            B::store(p0 + j, m.shrink2(h0));
            B::store(p1 + j, m.mont_mul_precomp(h1, r1, r1_niv));
            B::store(p2 + j, m.mont_mul_precomp(h2, r2, r2_niv));
            B::store(p3 + j, m.mont_mul_precomp(h3, r3, r3_niv));
        }
    }

    // DIT: outermost layer, no twiddle
    // Inputs may be in [0, 4M), outputs in [0, 2M) or [0, M) with final shrink
    NTT_FORCEINLINE static void dit_butterfly_notw(
        Vec* p0, Vec* p1, Vec* p2, Vec* p3,
        idt L, const M& m, bool final_shrink)
    {
        for (idt i = 0; i < L; ++i) {
            const Vec f0 = B::load(p0 + i);
            const Vec f1 = B::load(p1 + i);
            const Vec f2 = B::load(p2 + i);
            const Vec f3 = B::load(p3 + i);

            const Vec g3 = m.mul_by_img(m.lazy_sub(f3, f2));
            const Vec g2 = m.add2(f2, f3);
            const Vec g0 = m.add2(f0, f1);
            const Vec g1 = m.sub2(f0, f1);

            const Vec h0 = m.add2(g0, g2);
            const Vec h1 = m.add2(g1, g3);
            const Vec h2 = m.sub2(g0, g2);
            const Vec h3 = m.sub2(g1, g3);

            if (final_shrink) {
                B::store(p0 + i, m.shrink(h0));
                B::store(p1 + i, m.shrink(h1));
                B::store(p2 + i, m.shrink(h2));
                B::store(p3 + i, m.shrink(h3));
            } else {
                B::store(p0 + i, h0);
                B::store(p1 + i, h1);
                B::store(p2 + i, h2);
                B::store(p3 + i, h3);
            }
        }
    }
};

} // namespace ntt
