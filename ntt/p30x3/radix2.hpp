#pragma once
#include "../common.hpp"
#include "mont_vec.hpp"

namespace ntt {

template<typename B>
struct Radix2Kernel {
    using Vec = typename B::Vec;
    using M = MontVec<B>;

    // DIF radix-2 pass: half = n/2 Vecs
    // Inputs in [0, 2M), outputs: p0 [0,2M), p1 [0,4M)
    NTT_FORCEINLINE static void dif_pass(Vec* f, idt half, const M& m) {
        for (idt i = 0; i < half; ++i) {
            Vec* p0 = f + i;
            Vec* p1 = f + half + i;
            const Vec f0 = B::load(p0);
            const Vec f1 = B::load(p1);
            const Vec g0 = m.add2(f0, f1);
            const Vec g1 = m.lazy_sub(f0, f1);
            B::store(p0, g0);
            B::store(p1, g1);
        }
    }

    // DIT radix-2 pass: half = n/2 Vecs, with final shrink to [0, M)
    // Inputs may be in [0, 2M)
    NTT_FORCEINLINE static void dit_pass(Vec* f, idt half, const M& m) {
        for (idt i = 0; i < half; ++i) {
            Vec* p0 = f + i;
            Vec* p1 = f + half + i;
            const Vec f0 = B::load(p0);
            const Vec f1 = B::load(p1);
            const Vec g0 = m.add2(f0, f1);
            const Vec g1 = m.sub2(f0, f1);
            B::store(p0, m.shrink(g0));
            B::store(p1, m.shrink(g1));
        }
    }
};

} // namespace ntt
