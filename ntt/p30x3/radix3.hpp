#pragma once
#include "../common.hpp"
#include "mont_vec.hpp"
#include "root_plan.hpp"

namespace ntt {

// Radix-3 outer DIF/DIT passes for mixed-radix NTT.
// Applied once as the outermost layer when n = 3 * 2^k.
template<typename B, u32 Mod>
struct Radix3Kernel {
    using Vec = typename B::Vec;
    using MV = MontVec<B>;
    static constexpr MontScalar ms{Mod};

    // DIF pass: split n vecs into 3 sub-arrays of sub_n = n/3.
    // Inputs in [0, 2M), outputs in [0, 2M).
    static void dif_pass(Vec* f, idt n, const MV& m, const RootPlan<Mod>& roots) {
        const idt sub_n = n / 3;
        const int k = ntt_ctzll(sub_n);

        const Vec neg_half_v = B::broadcast(roots.neg_half);
        const Vec j3h_v = B::broadcast(roots.j3_half);

        const u32 tw_root_s = roots.tw3_root[k];
        const u32 tw2_root_s = ms.mul_s(tw_root_s, tw_root_s);
        const Vec tw_root_v = B::broadcast(tw_root_s);
        const Vec tw2_root_v = B::broadcast(tw2_root_s);

        // j = 0: no twiddle
        {
            const Vec a = B::load(f);
            const Vec b = B::load(f + sub_n);
            const Vec c = B::load(f + 2 * sub_n);

            const Vec s = m.add2(b, c);
            const Vec d = m.sub2(b, c);

            const Vec f0 = m.add2(a, s);
            const Vec hs = m.mont_mul_bsm(s, neg_half_v);
            const Vec jd = m.mont_mul_bsm(d, j3h_v);
            const Vec ahs = m.add2(a, hs);

            B::store(f, f0);
            B::store(f + sub_n, m.add2(ahs, jd));
            B::store(f + 2 * sub_n, m.sub2(ahs, jd));
        }

        // j > 0: with twiddles
        Vec tw = B::broadcast(tw_root_s);
        Vec tw2 = B::broadcast(tw2_root_s);
        for (idt j = 1; j < sub_n; ++j) {
            const Vec a = B::load(f + j);
            const Vec b = B::load(f + j + sub_n);
            const Vec c = B::load(f + j + 2 * sub_n);

            const Vec s = m.add2(b, c);
            const Vec d = m.sub2(b, c);

            const Vec f0 = m.add2(a, s);
            const Vec hs = m.mont_mul_bsm(s, neg_half_v);
            const Vec jd = m.mont_mul_bsm(d, j3h_v);
            const Vec ahs = m.add2(a, hs);

            B::store(f + j, f0);
            B::store(f + j + sub_n, m.mont_mul_bsm(m.lazy_add(ahs, jd), tw));
            B::store(f + j + 2 * sub_n, m.mont_mul_bsm(m.lazy_sub(ahs, jd), tw2));

            tw = m.mont_mul_bsm(tw, tw_root_v);
            tw2 = m.mont_mul_bsm(tw2, tw2_root_v);
        }
    }

    // DIT pass: inverse of DIF, fuses 1/3 scale.
    // Inputs in [0, M) (from inv_b2), outputs in [0, M) (with final shrink).
    static void dit_pass(Vec* f, idt n, const MV& m, const RootPlan<Mod>& roots) {
        const idt sub_n = n / 3;
        const int k = ntt_ctzll(sub_n);

        const Vec neg_half_v = B::broadcast(roots.neg_half);
        const Vec j3h_v = B::broadcast(roots.j3_half);
        const Vec inv3_v = B::broadcast(roots.inv3);

        const u32 tw_inv_root_s = roots.tw3i_root[k];
        const u32 tw2_inv_root_s = ms.mul_s(tw_inv_root_s, tw_inv_root_s);
        const Vec tw_inv_root_v = B::broadcast(tw_inv_root_s);
        const Vec tw2_inv_root_v = B::broadcast(tw2_inv_root_s);

        // Running twiddle*inv3 scalars: start at inv3 (j=0), multiply by tw_inv_root each step
        Vec tw1_s = inv3_v;       // ω_N^{0} * inv3 = inv3
        Vec tw2_s = inv3_v;       // ω_N^{0} * inv3 = inv3

        for (idt j = 0; j < sub_n; ++j) {
            Vec f0 = B::load(f + j);
            Vec f1 = B::load(f + j + sub_n);
            Vec f2 = B::load(f + j + 2 * sub_n);

            // Scale + undo twiddle (fused)
            f0 = m.mont_mul_bsm(f0, inv3_v);
            f1 = m.mont_mul_bsm(f1, tw1_s);
            f2 = m.mont_mul_bsm(f2, tw2_s);

            // Inverse butterfly (w3 ↔ w3^2 swap vs DIF)
            const Vec s = m.add2(f1, f2);
            const Vec d = m.sub2(f1, f2);

            const Vec hs = m.mont_mul_bsm(s, neg_half_v);
            const Vec jd = m.mont_mul_bsm(d, j3h_v);
            const Vec ahs = m.add2(f0, hs);

            B::store(f + j, m.shrink(m.add2(f0, s)));
            B::store(f + j + sub_n, m.shrink(m.sub2(ahs, jd)));
            B::store(f + j + 2 * sub_n, m.shrink(m.add2(ahs, jd)));

            // Update fused twiddle scalars
            tw1_s = m.mont_mul_bsm(tw1_s, tw_inv_root_v);
            tw2_s = m.mont_mul_bsm(tw2_s, tw2_inv_root_v);
        }
    }
};

} // namespace ntt
