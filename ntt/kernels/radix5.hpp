#pragma once
#include "../common.hpp"
#include "../mont/mont_vec.hpp"
#include "../roots/root_plan.hpp"

namespace ntt {

// Radix-5 outer DIF/DIT passes for mixed-radix NTT.
// Applied once as the outermost layer when n = 5 * 2^k.
// Uses Karatsuba-style 6-mul butterfly core.
template<typename B, u32 Mod>
struct Radix5Kernel {
    using Vec = typename B::Vec;
    using MV = MontVec<B>;
    static constexpr MontScalar ms{Mod};

    // DIF pass: split n vecs into 5 sub-arrays of sub_n = n/5.
    // Inputs in [0, 2M), outputs in [0, 2M).
    static void dif_pass(Vec* f, idt n, const MV& m, const RootPlan<Mod>& roots) {
        const idt sub_n = n / 5;
        const int k = ntt_ctzll(sub_n);

        const Vec c1h_v = B::broadcast(roots.c1h);
        const Vec c2h_v = B::broadcast(roots.c2h);
        const Vec j1h_v = B::broadcast(roots.j1h);
        const Vec j2h_v = B::broadcast(roots.j2h);

        // Precompute c12h = c1h + c2h, j12s = j1h + j2h (for Karatsuba)
        const Vec c12h_v = B::broadcast(ms.shrink(roots.c1h + roots.c2h));
        const Vec j12s_v = B::broadcast(ms.shrink(roots.j1h + roots.j2h));

        const u32 tw_root_s = roots.tw5_root[k];
        const u32 tw2_s = ms.mul_s(tw_root_s, tw_root_s);
        const u32 tw3_s = ms.mul_s(tw2_s, tw_root_s);
        const u32 tw4_s = ms.mul_s(tw2_s, tw2_s);
        const Vec tw_root_v = B::broadcast(tw_root_s);
        const Vec tw2_root_v = B::broadcast(tw2_s);
        const Vec tw3_root_v = B::broadcast(tw3_s);
        const Vec tw4_root_v = B::broadcast(tw4_s);

        // j = 0: no twiddle
        {
            const Vec a = B::load(f);
            const Vec b = B::load(f + sub_n);
            const Vec c = B::load(f + 2 * sub_n);
            const Vec d = B::load(f + 3 * sub_n);
            const Vec e = B::load(f + 4 * sub_n);

            const Vec s1 = m.add2(b, e);
            const Vec t1 = m.sub2(b, e);
            const Vec s2 = m.add2(c, d);
            const Vec t2 = m.sub2(c, d);

            // f0 = a + s1 + s2
            const Vec f0 = m.add2(a, m.add2(s1, s2));

            // Karatsuba for c-terms: p1=c1h*s1, p2=c2h*s2, p3=c12h*(s1+s2)
            const Vec p1 = m.mont_mul_bsm(s1, c1h_v);
            const Vec p2 = m.mont_mul_bsm(s2, c2h_v);
            const Vec p3 = m.mont_mul_bsm(m.add2(s1, s2), c12h_v);
            // alpha = a + p1 + p2, gamma = a + (p3 - p1 - p2)
            const Vec pp = m.add2(p1, p2);
            const Vec alpha = m.add2(a, pp);
            const Vec gamma = m.add2(a, m.sub2(p3, pp));

            // Karatsuba for j-terms: q1=j1h*t1, q2=j2h*t2, q3=(j1h+j2h)*(t1-t2)
            // beta = q1 + q2 = j1h*t1 + j2h*t2
            // delta = q3 - q1 + q2 = j2h*t1 - j1h*t2
            const Vec q1 = m.mont_mul_bsm(t1, j1h_v);
            const Vec q2 = m.mont_mul_bsm(t2, j2h_v);
            const Vec q3 = m.mont_mul_bsm(m.sub2(t1, t2), j12s_v);
            const Vec beta = m.add2(q1, q2);
            const Vec delta = m.add2(m.sub2(q3, q1), q2);

            B::store(f, f0);
            B::store(f + sub_n, m.add2(alpha, beta));
            B::store(f + 2 * sub_n, m.add2(gamma, delta));
            B::store(f + 3 * sub_n, m.sub2(gamma, delta));
            B::store(f + 4 * sub_n, m.sub2(alpha, beta));
        }

        // j > 0: with twiddles
        Vec tw1 = B::broadcast(tw_root_s);
        Vec tw2 = B::broadcast(tw2_s);
        Vec tw3 = B::broadcast(tw3_s);
        Vec tw4 = B::broadcast(tw4_s);
        for (idt j = 1; j < sub_n; ++j) {
            const Vec a = B::load(f + j);
            const Vec b = B::load(f + j + sub_n);
            const Vec c = B::load(f + j + 2 * sub_n);
            const Vec d = B::load(f + j + 3 * sub_n);
            const Vec e = B::load(f + j + 4 * sub_n);

            const Vec s1 = m.add2(b, e);
            const Vec t1 = m.sub2(b, e);
            const Vec s2 = m.add2(c, d);
            const Vec t2 = m.sub2(c, d);

            const Vec f0 = m.add2(a, m.add2(s1, s2));

            const Vec p1 = m.mont_mul_bsm(s1, c1h_v);
            const Vec p2 = m.mont_mul_bsm(s2, c2h_v);
            const Vec p3 = m.mont_mul_bsm(m.add2(s1, s2), c12h_v);
            const Vec pp = m.add2(p1, p2);
            const Vec alpha = m.add2(a, pp);
            const Vec gamma = m.add2(a, m.sub2(p3, pp));

            const Vec q1 = m.mont_mul_bsm(t1, j1h_v);
            const Vec q2 = m.mont_mul_bsm(t2, j2h_v);
            const Vec q3 = m.mont_mul_bsm(m.sub2(t1, t2), j12s_v);
            const Vec beta = m.add2(q1, q2);
            const Vec delta = m.add2(m.sub2(q3, q1), q2);

            B::store(f + j, f0);
            B::store(f + j + sub_n, m.mont_mul_bsm(m.lazy_add(alpha, beta), tw1));
            B::store(f + j + 2 * sub_n, m.mont_mul_bsm(m.lazy_add(gamma, delta), tw2));
            B::store(f + j + 3 * sub_n, m.mont_mul_bsm(m.lazy_sub(gamma, delta), tw3));
            B::store(f + j + 4 * sub_n, m.mont_mul_bsm(m.lazy_sub(alpha, beta), tw4));

            tw1 = m.mont_mul_bsm(tw1, tw_root_v);
            tw2 = m.mont_mul_bsm(tw2, tw2_root_v);
            tw3 = m.mont_mul_bsm(tw3, tw3_root_v);
            tw4 = m.mont_mul_bsm(tw4, tw4_root_v);
        }
    }

    // DIT pass: inverse of DIF, fuses 1/5 scale.
    // Inputs in [0, M) (from inv_b2), outputs in [0, M) (with final shrink).
    static void dit_pass(Vec* f, idt n, const MV& m, const RootPlan<Mod>& roots) {
        const idt sub_n = n / 5;
        const int k = ntt_ctzll(sub_n);

        const Vec c1h_v = B::broadcast(roots.c1h);
        const Vec c2h_v = B::broadcast(roots.c2h);
        const Vec j1h_v = B::broadcast(roots.j1h);
        const Vec j2h_v = B::broadcast(roots.j2h);
        const Vec c12h_v = B::broadcast(ms.shrink(roots.c1h + roots.c2h));
        const Vec j12s_v = B::broadcast(ms.shrink(roots.j1h + roots.j2h));
        const Vec inv5_v = B::broadcast(roots.inv5);

        const u32 tw_inv_root_s = roots.tw5i_root[k];
        const u32 tw2i_s = ms.mul_s(tw_inv_root_s, tw_inv_root_s);
        const u32 tw3i_s = ms.mul_s(tw2i_s, tw_inv_root_s);
        const u32 tw4i_s = ms.mul_s(tw2i_s, tw2i_s);
        const Vec tw_inv_root_v = B::broadcast(tw_inv_root_s);
        const Vec tw2i_root_v = B::broadcast(tw2i_s);
        const Vec tw3i_root_v = B::broadcast(tw3i_s);
        const Vec tw4i_root_v = B::broadcast(tw4i_s);

        // Fused twiddle*inv5 running scalars
        Vec tws1 = inv5_v;
        Vec tws2 = inv5_v;
        Vec tws3 = inv5_v;
        Vec tws4 = inv5_v;

        for (idt j = 0; j < sub_n; ++j) {
            Vec fa = B::load(f + j);
            Vec fb = B::load(f + j + sub_n);
            Vec fc = B::load(f + j + 2 * sub_n);
            Vec fd = B::load(f + j + 3 * sub_n);
            Vec fe = B::load(f + j + 4 * sub_n);

            // Scale + undo twiddle (fused)
            fa = m.mont_mul_bsm(fa, inv5_v);
            fb = m.mont_mul_bsm(fb, tws1);
            fc = m.mont_mul_bsm(fc, tws2);
            fd = m.mont_mul_bsm(fd, tws3);
            fe = m.mont_mul_bsm(fe, tws4);

            // Inverse butterfly (swapped roots vs DIF)
            // In DIF: output order is f0, f1=alpha+beta, f2=gamma+delta, f3=gamma-delta, f4=alpha-beta
            // DIT undoes: b↔e positions have alpha±beta, c↔d have gamma±delta
            const Vec s1 = m.add2(fb, fe);
            const Vec t1 = m.sub2(fb, fe);
            const Vec s2 = m.add2(fc, fd);
            const Vec t2 = m.sub2(fc, fd);

            const Vec f0 = m.add2(fa, m.add2(s1, s2));

            const Vec p1 = m.mont_mul_bsm(s1, c1h_v);
            const Vec p2 = m.mont_mul_bsm(s2, c2h_v);
            const Vec p3 = m.mont_mul_bsm(m.add2(s1, s2), c12h_v);
            const Vec pp = m.add2(p1, p2);
            const Vec alpha = m.add2(fa, pp);
            const Vec gamma = m.add2(fa, m.sub2(p3, pp));

            // Karatsuba for j-terms (same formula as DIF)
            const Vec q1 = m.mont_mul_bsm(t1, j1h_v);
            const Vec q2 = m.mont_mul_bsm(t2, j2h_v);
            const Vec q3 = m.mont_mul_bsm(m.sub2(t1, t2), j12s_v);
            const Vec beta = m.add2(q1, q2);
            const Vec delta = m.add2(m.sub2(q3, q1), q2);

            // Signs flipped vs DIF for outputs 1↔4 and 2↔3
            B::store(f + j, m.shrink(f0));
            B::store(f + j + sub_n, m.shrink(m.sub2(alpha, beta)));
            B::store(f + j + 2 * sub_n, m.shrink(m.sub2(gamma, delta)));
            B::store(f + j + 3 * sub_n, m.shrink(m.add2(gamma, delta)));
            B::store(f + j + 4 * sub_n, m.shrink(m.add2(alpha, beta)));

            // Update fused twiddle scalars
            tws1 = m.mont_mul_bsm(tws1, tw_inv_root_v);
            tws2 = m.mont_mul_bsm(tws2, tw2i_root_v);
            tws3 = m.mont_mul_bsm(tws3, tw3i_root_v);
            tws4 = m.mont_mul_bsm(tws4, tw4i_root_v);
        }
    }
};

} // namespace ntt
