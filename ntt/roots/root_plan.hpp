#pragma once
#include "../common.hpp"
#include "../mont/mont_scalar.hpp"
#include <array>

namespace ntt {

// RootPlan: precomputes all roots of unity and jump factors for ruler-sequence updates.
// Uses constexpr-capable bit ops so it can be initialized at compile time.
template<u32 Mod>
struct RootPlan {
    static constexpr int MAX_LG = MAX_LOG;
    static constexpr MontScalar ms{Mod};

    u32 img, imgniv;

    // RT1[i]: jump factor for twisted conv root chain
    u32 RT1[MAX_LG];
    // RT3[i]: jump factor for twiddle root chain
    u32 RT3[MAX_LG];

    // rt3[i]: packed twiddle jump vectors for DIF ruler sequence
    // Layout per entry: {r*niv, r, r^2*niv, r^2, r^3*niv, r^3, 0, 0}
    u32 rt3[MAX_LG - 2][8];
    // rt3i[i]: inverse twiddle jump vectors for DIT ruler sequence
    u32 rt3i[MAX_LG - 2][8];

    // bwb, bwbi: initial root state for DIF/DIT
    u32 bwb[8];
    u32 bwbi[8];

    // Twisted conv root chain jump factors (64-bit packed)
    u64 rt4n[MAX_LG - 3];
    u64 rt4n2[MAX_LG - 3];
    u64 rt4ni[MAX_LG - 3];
    u64 rt4n2i[MAX_LG - 3];

    // Twisted conv initial roots
    u32 rt4nr[8];
    u32 rt4nr2[8];
    u32 rt4nri[8];
    u32 rt4nr2i[8];

    // ── Radix-3/5 constants (Montgomery form, [0, M)) ──
    u32 neg_half;       // -1/2 mod P
    u32 j3_half;        // (ω_3 - ω_3^2) / 2
    u32 inv3;           // 1/3 mod P
    u32 inv5;           // 1/5 mod P
    u32 c1h, c2h;       // radix-5: (ω_5+ω_5^4)/2, (ω_5^2+ω_5^3)/2
    u32 j1h, j2h;       // radix-5: (ω_5-ω_5^4)/2, (ω_5^2-ω_5^3)/2

    // Twiddle roots for outer radix-3/5 DIF/DIT passes
    // tw3_root[k] = ω_{3*2^k} (forward), tw3i_root[k] = ω_{3*2^k}^{-1} (inverse)
    u32 tw3_root[MAX_LG + 1];
    u32 tw3i_root[MAX_LG + 1];
    u32 tw5_root[MAX_LG + 1];
    u32 tw5i_root[MAX_LG + 1];

    constexpr RootPlan() :
        img(0), imgniv(0),
        RT1{}, RT3{}, rt3{}, rt3i{}, bwb{}, bwbi{},
        rt4n{}, rt4n2{}, rt4ni{}, rt4n2i{},
        rt4nr{}, rt4nr2{}, rt4nri{}, rt4nr2i{},
        neg_half(0), j3_half(0), inv3(0), inv5(0),
        c1h(0), c2h(0), j1h(0), j2h(0),
        tw3_root{}, tw3i_root{}, tw5_root{}, tw5i_root{}
    {
        const int k = ctz_constexpr(Mod - 1);

        // Find primitive root in Montgomery form.
        // Must be a full primitive root (order = Mod-1) so that
        // radix-3/5 roots derived from it are non-trivial.
        // Check g^{(Mod-1)/q} != 1 for each prime factor q of Mod-1.
        u32 g_mont = ms.to_mont(3);
        for (;;) {
            bool ok = true;
            // Factor out all prime factors of Mod-1 and test each
            u32 rem = Mod - 1;
            // Test factor 2
            if (rem % 2 == 0) {
                if (ms.power_s(g_mont, (Mod - 1) / 2, ms.one) == ms.one) ok = false;
                while (rem % 2 == 0) rem /= 2;
            }
            // Test remaining odd prime factors
            for (u32 p = 3; ok && p * p <= rem; p += 2) {
                if (rem % p == 0) {
                    if (ms.power_s(g_mont, (Mod - 1) / p, ms.one) == ms.one) ok = false;
                    while (rem % p == 0) rem /= p;
                }
            }
            if (ok && rem > 1) {
                if (ms.power_s(g_mont, (Mod - 1) / rem, ms.one) == ms.one) ok = false;
            }
            if (ok) break;
            g_mont = ms.reduce(u64(g_mont) + ms.r2);
        }

        // Save full primitive root before powering
        const u32 g_full = g_mont;

        // omega_{2^k}
        g_mont = ms.power(g_mont, Mod >> k, ms.one);

        // rt1[i] = omega_{2^{i+2}}, rt1i[i] = omega_{2^{i+2}}^{-1}
        u32 rt1[MAX_LG - 1] = {};
        u32 rt1i[MAX_LG - 1] = {};
        rt1[k - 2] = g_mont;
        rt1i[k - 2] = ms.power(g_mont, Mod - 2, ms.one);
        for (int i = k - 2; i > 0; --i) {
            rt1[i - 1] = ms.mul(rt1[i], rt1[i]);
            rt1i[i - 1] = ms.mul(rt1i[i], rt1i[i]);
        }

        // img = omega_4
        img = rt1[0];
        imgniv = img * ms.niv;

        // RT1: jump factors for twisted conv
        RT1[k - 1] = ms.power_s(g_mont, 3, ms.one);
        for (int i = k - 1; i > 0; --i) {
            RT1[i - 1] = ms.mul_s(RT1[i], RT1[i]);
        }

        // bwb: initial state for DIF
        bwb[0] = rt1[1]; bwb[1] = 0;
        bwb[2] = rt1[0]; bwb[3] = 0;
        bwb[4] = Mod - ms.mul_s(rt1[0], rt1[1]); bwb[5] = 0;
        bwb[6] = 0; bwb[7] = 0;

        // bwbi: initial state for DIT
        bwbi[0] = rt1i[1]; bwbi[1] = 0;
        bwbi[2] = rt1i[0]; bwbi[3] = 0;
        bwbi[4] = ms.mul_s(rt1i[0], rt1i[1]); bwbi[5] = 0;
        bwbi[6] = 0; bwbi[7] = 0;

        // rt3 / rt3i: twiddle jump factors for ruler sequence
        u32 pr = ms.one, pri = ms.one;
        for (int i = 0; i < k - 2; ++i) {
            const u32 r = ms.mul_s(pr, rt1[i + 1]);
            const u32 ri = ms.mul_s(pri, rt1i[i + 1]);
            const u32 r2 = ms.mul_s(r, r);
            const u32 r2i = ms.mul_s(ri, ri);
            const u32 r3v = ms.mul_s(r, r2);
            const u32 r3i = ms.mul_s(ri, r2i);

            rt3[i][0] = r * ms.niv;  rt3[i][1] = r;
            rt3[i][2] = r2 * ms.niv; rt3[i][3] = r2;
            rt3[i][4] = r3v * ms.niv; rt3[i][5] = r3v;
            rt3[i][6] = 0;           rt3[i][7] = 0;

            RT3[i + 2] = rt3[i][1];

            rt3i[i][0] = ri * ms.niv;  rt3i[i][1] = ri;
            rt3i[i][2] = r2i * ms.niv; rt3i[i][3] = r2i;
            rt3i[i][4] = r3i * ms.niv; rt3i[i][5] = r3i;
            rt3i[i][6] = 0;           rt3i[i][7] = 0;

            pr = ms.mul(pr, rt1i[i + 1]);
            pri = ms.mul(pri, rt1[i + 1]);
        }

        // Twisted convolution roots
        u32 w[8] = {}, wi[8] = {};
        w[0] = ms.one; wi[0] = ms.one;
        for (int i = 0; i < 3; ++i) {
            u32 p = rt1[i], pi = rt1i[i];
            for (int j = 1 << i, kk = 0; kk < j; ++kk) {
                w[j + kk] = ms.mul_s(w[kk], p);
                wi[j + kk] = ms.mul_s(wi[kk], pi);
            }
        }

        rt4nr[0]=w[7]; rt4nr[1]=w[0]; rt4nr[2]=w[6]; rt4nr[3]=w[1];
        rt4nr[4]=w[5]; rt4nr[5]=w[2]; rt4nr[6]=w[4]; rt4nr[7]=w[3];
        rt4nr2[0]=w[1]; rt4nr2[1]=w[3]; rt4nr2[2]=w[1]; rt4nr2[3]=w[0];
        rt4nr2[4]=w[0]; rt4nr2[5]=w[2]; rt4nr2[6]=w[0]; rt4nr2[7]=w[1];

        rt4nri[0]=wi[7]; rt4nri[1]=wi[0]; rt4nri[2]=wi[6]; rt4nri[3]=wi[1];
        rt4nri[4]=wi[5]; rt4nri[5]=wi[2]; rt4nri[6]=wi[4]; rt4nri[7]=wi[3];
        rt4nr2i[0]=wi[1]; rt4nr2i[1]=wi[3]; rt4nr2i[2]=wi[1]; rt4nr2i[3]=wi[0];
        rt4nr2i[4]=wi[0]; rt4nr2i[5]=wi[2]; rt4nr2i[6]=wi[0]; rt4nr2i[7]=wi[1];

        // rt4n / rt4n2: twisted conv jump factors
        pr = ms.one; pri = ms.one;
        for (int i = 1; i < k - 3; ++i) {
            const u32 rv = ms.mul_s(pr, rt1[i + 2]);
            const u32 ri = ms.mul_s(pri, rt1i[i + 2]);
            const u32 rv2 = ms.mul_s(rv, rv);
            const u32 ri2 = ms.mul_s(ri, ri);
            const u32 rv4 = ms.mul_s(rv2, rv2);
            const u32 ri4 = ms.mul_s(ri2, ri2);

            rt4n[i]  = u64(rv * ms.niv) << 32 | rv;
            rt4ni[i] = u64(ri * ms.niv) << 32 | ri;
            rt4n2[i]  = u64(rv2) << 32 | rv4;
            rt4n2i[i] = u64(ri2) << 32 | ri4;

            pr = ms.mul(pr, rt1i[i + 2]);
            pri = ms.mul(pri, rt1[i + 2]);
        }

        // ── Radix-3/5 constants ──
        {
            const u32 half_m = ms.to_mont((Mod + 1) / 2);  // 1/2 in Montgomery
            neg_half = ms.to_mont((Mod - 1) / 2);          // -1/2 in Montgomery

            inv3 = ms.power_s(ms.to_mont(3), Mod - 2, ms.one);
            inv5 = ms.power_s(ms.to_mont(5), Mod - 2, ms.one);

            // ω_3 and derived
            u32 w3 = ms.power_s(g_full, (Mod - 1) / 3, ms.one);
            u32 w3sq = ms.mul_s(w3, w3);
            u32 j3 = ms.shrink(w3 + (Mod - w3sq));  // (ω_3 - ω_3^2) mod P
            j3_half = ms.mul_s(j3, half_m);

            // ω_5 and derived
            u32 w5_1 = ms.power_s(g_full, (Mod - 1) / 5, ms.one);
            u32 w5_2 = ms.mul_s(w5_1, w5_1);
            u32 w5_3 = ms.mul_s(w5_2, w5_1);
            u32 w5_4 = ms.mul_s(w5_3, w5_1);

            u32 s14 = ms.shrink(w5_1 + w5_4);
            u32 s23 = ms.shrink(w5_2 + w5_3);
            u32 d14 = ms.shrink(w5_1 + Mod - w5_4);
            u32 d23 = ms.shrink(w5_2 + Mod - w5_3);

            c1h = ms.mul_s(s14, half_m);
            c2h = ms.mul_s(s23, half_m);
            j1h = ms.mul_s(d14, half_m);
            j2h = ms.mul_s(d23, half_m);

            // Twiddle roots: tw3_root[j] = ω_{3*2^j}, built by squaring chain
            tw3_root[k] = ms.power_s(g_full, (Mod - 1) / (u32(3) << k), ms.one);
            for (int j = k; j > 0; --j)
                tw3_root[j - 1] = ms.mul_s(tw3_root[j], tw3_root[j]);

            tw3i_root[k] = ms.power_s(tw3_root[k], Mod - 2, ms.one);
            for (int j = k; j > 0; --j)
                tw3i_root[j - 1] = ms.mul_s(tw3i_root[j], tw3i_root[j]);

            tw5_root[k] = ms.power_s(g_full, (Mod - 1) / (u32(5) << k), ms.one);
            for (int j = k; j > 0; --j)
                tw5_root[j - 1] = ms.mul_s(tw5_root[j], tw5_root[j]);

            tw5i_root[k] = ms.power_s(tw5_root[k], Mod - 2, ms.one);
            for (int j = k; j > 0; --j)
                tw5i_root[j - 1] = ms.mul_s(tw5i_root[j], tw5i_root[j]);
        }
    }

    // Compute DIT scale factor: shrink(-(M-1)/N * r3)
    constexpr u32 compute_scale(idt n_vecs) const {
        return ms.mul_s(Mod - ((Mod - 1) >> ctzll_constexpr(n_vecs)), ms.r3);
    }
};

} // namespace ntt
