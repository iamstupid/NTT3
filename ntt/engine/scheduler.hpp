#pragma once
#include "../common.hpp"
#include "../mont/mont_vec.hpp"
#include "../roots/root_plan.hpp"
#include "../kernels/radix4.hpp"
#include "../kernels/radix2.hpp"
#include "../kernels/radix3.hpp"
#include "../kernels/radix5.hpp"
#include "../kernels/cyclic_conv.hpp"
#include <array>
#include <algorithm>

namespace ntt {

// NTTScheduler: cache-oblivious NTT engine.
// Implements j-based traversal with ruler-sequence root updates.
// Direct structural port of __vec_dif / __vec_dit / __vec_cvdt8 from ref.cpp.
template<typename B, u32 Mod>
struct NTTScheduler {
    using Vec = typename B::Vec;

    // Runtime-initialized singletons (avoid constexpr issues with MSVC)
    static const RootPlan<Mod>& get_roots() {
        static const RootPlan<Mod> r{};
        return r;
    }
    static const MontScalar& get_ms() {
        static constexpr MontScalar ms{Mod};
        return ms;
    }

    // Base-2 forward NTT (DIF), power-of-2 sizes only.
    static void fwd_b2(Vec* f, idt n) {
        const auto& roots = get_roots();
        const auto& ms = get_ms();

        alignas(32) u32 st_1_raw[(MAX_LOG >> 1) * B::LANES];
        // View as array of 8-u32 blocks
        auto st_1 = [&](int idx) -> u32* { return st_1_raw + idx * B::LANES; };

        const MontVec<B> m(ms.mod, ms.niv, roots.img);
        const Vec Niv = B::broadcast(ms.niv);
        const Vec id24 = B::setr(0, 2, 0, 4, 0, 2, 0, 4);

        const int lgn = ntt_ctzll(n);

        // Fill ruler sequence state with initial root state
        for (int i = 0; i < (lgn >> 1); ++i) {
            for (int j = 0; j < B::LANES; ++j)
                st_1(i)[j] = roots.bwb[j];
        }

        const idt nn = n >> (lgn & 1);
        const idt blk = (std::min)(n, BLOCK_SIZE);

        // Phase 1: Optional radix-2 pass for odd lgn
        if (nn != n) {
            Radix2Kernel<B>::dif_pass(f, nn, m);
        }

        // Phase 2: Pure butterfly chain (r=0 path, no twiddle)
        for (idt L = nn >> 2; L > 0; L >>= 2) {
            Radix4Kernel<B>::dif_butterfly_notw(f, f + L, f + 2 * L, f + 3 * L, L, m);
        }

        // Phase 3: j-based cache-oblivious blocked traversal
        int t = (std::min)((int)LOG_BLOCK, lgn) & ~1;
        int p = (t - 2) >> 1;

        for (idt j = 0; j < n; j += blk, t = ntt_ctzll(j) & ~1, p = (t - 2) >> 1) {
            Vec* const g = f + j;

            for (idt l = (idt(1) << t), L = l >> 2; L; l = L, L >>= 2, t -= 2, --p) {
                Vec rt = B::load(st_1(p));

                for (idt i = (j == 0 ? l : 0), k = (j + i) >> t; i < blk; i += l, ++k) {
                    // Extract twiddles from ruler-sequence state
                    Vec r1 = B::permutevar(rt, id24);
                    Vec r1_niv = B::permutevar(B::mul64(rt, Niv), id24);
                    rt = m.mul_upd_root(rt,
                        B::load(roots.rt3[ntt_ctzll(~(unsigned)k)]));

                    Vec r2 = B::shuffle_BBBB(r1);
                    Vec nr3 = B::shuffle_DDDD(r1);
                    Vec r2_niv = B::shuffle_BBBB(r1_niv);
                    Vec nr3_niv = B::shuffle_DDDD(r1_niv);

                    Radix4Kernel<B>::dif_butterfly(
                        g + i, g + i + L, g + i + 2 * L, g + i + 3 * L,
                        L, r1, r1_niv, r2, r2_niv, nr3, nr3_niv, m);
                }

                B::store(st_1(p), rt);
            }
        }
    }

    // Base-2 inverse NTT (DIT), power-of-2 sizes only.
    static void inv_b2(Vec* f, idt n) {
        const auto& roots = get_roots();
        const auto& ms = get_ms();

        alignas(32) u32 st_1_raw[(MAX_LOG >> 1) * B::LANES];
        auto st_1 = [&](int idx) -> u32* { return st_1_raw + idx * B::LANES; };

        const MontVec<B> m(ms.mod, ms.niv, roots.img);
        const Vec Niv = B::broadcast(ms.niv);
        const Vec id24 = B::setr(0, 2, 0, 4, 0, 2, 0, 4);

        const int lgn = ntt_ctzll(n);

        // Fill inverse root state (skip index 0, it gets scale factor)
        for (int i = 1; i < (lgn >> 1); ++i) {
            for (int j = 0; j < B::LANES; ++j)
                st_1(i)[j] = roots.bwbi[j];
        }

        const idt nn = n >> (lgn & 1);
        const idt blk = (std::min)(n, BLOCK_SIZE);

        // Compute N^{-1} scale factor
        const u32 fx = roots.compute_scale(n);
        const Vec Fx = B::broadcast(fx);
        const Vec FxNiv = B::broadcast(fx * ms.niv);
        B::store(st_1(0), Fx);

        // j-based blocked traversal (DIT)
        for (idt j = 0; j < n; j += blk) {
            int tt = ntt_ctzll(j + blk);
            int t = 4, p = 1;

            // Innermost layer (L=1): scale fusion
            {
                Vec rt = B::load(st_1(0));
                for (idt i = j; i < j + blk; i += 4) {
                    Vec r1 = B::permutevar(rt, id24);
                    rt = m.mul_upd_root(rt,
                        B::load(roots.rt3i[ntt_ctzll(~(unsigned)(i >> 2))]));

                    Vec r2 = B::shuffle_BBBB(r1);
                    Vec r3 = B::shuffle_DDDD(r1);

                    Radix4Kernel<B>::dit_butterfly_scale(
                        f + i, f + i + 1, f + i + 2, f + i + 3,
                        r1, r2, r3, Fx, FxNiv, m);
                }
                B::store(st_1(0), rt);
            }

            // Outer layers
            for (idt l = 16, L = 4; t <= tt; L = l, l <<= 2, t += 2, ++p) {
                idt diff = j + blk - (std::max)(l, blk);
                idt i = 0;
                Vec rt = B::load(st_1(p));
                Vec* const g = f + diff;

                if (diff == 0) {
                    bool is_outermost = (l == n);
                    Radix4Kernel<B>::dit_butterfly_notw(
                        f, f + L, f + 2 * L, f + 3 * L, L, m, is_outermost);
                    i = l;
                }

                for (idt k = (j + i) >> t; i < blk; i += l, ++k) {
                    Vec r1 = B::permutevar(rt, id24);
                    Vec r1_niv = B::permutevar(B::mul64(rt, Niv), id24);
                    rt = m.mul_upd_root(rt,
                        B::load(roots.rt3i[ntt_ctzll(~(unsigned)k)]));

                    Vec r2 = B::shuffle_BBBB(r1);
                    Vec r3 = B::shuffle_DDDD(r1);
                    Vec r2_niv = B::shuffle_BBBB(r1_niv);
                    Vec r3_niv = B::shuffle_DDDD(r1_niv);

                    Radix4Kernel<B>::dit_butterfly(
                        g + i, g + i + L, g + i + 2 * L, g + i + 3 * L,
                        L, r1, r1_niv, r2, r2_niv, r3, r3_niv, m);
                }

                B::store(st_1(p), rt);
            }
        }

        // Optional radix-2 pass for odd lgn
        if (nn != n) {
            Radix2Kernel<B>::dit_pass(f, nn, m);
        }

        // Final reduction: ensure all elements are in [0, M)
        // The DIT may leave values in [0, 2M) when the outermost layer
        // is the innermost (e.g., nvecs=4) or certain size configurations.
        {
            const Vec vMod = B::broadcast(ms.mod);
            for (idt i = 0; i < n; ++i) {
                Vec v = B::load(f + i);
                B::store(f + i, B::min32(v, B::sub32(v, vMod)));
            }
        }
    }

    // ── Mixed-radix dispatch ──
    // n = m * 2^k where m ∈ {1, 3, 5}

    // Forward NTT (DIF): outer radix-m pass, then fwd_b2 on each sub-array.
    static void forward(Vec* f, idt n) {
        const int k = ntt_ctzll(n);
        const idt m = n >> k;
        if (m == 1) { fwd_b2(f, n); return; }

        const auto& roots = get_roots();
        const auto& ms_s = get_ms();
        const MontVec<B> mv(ms_s.mod, ms_s.niv, roots.img);

        if (m == 3) Radix3Kernel<B, Mod>::dif_pass(f, n, mv, roots);
        else        Radix5Kernel<B, Mod>::dif_pass(f, n, mv, roots);

        const idt sub_n = idt(1) << k;
        for (idt i = 0; i < m; ++i)
            fwd_b2(f + i * sub_n, sub_n);
    }

    // Inverse NTT (DIT): inv_b2 on each sub-array, then outer radix-m DIT pass.
    static void inverse(Vec* f, idt n) {
        const int k = ntt_ctzll(n);
        const idt m = n >> k;
        if (m == 1) { inv_b2(f, n); return; }

        const idt sub_n = idt(1) << k;
        for (idt i = 0; i < m; ++i)
            inv_b2(f + i * sub_n, sub_n);

        const auto& roots = get_roots();
        const auto& ms_s = get_ms();
        const MontVec<B> mv(ms_s.mod, ms_s.niv, roots.img);

        if (m == 3) Radix3Kernel<B, Mod>::dit_pass(f, n, mv, roots);
        else        Radix5Kernel<B, Mod>::dit_pass(f, n, mv, roots);
    }

    // Frequency-domain multiply (twisted convolution on each sub-array).
    static void freq_multiply(Vec* f, Vec* g, idt n) {
        const auto& roots = get_roots();
        const auto& ms_s = get_ms();

        const int k = ntt_ctzll(n);
        const idt m = n >> k;
        const idt sub_n = idt(1) << k;

        if (m == 1) {
            CyclicConv<B>::twisted_conv(f, g, n, ms_s, roots.img, roots.RT3);
            return;
        }

        // For mixed-radix: sub-array r needs twist offset ω_N^r where N = n vecs.
        // ω_N = tw{m}_root[k] (primitive N-th root of unity).
        const u32 omega_N = (m == 3) ? roots.tw3_root[k] : roots.tw5_root[k];
        u32 rr = ms_s.one;  // ω_N^0 = 1 for sub-array 0
        for (idt i = 0; i < m; ++i) {
            CyclicConv<B>::twisted_conv(
                f + i * sub_n, g + i * sub_n, sub_n,
                ms_s, roots.img, roots.RT3, rr);
            rr = ms_s.mul_s(rr, omega_N);
        }
    }
};

} // namespace ntt
