#pragma once
#include "common.hpp"
#include "simd/avx2.hpp"
#include "p30x3/mont_scalar.hpp"
#include "p30x3/mont_vec.hpp"
#include "p30x3/root_plan.hpp"
#include "p30x3/radix4.hpp"
#include "p30x3/radix2.hpp"
#include "p30x3/cyclic_conv.hpp"
#include "p30x3/scheduler.hpp"
#include "p30x3/crt.hpp"
#include "arena.hpp"
#include "profile.hpp"
#include "p50x4/multiply.hpp"
#include <algorithm>
#include <cstring>

namespace ntt {

// Reduce src[0..src_len) to [0, 2*Mod) and zero-pad buf[src_len..N).
// Single-pass: loadu from src, reduce, store to aligned buf. One memory pass.
template<typename B, u32 Mod>
inline void reduce_and_pad(u32* buf, const u32* src, idt src_len, idt N) {
    using Vec = typename B::Vec;

    constexpr u64 MAX32 = 0xFFFFFFFFULL;
    constexpr u32 Mod2 = u32(2ULL * Mod);
    constexpr u32 Mod4 = (4ULL * Mod <= MAX32) ? u32(4ULL * Mod) : 0;
    constexpr u32 Mod8 = (8ULL * Mod <= MAX32) ? u32(8ULL * Mod) : 0;

    const Vec vMod2 = B::broadcast(Mod2);
    [[maybe_unused]] const Vec vMod4 = B::broadcast(Mod4);
    [[maybe_unused]] const Vec vMod8 = B::broadcast(Mod8);

    const idt nvecs = src_len / B::LANES;
    idt j = 0;
    for (; j + 4 <= nvecs; j += 4) {
        Vec x0 = B::loadu(src + (j) * B::LANES);
        Vec x1 = B::loadu(src + (j + 1) * B::LANES);
        Vec x2 = B::loadu(src + (j + 2) * B::LANES);
        Vec x3 = B::loadu(src + (j + 3) * B::LANES);
        if constexpr (Mod8 != 0) {
            x0 = B::min32(x0, B::sub32(x0, vMod8));
            x1 = B::min32(x1, B::sub32(x1, vMod8));
            x2 = B::min32(x2, B::sub32(x2, vMod8));
            x3 = B::min32(x3, B::sub32(x3, vMod8));
        }
        if constexpr (Mod4 != 0) {
            x0 = B::min32(x0, B::sub32(x0, vMod4));
            x1 = B::min32(x1, B::sub32(x1, vMod4));
            x2 = B::min32(x2, B::sub32(x2, vMod4));
            x3 = B::min32(x3, B::sub32(x3, vMod4));
        }
        x0 = B::min32(x0, B::sub32(x0, vMod2));
        x1 = B::min32(x1, B::sub32(x1, vMod2));
        x2 = B::min32(x2, B::sub32(x2, vMod2));
        x3 = B::min32(x3, B::sub32(x3, vMod2));
        Vec* d = (Vec*)buf + j;
        B::store(d, x0);
        B::store(d + 1, x1);
        B::store(d + 2, x2);
        B::store(d + 3, x3);
    }
    for (; j < nvecs; ++j) {
        Vec x = B::loadu(src + j * B::LANES);
        if constexpr (Mod8 != 0) x = B::min32(x, B::sub32(x, vMod8));
        if constexpr (Mod4 != 0) x = B::min32(x, B::sub32(x, vMod4));
        x = B::min32(x, B::sub32(x, vMod2));
        B::store((Vec*)buf + j, x);
    }
    // Scalar tail
    for (idt i = nvecs * B::LANES; i < src_len; ++i) {
        u32 v = src[i];
        if constexpr (Mod8 != 0) { if (v >= Mod8) v -= Mod8; }
        if constexpr (Mod4 != 0) { if (v >= Mod4) v -= Mod4; }
        if (v >= Mod2) v -= Mod2;
        buf[i] = v;
    }
    std::memset(buf + src_len, 0, (N - src_len) * sizeof(u32));
}

// Run NTT convolution for one prime: forward, multiply, inverse
template<typename B, u32 Mod>
inline void ntt_conv_one_prime(
    u32* f, u32* g, idt ntt_vecs,
    const u32* a, idt na, const u32* b, idt nb, idt N)
{
    using Vec = typename B::Vec;
    using S = NTTScheduler<B, Mod>;

    {
        ProfileScope ps(&profile_counters().api_reduce_pad_ns);
        reduce_and_pad<B, Mod>(f, a, na, N);
        reduce_and_pad<B, Mod>(g, b, nb, N);
    }
    {
        ProfileScope ps(&profile_counters().api_forward_ns);
        S::forward((Vec*)f, ntt_vecs);
        S::forward((Vec*)g, ntt_vecs);
    }
    {
        ProfileScope ps(&profile_counters().api_freqmul_ns);
        S::freq_multiply((Vec*)f, (Vec*)g, ntt_vecs);
    }
    {
        ProfileScope ps(&profile_counters().api_inverse_ns);
        S::inverse((Vec*)f, ntt_vecs);
    }
}

// Three-prime NTT-based big integer multiplication.
// Input: a[0..na), b[0..nb) are arrays of u32 limbs (base 2^32).
// Output: out[0..out_len) is the product (at least na+nb limbs needed).
inline void big_multiply(
    u32* out, idt out_len,
    const u32* a, idt na,
    const u32* b, idt nb)
{
    ProfileScope ps_total(&profile_counters().api_total_ns);

    using B = Avx2;
    using Vec = typename B::Vec;

    const idt min_len = na + nb;
    idt N = ceil_smooth(min_len > 64 ? min_len : 64);
    idt ntt_vecs = N / B::LANES;
    if (ntt_vecs < 8) { ntt_vecs = 8; N = ntt_vecs * B::LANES; }

    // Pool: 4 buffers (3 f-buffers for CRT + 1 shared g-buffer)
    NTTArena& arena = NTTArena::instance();

    // Tagged pointers (2 bits encode bin offset for recycling)
    Vec* f0 = arena.alloc<Vec>(ntt_vecs);
    Vec* f1 = arena.alloc<Vec>(ntt_vecs);
    Vec* f2 = arena.alloc<Vec>(ntt_vecs);
    Vec* g  = arena.alloc<Vec>(ntt_vecs);

    // Raw pointers for computation
    auto* rf0 = NTTArena::raw(f0);
    auto* rf1 = NTTArena::raw(f1);
    auto* rf2 = NTTArena::raw(f2);
    auto* rg  = NTTArena::raw(g);

    ntt_conv_one_prime<B, CRT_P0>((u32*)rf0, (u32*)rg, ntt_vecs, a, na, b, nb, N);
    ntt_conv_one_prime<B, CRT_P1>((u32*)rf1, (u32*)rg, ntt_vecs, a, na, b, nb, N);
    ntt_conv_one_prime<B, CRT_P2>((u32*)rf2, (u32*)rg, ntt_vecs, a, na, b, nb, N);

    idt result_len = (std::min)(min_len, out_len);
    {
        ProfileScope ps(&profile_counters().api_crt_ns);
        crt_and_propagate(out, result_len, (u32*)rf0, (u32*)rf1, (u32*)rf2);
    }

    // Return tagged pointers to arena (tag tells it the actual bin)
    arena.dealloc(g,  ntt_vecs);
    arena.dealloc(f2, ntt_vecs);
    arena.dealloc(f1, ntt_vecs);
    arena.dealloc(f0, ntt_vecs);
}

// Max p30x3 NTT size in u32 elements: 3*2^22 = 12582912
static constexpr idt P30X3_MAX_NTT = 12582912;

// Big integer multiplication (u64 limbs, base 2^64).
// Input: a[0..na), b[0..nb) are arrays of u64 limbs (little-endian).
// Output: out[0..out_len) is the product (at least na+nb limbs needed).
// Dispatches to p30x3 (faster, 3-prime u32) when NTT size fits, else p50x4.
inline void big_multiply_u64(
    u64* out, idt out_len,
    const u64* a, idt na,
    const u64* b, idt nb)
{
    // u64 little-endian in memory is already a u32 stream — just cast.
    idt na32 = 2 * na, nb32 = 2 * nb, out32 = 2 * out_len;
    idt ntt_size = ceil_smooth(na32 + nb32 > 64 ? na32 + nb32 : 64);

    if (ntt_size <= P30X3_MAX_NTT) {
        big_multiply((u32*)out, out32, (const u32*)a, na32, (const u32*)b, nb32);
    } else {
        p50x4::Ntt4& engine = p50x4::Ntt4::instance();
        engine.multiply(out, static_cast<std::size_t>(out_len),
                        a, static_cast<std::size_t>(na),
                        b, static_cast<std::size_t>(nb));
    }
}

} // namespace ntt
