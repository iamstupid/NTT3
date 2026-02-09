#pragma once
#include "../common.hpp"
#include "../simd/avx2.hpp"
#include "mont_scalar.hpp"

namespace ntt {

// Montgomery vector arithmetic, templated on SIMD backend.
// All operations mirror ref.cpp: shrk32, dilt32, add32, sub32, Ladd32, Lsub32,
// mul, mul_bsm, mul_bsmfxd, mul_bfxd, mul_upd_rt, reduce, etc.
template<typename B>
struct MontVec {
    using Vec = typename B::Vec;
    Vec vMod, vMod2, vNiv, vImg, vImgNiv;
    u32 scalar_mod, scalar_niv;

    MontVec() = default;

    MontVec(u32 mod, u32 niv, u32 img)
        : vMod(B::broadcast(mod)),
          vMod2(B::broadcast(mod * 2)),
          vNiv(B::broadcast(niv)),
          vImg(B::broadcast(img)),
          vImgNiv(B::broadcast(img * niv)),
          scalar_mod(mod), scalar_niv(niv)
    {}

    // shrink: [0, 2M) -> [0, M)  (ref: shrk32 with Mod)
    NTT_FORCEINLINE Vec shrink(Vec x) const {
        return B::min32(x, B::sub32(x, vMod));
    }

    // shrink2: [0, 4M) -> [0, 2M)  (ref: shrk32 with Mod2)
    NTT_FORCEINLINE Vec shrink2(Vec x) const {
        return B::min32(x, B::sub32(x, vMod2));
    }

    // dilate2: unsigned wrap -> [0, 2M)  (ref: dilt32 with Mod2)
    NTT_FORCEINLINE Vec dilate2(Vec x) const {
        return B::min32(x, B::add32(x, vMod2));
    }

    // add2: [0,4M) -> [0,2M)  (ref: add32 with Mod2)
    NTT_FORCEINLINE Vec add2(Vec a, Vec b) const {
        return shrink2(B::add32(a, b));
    }

    // sub2: -> [0,2M)  (ref: sub32 with Mod2)
    NTT_FORCEINLINE Vec sub2(Vec a, Vec b) const {
        return dilate2(B::sub32(a, b));
    }

    // lazy_add: a + b, no reduction  (ref: Ladd32)
    NTT_FORCEINLINE Vec lazy_add(Vec a, Vec b) const {
        return B::add32(a, b);
    }

    // lazy_sub: a + (2M - b)  (ref: Lsub32)
    NTT_FORCEINLINE Vec lazy_sub(Vec a, Vec b) const {
        return B::add32(a, B::sub32(vMod2, b));
    }

    // Montgomery reduction from (even64, odd64) pairs  (ref: reduce)
    NTT_FORCEINLINE Vec mont_reduce(Vec even, Vec odd) const {
        Vec ce = B::mul64(even, vNiv);
        Vec co = B::mul64(odd, vNiv);
        ce = B::mul64(ce, vMod);
        co = B::mul64(co, vMod);
        return B::blend_0xaa(
            B::srl64(B::add64(even, ce), 32),
            B::add64(odd, co)
        );
    }

    // Generic Montgomery mul: a * b  (ref: mul)
    NTT_FORCEINLINE Vec mont_mul(Vec a, Vec b) const {
        Vec even = B::mul64(a, b);
        Vec odd = B::mul64(B::srl64(a, 32), B::srl64(b, 32));
        return mont_reduce(even, odd);
    }

    // Montgomery mul where b is broadcast scalar  (ref: mul_bsm)
    // b has same value in all 32-bit lanes; odd lanes of b are copies of even
    NTT_FORCEINLINE Vec mont_mul_bsm(Vec a, Vec b) const {
        Vec even = B::mul64(a, b);
        Vec odd = B::mul64(B::srl64(a, 32), b);
        return mont_reduce(even, odd);
    }

    // Montgomery mul with precomputed (b, b*niv) as broadcast scalar  (ref: mul_bsmfxd)
    // b is broadcast, bniv = b * niv (also broadcast)
    NTT_FORCEINLINE Vec mont_mul_precomp(Vec a, Vec b, Vec bniv) const {
        Vec ce = B::mul64(a, bniv);
        Vec co = B::mul64(B::srl64(a, 32), bniv);
        Vec pe = B::mul64(a, b);
        Vec po = B::mul64(B::srl64(a, 32), b);
        ce = B::mul64(ce, vMod);
        co = B::mul64(co, vMod);
        return B::blend_0xaa(
            B::srl64(B::add64(pe, ce), 32),
            B::add64(po, co)
        );
    }

    // Montgomery mul with precomputed (b, bniv) where b/bniv have different
    // values in even and odd lanes  (ref: mul_bfxd)
    NTT_FORCEINLINE Vec mont_mul_precomp_full(Vec a, Vec b, Vec bniv) const {
        Vec ce = B::mul64(a, bniv);
        Vec co = B::mul64(B::srl64(a, 32), B::srl64(bniv, 32));
        Vec pe = B::mul64(a, b);
        Vec po = B::mul64(B::srl64(a, 32), B::srl64(b, 32));
        ce = B::mul64(ce, vMod);
        co = B::mul64(co, vMod);
        return B::blend_0xaa(
            B::srl64(B::add64(pe, ce), 32),
            B::add64(po, co)
        );
    }

    // Multiply by img (omega_4)  (ref: mul_bsmfxd with Img, ImgNiv)
    NTT_FORCEINLINE Vec mul_by_img(Vec a) const {
        return mont_mul_precomp(a, vImg, vImgNiv);
    }

    // Root update multiply  (ref: mul_upd_rt)
    // bu layout: 64-bit pairs of {val*niv (lo32), val (hi32)}
    // Returns shrink'd result (only even-lane products, broadcast to all)
    NTT_FORCEINLINE Vec mul_upd_root(Vec a, Vec bu) const {
        Vec bniv = bu;                        // even lanes: val*niv
        Vec b = B::srl64(bu, 32);             // even lanes: val
        Vec ce = B::mul64(a, bniv);
        Vec pe = B::mul64(a, b);
        ce = B::mul64(ce, vMod);
        return shrink(B::srl64(B::add64(pe, ce), 32));
    }

    // Root update multiply keeping full even+odd  (ref: mul_upd_rr -> mul_bsmfxd)
    NTT_FORCEINLINE Vec mul_upd_root_full(Vec a, Vec bu) const {
        Vec b = B::srl64(bu, 32);
        return mont_mul_precomp(a, b, bu);
    }
};

} // namespace ntt
