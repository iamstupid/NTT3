#pragma once
#include "../common.hpp"

namespace ntt {

struct MontScalar {
    u32 mod, mod2, niv, one, r2, r3;

    constexpr MontScalar(u32 m) :
        mod(m), mod2(m * 2),
        niv(compute_niv(m)),
        one(u32(-m) % m),            // (2^32 - m) % m = 2^32 mod m = R mod m
        r2(u32(u64(-u64(m)) % m)),   // (2^64 - m) % m = 2^64 mod m = R^2 mod m
        r3(0)
    {
        r3 = mul_s(r2, r2);          // R^2 * R^2 / R mod m = R^3 mod m
    }

    // shrink: [0, 2M) -> [0, M)
    constexpr u32 shrink(u32 x) const {
        return (x >= mod) ? (x - mod) : x;
    }

    // dilate: unsigned repr of (-M, M) -> [0, M)
    // min(x, x+mod): if x was small (positive), x+mod > x so result = x
    // if x was large (wrapping negative), x+mod wraps to small, result = x+mod
    constexpr u32 dilate(u32 x) const {
        u32 y = x + mod;
        return (x < y) ? x : y;
    }

    constexpr u32 reduce(u64 x) const {
        return (u32)((x + u64(u32(x) * niv) * mod) >> 32);
    }

    constexpr u32 mul(u32 a, u32 b) const {
        return reduce(u64(a) * b);
    }

    constexpr u32 mul_s(u32 a, u32 b) const {
        u32 r = mul(a, b);
        return (r >= mod) ? (r - mod) : r;
    }

    constexpr u32 power(u32 a, u32 b, u32 r) const {
        for (; b; b >>= 1, a = mul(a, a)) {
            if (b & 1) r = mul(r, a);
        }
        return r;
    }

    constexpr u32 power_s(u32 a, u32 b, u32 r) const {
        u32 res = power(a, b, r);
        return (res >= mod) ? (res - mod) : res;
    }

    constexpr u32 to_mont(u32 x) const {
        return mul(x, r2);
    }

    constexpr u32 from_mont(u32 x) const {
        u32 r = reduce(x);
        return (r >= mod) ? (r - mod) : r;
    }

private:
    static constexpr u32 compute_niv(u32 m) {
        u32 n = 2 + m;
        for (int i = 0; i < 4; ++i) {
            n *= 2 + m * n;
        }
        return n;
    }
};

} // namespace ntt
