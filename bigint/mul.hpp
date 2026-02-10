#pragma once
// mul.hpp - Multiplication algorithms for arbitrary-precision integers
//
// Dispatch chain: basecase (O(n^2)) -> Karatsuba (O(n^1.585)) -> NTT (O(n log n))
// NTT bridge uses ntt/p50x4 engine via ntt::big_multiply_u64().

#include "mpn.hpp"
#include <cstring>
#include <algorithm>

// NTT bridge: include the NTT API for large multiplications
#include "../ntt/api.hpp"

namespace bi {

// ============================================================
// Tuning thresholds (determined by benchmarking)
// ============================================================

// Below this, use schoolbook basecase
static constexpr uint32_t KARATSUBA_THRESHOLD = 32;

// Below this, use Karatsuba; above, use NTT
static constexpr uint32_t NTT_THRESHOLD = 1024;

// For squaring, thresholds can be different (squaring is ~1.5x faster)
static constexpr uint32_t SQR_KARATSUBA_THRESHOLD = 40;
static constexpr uint32_t SQR_NTT_THRESHOLD = 1024;

// ============================================================
// Basecase multiplication (schoolbook)
// ============================================================

// rp[0..an+bn) = ap[0..an) * bp[0..bn)
// Precondition: an >= bn > 0; rp does NOT alias ap or bp
inline void mpn_mul_basecase(limb_t* rp, const limb_t* ap, uint32_t an,
                              const limb_t* bp, uint32_t bn)
{
    // First row: rp = ap * bp[0]
    rp[an] = mpn_mul_1(rp, ap, an, bp[0]);

    // Remaining rows: rp += ap * bp[j], shifted by j
    for (uint32_t j = 1; j < bn; j++) {
        rp[j + an] = mpn_addmul_1(rp + j, ap, an, bp[j]);
    }
}

// ============================================================
// Basecase squaring (exploits symmetry)
// ============================================================

// rp[0..2*n) = ap[0..n)^2
// Uses the identity: a^2 = sum_i(a[i]^2 * B^(2i)) + 2*sum_{i<j}(a[i]*a[j]*B^(i+j))
inline void mpn_sqr_basecase(limb_t* rp, const limb_t* ap, uint32_t n) {
    if (n == 1) {
        rp[0] = umul_hilo(ap[0], ap[0], &rp[1]);
        return;
    }

    // Compute off-diagonal products (upper triangle only)
    // rp = ap[0] * ap[1..n)
    rp[n] = mpn_mul_1(rp + 1, ap + 1, n - 1, ap[0]);
    rp[0] = 0; // will be filled by diagonal

    // Add remaining off-diagonal products
    for (uint32_t i = 1; i < n - 1; i++) {
        rp[i + n] = mpn_addmul_1(rp + 2 * i + 1, ap + i + 1, n - i - 1, ap[i]);
    }
    rp[2 * n - 1] = 0;

    // Double the off-diagonal part (left shift by 1 bit)
    limb_t carry = mpn_lshift(rp + 1, rp + 1, 2 * n - 2, 1);
    rp[2 * n - 1] = carry;

    // Add diagonal: rp[2*i..2*i+1] += ap[i]^2
    unsigned char c = 0;
    for (uint32_t i = 0; i < n; i++) {
        limb_t hi, lo;
        lo = umul_hilo(ap[i], ap[i], &hi);
        c = _addcarry_u64(c, rp[2 * i], lo, (unsigned long long*)&rp[2 * i]);
        c = _addcarry_u64(c, rp[2 * i + 1], hi, (unsigned long long*)&rp[2 * i + 1]);
    }
}

// ============================================================
// Karatsuba multiplication
// ============================================================

// Scratch space needed: about 4*n limbs for the deepest recursion level
// (each level needs ~2n scratch, geometric series converges to ~4n)

// Internal Karatsuba: rp[0..2*n) = ap[0..n) * bp[0..n)
// scratch must have at least 4*n limbs available
static void mpn_mul_karatsuba_n(limb_t* rp, const limb_t* ap, const limb_t* bp,
                                 uint32_t n, limb_t* scratch)
{
    if (n < KARATSUBA_THRESHOLD) {
        mpn_mul_basecase(rp, ap, n, bp, n);
        return;
    }

    uint32_t m = n / 2;        // low half size
    uint32_t h = n - m;        // high half size (h >= m)

    // Split: a = a1*B^m + a0,  b = b1*B^m + b0
    const limb_t* a0 = ap;
    const limb_t* a1 = ap + m;
    const limb_t* b0 = bp;
    const limb_t* b1 = bp + m;

    // Step 1: z0 = a0 * b0 (stored in rp[0..2m))
    mpn_mul_karatsuba_n(rp, a0, b0, m, scratch);

    // Step 2: z2 = a1 * b1 (stored in rp[2m..2n))
    if (h == m) {
        mpn_mul_karatsuba_n(rp + 2 * m, a1, b1, h, scratch);
    } else {
        // h = m+1 when n is odd; use basecase or recurse
        if (h < KARATSUBA_THRESHOLD) {
            mpn_mul_basecase(rp + 2 * m, a1, h, b1, h);
        } else {
            mpn_mul_karatsuba_n(rp + 2 * m, a1, b1, h, scratch);
        }
    }

    // Step 3: Compute |a0 - a1| and |b0 - b1|, track signs
    // t0 = |a0 - a1| in scratch[0..h)
    // t1 = |b0 - b1| in scratch[h..2h)
    limb_t* t0 = scratch;
    limb_t* t1 = scratch + h;
    limb_t* t2 = scratch + 2 * h; // scratch for recursive mul: at least 2h limbs

    int sign_a, sign_b;

    // Compare a0 (m limbs) vs a1 (h limbs)
    if (h > m) {
        // a1 has one more limb; if top limb nonzero, a1 > a0
        if (a1[m] != 0) {
            sign_a = -1; // a0 < a1
        } else {
            int cmp = mpn_cmp(a0, a1, m);
            sign_a = cmp;
        }
    } else {
        sign_a = mpn_cmp(a0, a1, m);
    }

    if (sign_a >= 0) {
        // a0 >= a1: t0 = a0 - a1
        mpn_zero(t0, h);
        mpn_copyi(t0, a0, m);
        mpn_sub(t0, t0, h, a1, h);
    } else {
        // a0 < a1: t0 = a1 - a0
        mpn_copyi(t0, a1, h);
        mpn_sub(t0, t0, h, a0, m);
    }

    // Same for b
    if (h > m) {
        if (b1[m] != 0) {
            sign_b = -1;
        } else {
            int cmp = mpn_cmp(b0, b1, m);
            sign_b = cmp;
        }
    } else {
        sign_b = mpn_cmp(b0, b1, m);
    }

    if (sign_b >= 0) {
        mpn_zero(t1, h);
        mpn_copyi(t1, b0, m);
        mpn_sub(t1, t1, h, b1, h);
    } else {
        mpn_copyi(t1, b1, h);
        mpn_sub(t1, t1, h, b0, m);
    }

    // Step 4: t2 = |a0-a1| * |b0-b1| (in t2[0..2h))
    uint32_t t0_n = mpn_normalize(t0, h);
    uint32_t t1_n = mpn_normalize(t1, h);

    if (t0_n == 0 || t1_n == 0) {
        mpn_zero(t2, 2 * h);
    } else if (t0_n < KARATSUBA_THRESHOLD || t1_n < KARATSUBA_THRESHOLD) {
        mpn_zero(t2, 2 * h);
        if (t0_n >= t1_n)
            mpn_mul_basecase(t2, t0, t0_n, t1, t1_n);
        else
            mpn_mul_basecase(t2, t1, t1_n, t0, t0_n);
    } else {
        // Pad to same size for recursive Karatsuba
        uint32_t tn = (t0_n > t1_n) ? t0_n : t1_n;
        mpn_zero(t0 + t0_n, tn - t0_n);
        mpn_zero(t1 + t1_n, tn - t1_n);
        mpn_mul_karatsuba_n(t2, t0, t1, tn, t2 + 2 * tn);
    }

    // Step 5: Recombination
    // z1 = z0 + z2 - (a0-a1)*(b0-b1) = z0 + z2 - sign_a * sign_b * t2
    // If sign_a * sign_b > 0: z1 = z0 + z2 - t2
    // If sign_a * sign_b < 0: z1 = z0 + z2 + t2
    //
    // rp currently: [z0 (2m limbs)] [z2 (2h limbs)]
    // We need: rp = z0 + z1*B^m + z2*B^(2m)
    //
    // To avoid aliasing bugs (rp[0..2m) overlaps rp[m..] target),
    // compute z1 entirely in scratch[0..2h), then add it to rp[m..].
    // At this point scratch[0..2h) is free (t0/t1 no longer needed)
    // and t2 is at scratch[2h..4h).

    // z1 = z0 + z2 in scratch[0..2h), with carry in z1_hi
    std::memcpy(scratch, rp, 2 * m * sizeof(limb_t)); // copy z0
    if (h > m)
        mpn_zero(scratch + 2 * m, 2 * h - 2 * m); // zero-extend z0 to 2h limbs
    limb_t z1_hi = mpn_add_n(scratch, scratch, rp + 2 * m, 2 * h); // += z2

    // ± t2 (t2 is at scratch[2h..], non-overlapping with scratch[0..2h))
    bool subtract_t2 = (sign_a > 0 && sign_b > 0) || (sign_a < 0 && sign_b < 0);
    uint32_t t2_n2 = mpn_normalize(t2, 2 * h);
    if (t2_n2 > 0) {
        if (subtract_t2) {
            limb_t borrow = mpn_sub(scratch, scratch, 2 * h, t2, t2_n2);
            z1_hi -= borrow; // safe: z1 = a0*b1 + a1*b0 >= 0
        } else {
            limb_t c = mpn_add(scratch, scratch, 2 * h, t2, t2_n2);
            z1_hi += c;
        }
    }

    // Add z1 (scratch[0..2h) + z1_hi) to rp[m..2n)
    limb_t carry = mpn_add(rp + m, rp + m, 2 * n - m, scratch, 2 * h);
    carry += z1_hi;
    if (carry && m + 2 * h < 2 * n) {
        mpn_add_1v(rp + m + 2 * h, rp + m + 2 * h, 2 * n - m - 2 * h, (limb_t)carry);
    }
}

// General unbalanced Karatsuba: an >= bn
static void mpn_mul_karatsuba(limb_t* rp, const limb_t* ap, uint32_t an,
                               const limb_t* bp, uint32_t bn, limb_t* scratch)
{
    if (an == bn) {
        mpn_mul_karatsuba_n(rp, ap, bp, an, scratch);
        return;
    }

    // Unbalanced: slice 'a' into chunks of size bn and multiply each
    mpn_zero(rp, an + bn);

    uint32_t pos = 0;
    while (pos + bn <= an) {
        // Multiply ap[pos..pos+bn) * bp[0..bn)
        // Use scratch for product + recursive scratch
        limb_t* prod = scratch;
        limb_t* rec_scratch = scratch + 2 * bn;

        mpn_mul_karatsuba_n(prod, ap + pos, bp, bn, rec_scratch);

        // Add to result at rp[pos..]
        limb_t carry = mpn_add(rp + pos, rp + pos, an + bn - pos, prod, 2 * bn);
        (void)carry;
        pos += bn;
    }

    // Handle remaining partial block
    uint32_t rem = an - pos;
    if (rem > 0) {
        uint32_t prod_n = rem + bn;
        uint32_t larger = (rem >= bn) ? rem : bn;
        uint32_t smaller = (rem >= bn) ? bn : rem;
        limb_t* prod = scratch;
        limb_t* rec_scratch = scratch + prod_n;

        if (smaller < KARATSUBA_THRESHOLD) {
            if (rem >= bn)
                mpn_mul_basecase(prod, ap + pos, rem, bp, bn);
            else
                mpn_mul_basecase(prod, bp, bn, ap + pos, rem);
        } else {
            if (rem >= bn)
                mpn_mul_karatsuba(prod, ap + pos, rem, bp, bn, rec_scratch);
            else
                mpn_mul_karatsuba(prod, bp, bn, ap + pos, rem, rec_scratch);
        }

        limb_t carry = mpn_add(rp + pos, rp + pos, an + bn - pos, prod, prod_n);
        (void)carry;
    }
}

// ============================================================
// NTT multiplication (bridge to ntt/p50x4)
// ============================================================

inline void mpn_mul_ntt(limb_t* rp, const limb_t* ap, uint32_t an,
                         const limb_t* bp, uint32_t bn)
{
    ntt::big_multiply_u64(rp, (ntt::idt)(an + bn), ap, (ntt::idt)an, bp, (ntt::idt)bn);
}

// ============================================================
// Top-level multiply dispatch
// ============================================================

// rp[0..an+bn) = ap[0..an) * bp[0..bn)
// Precondition: an >= bn > 0; rp does not alias ap or bp
inline void mpn_mul(limb_t* rp, const limb_t* ap, uint32_t an,
                     const limb_t* bp, uint32_t bn)
{
    assert(an >= bn && bn > 0);

    if (bn < KARATSUBA_THRESHOLD) {
        mpn_mul_basecase(rp, ap, an, bp, bn);
    } else if (an >= NTT_THRESHOLD) {
        // Use NTT when the larger operand is NTT-sized
        mpn_mul_ntt(rp, ap, an, bp, bn);
    } else if (bn < NTT_THRESHOLD) {
        uint32_t mx = an > bn ? an : bn;
        uint32_t scratch_n = 6 * mx + 128;
        limb_t* scratch = mpn_alloc(scratch_n);
        mpn_mul_karatsuba(rp, ap, an, bp, bn, scratch);
        mpn_free(scratch);
    } else {
        mpn_mul_ntt(rp, ap, an, bp, bn);
    }
}

// Squaring: rp[0..2n) = ap[0..n)^2
inline void mpn_sqr(limb_t* rp, const limb_t* ap, uint32_t n) {
    assert(n > 0);

    if (n < SQR_KARATSUBA_THRESHOLD) {
        mpn_sqr_basecase(rp, ap, n);
    } else if (n < SQR_NTT_THRESHOLD) {
        // Use Karatsuba for squaring (same algorithm, a == b)
        uint32_t scratch_n = 6 * n + 128;
        limb_t* scratch = mpn_alloc(scratch_n);
        mpn_mul_karatsuba_n(rp, ap, ap, n, scratch);
        mpn_free(scratch);
    } else {
        mpn_mul_ntt(rp, ap, n, ap, n);
    }
}

} // namespace bi
