#pragma once
// div.hpp - Division algorithms for arbitrary-precision integers
//
// Dispatch: single-limb -> schoolbook (Knuth D) -> Newton reciprocal
// Single-limb division is in mpn.hpp (mpn_divrem_1).

#include "mpn.hpp"
#include "mul.hpp"
#include <cstring>
#include <algorithm>

namespace bi {

// ============================================================
// Tuning thresholds
// ============================================================

static constexpr uint32_t DIV_DC_THRESHOLD = 60;  // schoolbook -> D&C/Newton

// ============================================================
// Schoolbook division (Knuth Algorithm D)
// ============================================================

// Divide np[0..nn) by dp[0..dn), producing quotient qp[0..nn-dn+1) and
// remainder in np[0..dn). Quotient can be null (remainder-only).
// Preconditions: nn >= dn >= 2; dp[dn-1] != 0; np[nn-1] < dp[dn-1] (or nn > dn).
// np is modified in-place (overwritten with remainder in low dn limbs).
//
// Based on Knuth TAOCP Vol.2, Algorithm D.
static void mpn_div_qr_schoolbook(limb_t* qp, limb_t* np, uint32_t nn,
                                    const limb_t* dp, uint32_t dn)
{
    assert(nn >= dn && dn >= 2);
    assert(dp[dn - 1] != 0);

    // Step D1: Normalize — shift so that dp[dn-1] has its MSB set
    unsigned shift = clz64(dp[dn - 1]);
    limb_t* d_norm = mpn_alloc(dn);
    if (shift > 0) {
        mpn_lshift(d_norm, dp, dn, shift);
        // Also shift numerator (add one extra limb for overflow)
        np[nn] = mpn_lshift(np, np, nn, shift); // np now has nn+1 limbs effectively
        nn++;
    } else {
        std::memcpy(d_norm, dp, dn * sizeof(limb_t));
        np[nn] = 0;
        nn++;
    }

    limb_t d1 = d_norm[dn - 1]; // most significant divisor limb (MSB set)
    limb_t d0 = d_norm[dn - 2]; // second most significant

    uint32_t qn = nn - dn; // number of quotient limbs

    // Step D2-D7: Main loop (from most significant quotient limb down)
    for (uint32_t j = qn; j-- > 0; ) {
        // Step D3: Estimate quotient digit q_hat
        // Consider the 2-limb value (np[j+dn], np[j+dn-1]) / d1
        limb_t n2 = np[j + dn];     // high limb
        limb_t n1 = np[j + dn - 1]; // next limb
        limb_t n0 = np[j + dn - 2]; // limb below that

        limb_t q_hat, r_hat;
        if (n2 == d1) {
            q_hat = ~(limb_t)0; // UINT64_MAX
            r_hat = n1;
            // Check overflow: r_hat += d1; if it overflows, skip the adjustment
            unsigned char c = _addcarry_u64(0, r_hat, d1, (unsigned long long*)&r_hat);
            if (!c) {
                // Adjust: while q_hat * d0 > (r_hat, n0)
                limb_t ph;
                limb_t pl = umul_hilo(q_hat, d0, &ph);
                if (ph > r_hat || (ph == r_hat && pl > n0)) {
                    q_hat--;
                    c = _addcarry_u64(0, r_hat, d1, (unsigned long long*)&r_hat);
                    if (!c) {
                        pl = umul_hilo(q_hat, d0, &ph);
                        if (ph > r_hat || (ph == r_hat && pl > n0)) {
                            q_hat--;
                        }
                    }
                }
            }
        } else {
            q_hat = udiv_128(n2, n1, d1, &r_hat);

            // Adjust: while q_hat * d0 > (r_hat, n0)
            limb_t ph;
            limb_t pl = umul_hilo(q_hat, d0, &ph);
            if (ph > r_hat || (ph == r_hat && pl > n0)) {
                q_hat--;
                unsigned char c = _addcarry_u64(0, r_hat, d1, (unsigned long long*)&r_hat);
                if (!c) {
                    pl = umul_hilo(q_hat, d0, &ph);
                    if (ph > r_hat || (ph == r_hat && pl > n0)) {
                        q_hat--;
                    }
                }
            }
        }

        // Step D4: Multiply and subtract: np[j..j+dn] -= q_hat * d_norm[0..dn)
        limb_t borrow = mpn_submul_1(np + j, d_norm, dn, q_hat);
        limb_t old_np_top = np[j + dn];
        np[j + dn] -= borrow;

        // Step D5: Test remainder
        if (np[j + dn] > old_np_top) { // borrow from subtraction means q_hat was too large
            // Step D6: Add back
            q_hat--;
            limb_t carry = mpn_add_n(np + j, np + j, d_norm, dn);
            np[j + dn] += carry;
        }

        if (qp) qp[j] = q_hat;
    }

    // Un-normalize the remainder
    if (shift > 0) {
        mpn_rshift(np, np, dn, shift);
    }

    mpn_free(d_norm);
}

// ============================================================
// Newton reciprocal inversion
// ============================================================

static constexpr uint32_t INVERT_THRESHOLD = 32;

// Compute ip[0..dn) such that (B^dn + ip) ≈ floor(B^(2*dn) / D)
// D = dp[0..dn) must be normalized (dp[dn-1] has MSB set).
// Result may be off by ±2.
static void mpn_newton_invert(limb_t* ip, const limb_t* dp, uint32_t dn)
{
    assert(dn >= 1 && (dp[dn - 1] >> 63) != 0);

    if (dn == 1) {
        limb_t rem;
        ip[0] = udiv_128(~dp[0], ~(limb_t)0, dp[0], &rem);
        return;
    }

    if (dn <= INVERT_THRESHOLD) {
        // Base case: schoolbook division of (B^(2n) - 1) by D
        uint32_t nn = 2 * dn;
        limb_t* num = mpn_alloc(nn + 1);
        for (uint32_t i = 0; i < nn; i++) num[i] = ~(limb_t)0;
        num[nn] = 0;
        limb_t* qp = mpn_alloc(dn + 1);
        mpn_div_qr_schoolbook(qp, num, nn, dp, dn);
        mpn_copyi(ip, qp, dn);
        mpn_free(num);
        mpn_free(qp);
        return;
    }

    // Recursive Newton step: double precision from h to dn
    uint32_t h = (dn + 1) / 2;  // ceil(dn/2)
    uint32_t l = dn - h;        // floor(dn/2), l <= h

    // Step 1: Recursively invert top h limbs of D
    mpn_newton_invert(ip + l, dp + l, h);
    // V_h = B^h + ip[l..dn)

    // Step 2: T = D * V_h = D * I_h + D * B^h
    limb_t* tp = mpn_alloc(dn + h + 2);
    mpn_mul(tp, dp, dn, ip + l, h);   // tp[0..dn+h)
    tp[dn + h] = 0;
    limb_t carry = mpn_add_n(tp + h, tp + h, dp, dn);
    tp[dn + h] += carry;
    // T = tp[0..dn+h+1), T[dn+h] ∈ {0, 1}, T ≈ B^(dn+h)

    // Step 3: err = B^(dn+h) - T (signed)
    bool err_neg;
    limb_t* err = mpn_alloc(dn + h);
    if (tp[dn + h] == 0) {
        // T < B^(dn+h), err > 0: negate T (two's complement)
        err_neg = false;
        limb_t c = 1;
        for (uint32_t i = 0; i < dn + h; i++) {
            unsigned char cc = _addcarry_u64(0, ~tp[i], c, (unsigned long long*)&err[i]);
            c = cc;
        }
    } else {
        // T >= B^(dn+h), err < 0: |err| = T[0..dn+h)
        err_neg = true;
        mpn_copyi(err, tp, dn + h);
    }
    mpn_free(tp);

    uint32_t err_n = mpn_normalize(err, dn + h);
    if (err_n == 0) {
        mpn_zero(ip, l);
        mpn_free(err);
        return;
    }

    // Step 4: correction = V_h * |err| / B^(2h)
    limb_t* vh = mpn_alloc(h + 1);
    mpn_copyi(vh, ip + l, h);
    vh[h] = 1;

    uint32_t prod_n = (h + 1) + err_n;
    limb_t* prod = mpn_alloc(prod_n);
    if (h + 1 >= err_n)
        mpn_mul(prod, vh, h + 1, err, err_n);
    else
        mpn_mul(prod, err, err_n, vh, h + 1);
    mpn_free(vh);
    mpn_free(err);

    uint32_t corr_start = 2 * h;
    uint32_t corr_n = (prod_n > corr_start) ? prod_n - corr_start : 0;

    // Step 5: I_full = I_h * B^l ± correction
    if (err_neg) {
        // Subtract correction from I_h * B^l
        mpn_zero(ip, l);
        if (corr_n > 0) {
            uint32_t sn = (corr_n <= dn) ? corr_n : dn;
            mpn_sub(ip, ip, dn, prod + corr_start, sn);
        }
    } else {
        // Add correction to low l limbs, carry into I_h
        if (corr_n <= l) {
            if (corr_n > 0) mpn_copyi(ip, prod + corr_start, corr_n);
            if (corr_n < l) mpn_zero(ip + corr_n, l - corr_n);
        } else {
            mpn_copyi(ip, prod + corr_start, l);
            uint32_t overflow = corr_n - l;
            uint32_t an = (overflow <= h) ? overflow : h;
            mpn_add(ip + l, ip + l, h, prod + corr_start + l, an);
        }
    }
    mpn_free(prod);
}

// ============================================================
// Newton division: single block (wn ≤ 2*dn)
// ============================================================

// Divide np[0..wn) by d[0..dn) using precomputed reciprocal inv.
// wn > dn, wn ≤ 2*dn+1. D normalized. inv = Newton reciprocal of D.
// Quotient in qp[0..wn-dn) (can be null). Remainder in np[0..dn).
// np modified in-place.
static void newton_div_block(limb_t* qp, limb_t* np, uint32_t wn,
                              const limb_t* d, uint32_t dn,
                              const limb_t* inv)
{
    uint32_t this_qn = wn - dn;

    // Quotient estimate: Q ≈ (np_hi * inv) >> (dn * 64) + np_hi
    // where np_hi = np[dn..wn) = the top this_qn limbs.
    // We only need the high part of np_hi * inv.
    //
    // np_hi has this_qn limbs, inv has dn limbs.
    // Product has this_qn + dn limbs. We want limbs [dn..this_qn+dn).
    // That gives us this_qn limbs of quotient estimate.
    uint32_t hi_n = this_qn; // number of limbs in np_hi
    const limb_t* np_hi = np + dn;

    uint32_t pn = hi_n + dn;
    limb_t* pp = mpn_alloc(pn + 2);
    pp[pn] = pp[pn + 1] = 0;
    if (hi_n >= dn)
        mpn_mul(pp, np_hi, hi_n, inv, dn);
    else
        mpn_mul(pp, inv, dn, np_hi, hi_n);

    // Q_est = (pp >> (dn*64)) + np_hi = pp[dn..pn) + np_hi[0..hi_n)
    limb_t* q_est = mpn_alloc(this_qn + 2);
    mpn_zero(q_est, this_qn + 2);
    uint32_t avail = (pn > dn) ? pn - dn : 0;
    uint32_t copy_n = (avail > this_qn) ? this_qn : avail;
    if (copy_n > 0) mpn_copyi(q_est, pp + dn, copy_n);
    // Add np_hi (implicit B^dn factor in the reciprocal)
    if (hi_n > 0) {
        limb_t c = mpn_add(q_est, q_est, this_qn + 1, np_hi, hi_n);
        (void)c;
    }
    mpn_free(pp);

    // Q_est might be off by a few. Compute remainder = np - Q_est * D.
    uint32_t qn_norm = mpn_normalize(q_est, this_qn + 1);
    if (qn_norm > 0) {
        uint32_t prod_n = qn_norm + dn;
        limb_t* prod = mpn_alloc(prod_n + 1);
        prod[prod_n] = 0;
        if (qn_norm >= dn)
            mpn_mul(prod, q_est, qn_norm, d, dn);
        else
            mpn_mul(prod, d, dn, q_est, qn_norm);

        uint32_t sn = (prod_n <= wn) ? prod_n : wn;
        limb_t borrow = mpn_sub(np, np, wn, prod, sn);
        mpn_free(prod);

        // Adjust Q down while remainder negative
        for (int adj = 0; adj < 5 && borrow; adj++) {
            mpn_sub_1(q_est, q_est, this_qn + 1, 1);
            limb_t ac = mpn_add(np, np, wn, d, dn);
            if (ac) borrow = 0;
        }
    }

    // Adjust Q up while remainder >= D
    for (int adj = 0; adj < 5; adj++) {
        bool high_nz = false;
        for (uint32_t i = dn; i < wn; i++) {
            if (np[i] != 0) { high_nz = true; break; }
        }
        if (!high_nz && mpn_cmp(np, d, dn) < 0) break;
        mpn_add_1(q_est, q_est, this_qn + 1, 1);
        mpn_sub(np, np, wn, d, dn);
    }

    if (qp) mpn_copyi(qp, q_est, this_qn);
    mpn_free(q_est);
}

// ============================================================
// Newton-based division (top-level)
// ============================================================

// Divide np[0..nn) by dp[0..dn) using Newton reciprocal.
// qp[0..nn-dn+1) = quotient (can be null). Remainder in np[0..dn).
// np must have nn+1 limbs allocated.
static void mpn_div_qr_newton(limb_t* qp, limb_t* np, uint32_t nn,
                                const limb_t* dp, uint32_t dn)
{
    assert(nn >= dn && dn >= 2 && dp[dn - 1] != 0);

    // Normalize divisor
    unsigned shift = clz64(dp[dn - 1]);
    limb_t* d_norm = mpn_alloc(dn);
    if (shift > 0) {
        mpn_lshift(d_norm, dp, dn, shift);
        np[nn] = mpn_lshift(np, np, nn, shift);
    } else {
        std::memcpy(d_norm, dp, dn * sizeof(limb_t));
        np[nn] = 0;
    }
    uint32_t nn1 = nn + 1; // np now has nn+1 limbs
    uint32_t qn = nn - dn + 1;

    // Compute reciprocal once
    limb_t* inv = mpn_alloc(dn);
    mpn_newton_invert(inv, d_norm, dn);

    if (qp) mpn_zero(qp, qn);

    // Process from top to bottom, peeling off ≤dn quotient limbs per block
    uint32_t top = nn1;
    while (top > dn) {
        uint32_t wn = (top > 2 * dn) ? 2 * dn : top;
        uint32_t base = top - wn;

        newton_div_block(qp ? qp + base : nullptr,
                         np + base, wn, d_norm, dn, inv);
        // np[base..base+dn) = remainder, np[base+dn..top) = 0
        top = base + dn;
    }

    // Un-normalize remainder
    if (shift > 0) {
        mpn_rshift(np, np, dn, shift);
    }

    mpn_free(d_norm);
    mpn_free(inv);
}

// ============================================================
// Top-level division dispatch
// ============================================================

// Divide np[0..nn) by dp[0..dn).
// Quotient written to qp[0..nn-dn+1) (can be null for remainder-only).
// Remainder is left in np[0..dn).
// Preconditions: nn >= dn >= 1; dp[dn-1] != 0.
// np is modified in-place.
inline void mpn_div_qr(limb_t* qp, limb_t* np, uint32_t nn,
                         const limb_t* dp, uint32_t dn)
{
    assert(nn >= dn && dn >= 1 && dp[dn - 1] != 0);

    if (dn == 1) {
        limb_t rem = mpn_divrem_1(qp ? qp : np, np, nn, dp[0]);
        np[0] = rem;
        return;
    }

    if (dn < DIV_DC_THRESHOLD) {
        mpn_div_qr_schoolbook(qp, np, nn, dp, dn);
    } else {
        mpn_div_qr_newton(qp, np, nn, dp, dn);
    }
}

// ============================================================
// Convenience: quotient and remainder as separate arrays
// ============================================================

// qp[0..nn-dn+1) = np[0..nn) / dp[0..dn)
// rp[0..dn) = np[0..nn) % dp[0..dn)
// np is NOT modified (copied internally).
inline void mpn_tdiv_qr(limb_t* qp, limb_t* rp,
                          const limb_t* np, uint32_t nn,
                          const limb_t* dp, uint32_t dn)
{
    assert(nn >= dn && dn >= 1 && dp[dn - 1] != 0);

    // Copy np so we can modify it
    limb_t* tmp = mpn_alloc(nn + 1); // +1 for normalization
    std::memcpy(tmp, np, nn * sizeof(limb_t));
    tmp[nn] = 0;

    mpn_div_qr(qp, tmp, nn, dp, dn);

    // Copy remainder
    std::memcpy(rp, tmp, dn * sizeof(limb_t));
    mpn_free(tmp);
}

} // namespace bi
