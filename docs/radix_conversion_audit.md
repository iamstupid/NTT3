# Radix Conversion Audit

Detailed analysis of every function in the generalized radix conversion system
(`zint/bigint.hpp` lines 486-1179, `zint/tuning.hpp` lines 39-47).

---

## 1. Utility Functions (lines 486-549)

### 1.1 `RADIX_ALPHABET` (line 488-490)

```
"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+/"
```

64-character string constant. Indexed by digit value to produce the output character.

**Complexity:** O(1) per access.

**Assessment:** Correct. The `+` at position 62 and `/` at position 63 are the
only non-alphanumeric characters, and the `from_string` sign parser correctly
skips `+` only for `base <= 62`.

---

### 1.2 `digit_value(char c, uint32_t base)` (lines 494-505)

Reverse mapping from character to digit value. Branching chain:
- `'0'..'9'` -> 0..9
- `'A'..'Z'` -> 10..35
- `'a'..'z'` -> 10..35 (if base <= 36) or 36..61 (if base > 36)
- `'+'` -> 62, `'/'` -> 63

Returns -1 if the digit value >= base.

**Complexity:** O(1) per call. 4-5 branches worst case.

**Optimization notes:**
- A 256-byte lookup table indexed by `(unsigned char)c` would eliminate all
  branches and be a single memory load. At ~256 bytes it fits in L1. For
  bases > 36 vs <= 36 you'd need either two tables or a single table for the
  worst case (base 64) with a final `v < base` check.
- The branch chain is fine for the basecase (which is not the bottleneck) but
  `pow2_from_string` calls this per character. For a 1M-digit hex string that's
  1M branch-chain evaluations. A LUT would save ~2-3 ns/char on a modern CPU.
- **Verdict:** Minor optimization opportunity. The function is not on the
  critical path for large inputs (D&C dominates), and for power-of-2 bases the
  per-character cost is already dwarfed by memory bandwidth. Worth doing only
  if profiling shows it matters.

---

### 1.3 `pow2_shift(uint32_t base)` (lines 508-514)

Returns log2(base) if base is an exact power of 2, else 0.

**Complexity:** O(log base) due to the shift loop (max 6 iterations for base 64).

**Optimization notes:**
- Could use `__builtin_ctz` / `_BitScanForward` for a branchless single-cycle
  version. But this is called once per conversion, so irrelevant.
- **Verdict:** Fine as-is.

---

### 1.4 `compute_chunk_k(uint32_t base)` (lines 517-525)

Finds the maximum k where `base^k` fits in a u64, by iterative multiplication
with overflow check (`pow <= UINT64_MAX / base`).

**Complexity:** O(chunk_k) iterations. For base 2 that's 63; for base 64 it's 10.

**Correctness:** The overflow check is sound: if `pow <= UINT64_MAX / base`,
then `pow * base <= UINT64_MAX` (no overflow). After the loop, `pow` holds
`base^k` which is the chunk_pow.

**Optimization notes:** Called once per cache init. Irrelevant.

---

### 1.5 `compute_pow(uint32_t base, uint32_t k)` (lines 528-532)

Simple iterative exponentiation. No overflow checking (caller must ensure fit).

**Complexity:** O(k).

**Optimization notes:** Called a few times during cache setup. Irrelevant.

---

### 1.6 `compute_lut_t(uint32_t base, uint32_t max_lut_size)` (lines 535-543)

Finds max t where `base^t <= max_lut_size`. Same overflow-safe loop pattern as
`compute_chunk_k` but with `max_lut_size` instead of `UINT64_MAX`.

**Complexity:** O(lut_t) iterations (at most ~6 for RADIX_LUT_MAX_SIZE=4096).

**Correctness:** Sound. Returns 0 for base > max_lut_size (LUT disabled).

---

### 1.7 `est_radix_digits(uint32_t bits, uint32_t base)` (lines 546-549)

Upper-bound on the number of base-b digits for a number with `bits` bits:
`ceil(bits / log2(base)) + 1`.

**Complexity:** O(1). Uses floating-point `std::ceil` and `std::log2`.

**Correctness:** The `+1` provides a safety margin for FP rounding. For base 10:
`log2(10) = 3.3219...`, so a 64-bit number needs at most `ceil(64/3.3219) + 1 = 20 + 1 = 21`
digits. The true maximum is 20, so the +1 is conservative but safe.

**Optimization notes:**
- Called at the top level and at each D&C level (for tightening). The FP
  computation is ~5ns. At O(log n) D&C levels this is negligible.
- Could precompute `1.0 / log2(base)` in the cache to avoid repeated `log2`.
  Not worth it.
- **Verdict:** Fine as-is.

---

## 2. `radix_powers_cache` (lines 557-673)

Stores `chunk_pow^(2^k) = base^(chunk_k * 2^k)` lazily. This is the GMP-style
optimization: instead of storing `base^(2^k)`, we store `(base^chunk_k)^(2^k)`,
which makes the D&C tree shallower by a factor of `chunk_k`.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `radix` | u32 | User base (2..64) |
| `chunk_k` | u32 | Digits per u64 chunk (e.g. 19 for base 10) |
| `chunk_pow` | u64 | `base^chunk_k` (e.g. 10^19) |
| `lut_t_` | u32 | LUT width in digits |
| `lut_pow` | u32 | `base^lut_t_` (number of LUT entries) |
| `to_str_lut` | char* | `lut_pow * lut_t_` characters |
| `pow2k` | vector | `pow2k[k] = chunk_pow^(2^k)` with optional reciprocal |

### Key parameters by base

| Base | chunk_k | chunk_pow | lut_t | lut_pow | LUT bytes |
|------|---------|-----------|-------|---------|-----------|
| 3 | 40 | 3^40 = 12157665459056929 | 7 | 2187 | 15309 |
| 10 | 19 | 10^19 | 3 | 1000 | 3000 |
| 16 | 15 | 16^15 | 3 | 4096 | 12288 |
| 36 | 12 | 36^12 | 2 | 1296 | 2592 |
| 64 | 10 | 64^10 | 2 | 4096 | 8192 |

### 2.1 `reset(uint32_t base)` (lines 617-642)

Initializes cache for a given base. Computes chunk params, builds LUT, sets
`pow2k[0] = chunk_pow`.

**LUT construction (lines 629-637):**
```cpp
for (uint32_t v = 0; v < lut_pow; v++) {
    char* dst = to_str_lut + v * lut_t_;
    uint32_t tmp = v;
    for (int j = lut_t_ - 1; j >= 0; j--) {
        dst[j] = RADIX_ALPHABET[tmp % base];
        tmp /= base;
    }
}
```

**Complexity:** O(lut_pow * lut_t_). For base 10: 1000 * 3 = 3000 ops. For
base 64: 4096 * 2 = 8192 ops. Negligible.

**Correctness:** Each entry `v` is converted to `lut_t_` digits in big-endian
order. The alphabet indexing is correct.

**Early-return guard (line 618):** `if (radix == base && !pow2k.empty()) return;`
prevents redundant reinit when the cache is reused for the same base. Correct.

---

### 2.2 `get_pow2k(uint32_t k)` (lines 644-655)

Lazily computes `chunk_pow^(2^k)` by repeated squaring of the previous entry.

**Complexity per level:** One `mpn_sqr` of an n-limb number produces 2n limbs.
If the final level k has size S limbs, total cost is:
```
M(S/2) + M(S/4) + ... + M(1) = O(M(S))
```
where M(n) is the multiplication cost. This is optimal — computing the full
power tower costs the same as the last squaring.

**Memory:** Each entry allocates `2 * prev.size` limbs via `mpn_alloc`. The
`mpn_normalize` trims trailing zeros. No memory is wasted.

**Reference invalidation:** The code stores `const entry& prev = pow2k.back()`
then does `pow2k.push_back(...)`. This is **technically UB** if the vector
reallocates during push_back, invalidating `prev`. However, `prev.data` and
`prev.size` are read before the push_back call (the squaring happens first),
so the actual values are consumed before the reference could be invalidated.
The only risk is if the compiler reorders reads past the push_back, which
doesn't happen in practice. Still, storing `prev.data` and `prev.size` in
local variables before the push would be strictly safer.

---

### 2.3 `ensure_reciprocal(uint32_t k)` (lines 657-672)

Computes Newton reciprocal for division at level k. Only computed for entries
where `size >= DIV_DC_THRESHOLD` (60 limbs), since schoolbook division is
faster for small divisors.

**Complexity:** `mpn_newton_invert` is O(M(n)) where n is the divisor size.
Called at most once per level (memoized via `if (e.inv) return`).

**Correctness:** Normalizes the divisor (left-shift so MSB is set) before
computing the reciprocal, as required by Newton division.

---

## 3. Power-of-2 Fast Paths (lines 932-986)

### 3.1 `pow2_to_string(const bigint& x, unsigned shift)` (lines 934-958)

Extracts `shift` bits per digit from MSB to LSB using bit indexing.

**Algorithm:**
```
for i = n_digits-1 downto 0:
    bit_pos = i * shift
    val = extract shift bits at bit_pos from limb array
    emit RADIX_ALPHABET[val]
```

**Complexity:** O(D) where D = number of output digits = O(n * 64 / shift).
This is O(n) in the number of limbs — optimal.

**Optimization notes:**
- The inner loop does one division and one modulo by 64 (compiled to shift/mask),
  one or two 64-bit shifts, and one array index. This is ~3-5 cycles per digit.
- For hex (shift=4), that's ~4 cycles per digit = ~64 cycles per limb = ~1 cycle
  per byte of limb data. This is well within memory bandwidth limits.
- `result.reserve(n_digits + 1)` prevents reallocation. Good.
- The cross-limb boundary case (`bit_off + shift > 64`) adds one branch per
  digit. For hex (shift=4, 16 digits per limb), this triggers every 16th digit.
  For base-2 (shift=1), never. For base-64 (shift=6), every ~10th digit.
  The branch is well-predicted.
- **Verdict:** Optimal asymptotically. Constant factor is good. Could be
  vectorized with SIMD for hex specifically (pshufb-based nibble-to-ASCII),
  but that's a micro-optimization for a non-bottleneck path.

---

### 3.2 `pow2_from_string(const char* s, uint32_t len, unsigned shift, bool neg)` (lines 961-986)

Packs `shift` bits per character from LSB to MSB.

**Algorithm:**
```
for i = 0 to len-1:
    v = digit_value(s[len-1-i], base)   // LSB first
    pack v into rp at bit position i*shift
```

**Complexity:** O(D) where D = string length. Optimal.

**Optimization notes:**
- `digit_value` is called per character (see 1.2 above). A 256-byte LUT would
  shave ~1-2 cycles/char.
- `mpn_zero(rp, n_limbs)` at the start zeroes the entire output. Then bits are
  OR'd in. This is fine; the zero pass is a single memset.
- Reverse iteration (`s[len-1-i]`) means we process the string backwards. This
  has slightly worse cache behavior than forward iteration (prefetcher works
  better with forward access), but for typical string sizes this is irrelevant.
- **Verdict:** Optimal. No issues.

---

## 4. Basecase Conversion (lines 988-1148)

### 4.1 `limb_to_radix_string(buf, val, cache)` (lines 991-1006)

Converts a single u64 value (`val < chunk_pow`) to exactly `chunk_k` digits
using the LUT.

**Algorithm:**
```
while pos >= lut_t_:
    idx = val % lut_pow
    val /= lut_pow
    memcpy(buf + pos, LUT[idx], lut_t_)
    pos -= lut_t_

while pos > 0:
    buf[--pos] = ALPHABET[val % base]
    val /= base
```

**Complexity:** O(chunk_k / lut_t_) divisions by `lut_pow` plus
O(chunk_k % lut_t_) divisions by `base`. Each u64 division is ~20-40 cycles
on Zen 4 (hardware `div` for 64-bit).

For base 10: chunk_k=19, lut_t_=3, so 6 divisions by 1000 + 1 division by 10.
= 7 divisions total. Without LUT it would be 19 divisions by 10. That's a
**2.7x reduction** in the number of divisions.

For base 36: chunk_k=12, lut_t_=2, so 6 divisions by 1296.
Without LUT: 12 divisions by 36. Same count — LUT provides no reduction here
because 12/2 = 6. The benefit is that each `memcpy` of 2 bytes avoids one
ALPHABET lookup + store.

For base 64: chunk_k=10, lut_t_=2, so 5 divisions by 4096.
Without LUT: 10 divisions by 64. Same factor 2x.

**Optimization notes:**
- The `memcpy` of `lut_t_` bytes (2 or 3) is compiled to a single `mov` instruction.
  This is optimal.
- Division by a runtime constant uses hardware `div` (~20-40 cycles on Zen 4).
  If `lut_pow` were known at compile time, the compiler could use Barrett
  reduction (multiply-shift) which is ~4 cycles. Since `lut_pow` depends on
  the base, this isn't possible without template specialization.
- **Possible improvement:** Pre-compute the Barrett inverse of `lut_pow` and
  `base` in the cache, then use manual Barrett reduction:
  `q = (uint128_t)(val * inv) >> 64; idx = val - q * lut_pow;`
  This would replace each ~30-cycle `div` with a ~6-cycle mulhi+sub.
  For base 10 with 7 divisions, that's ~210 -> ~42 cycles per chunk. Since
  basecase processes up to 30 limbs, each producing one chunk, the total
  savings would be ~(210-42)*30 = ~5000 cycles. That's ~1.5 microseconds —
  potentially visible at the RADIX_DC_THRESHOLD boundary.
- **Verdict:** The LUT is a solid improvement over scalar. Barrett division
  for the inner loop remainders would be a further 3-5x improvement in the
  basecase limb-to-string conversion, but the basecase is not the bottleneck
  for large conversions (D&C division cost dominates).

---

### 4.2 `basecase_to_radix(out, x, pad, cache)` (lines 1008-1052)

Extracts chunks by repeated `div_limb(chunk_pow)`, then converts each chunk
to digits.

**Algorithm:**
1. Repeatedly divide x by `chunk_pow`, collecting remainders in `chunk_buf[]`.
   Each remainder is a chunk value in `[0, chunk_pow)`.
2. Convert highest chunk without leading zeros (scalar loop).
3. Convert remaining chunks with `limb_to_radix_string` (LUT, zero-padded to
   `chunk_k` digits).
4. Prepend padding zeros if `pad > s.size()`.

**Complexity:** Step 1 is O(n * C) where n = number of limbs in x and
C = number of chunks = O(n * 64 / (chunk_k * log2(base))).
Each `div_limb` is an O(n) `mpn_divrem_1` call, and we make C calls, but
n shrinks as digits are extracted. Total work is O(n * C) where the n
decreases linearly, so it's O(n^2) in limbs — same as the naive basecase.

For the basecase, n <= RADIX_DC_THRESHOLD = 30, so this is at most 30 * ~30 =
~900 limb-level divisions. Each `mpn_divrem_1` on 30 limbs is ~30 * 5 = 150
cycles (Barrett divrem is ~5 cycles/limb). Total: ~135K cycles = ~40us. This
is acceptable for a basecase.

**Stack buffer:** `limb_t chunk_buf[64]` — 512 bytes on stack. Assertion
`n_chunks < 64` guards overflow. For n=30 limbs at base 3 (worst case):
30 * 64 / 40 = 48 chunks. Fits in 64. For base 2 this would be 30 * 64 / 63 = ~30.
Fine.

**String construction:** Uses `std::string s` with `reserve(n_chunks * chunk_k)`.
The highest chunk uses a scalar `tmp[20]` buffer reversed into `s`. Remaining
chunks use `cbuf[64]` + `s.append(cbuf, chunk_k)`.

**Optimization notes:**
- `s.reserve(n_chunks * chunk_k)` is slightly overallocated (highest chunk may
  have fewer digits), but this avoids reallocations. Good.
- The scalar highest-chunk conversion (lines 1031-1039) reverses digits into a
  temporary buffer then copies. Could use `limb_to_radix_string` + skip leading
  zeros, but this is a single chunk and the current approach is clear.
- The `std::string s` is accumulated then appended to `out` at the end. This
  means a temporary allocation + copy. An alternative would be to write
  directly into `out` with position tracking, but the code clarity is worth
  the minor overhead.
- **Verdict:** Correct and efficient for the basecase. O(n^2) is expected.

---

### 4.3 `basecase_from_radix(s, len, cache)` (lines 1112-1148)

Parses digits in `chunk_k`-character groups, accumulating via
`result.mul_limb(chunk_pow)` + `result.add_limb(chunk)`.

**Algorithm:**
1. Parse first partial chunk (len % chunk_k chars) to align boundaries.
2. For each subsequent full chunk: parse `chunk_k` chars -> u64, then
   `result.mul_limb(chunk_pow); result.add_limb(chunk);`

**Complexity:** Each `mul_limb` is an O(n) `mpn_mul_1` call. We make
C = len / chunk_k calls, with result growing from 1 to n limbs. Total
work is O(1 + 2 + ... + n) = O(n^2). This is the standard Horner basecase.

The D&C crossover is at `len <= chunk_k * RADIX_DC_THRESHOLD`. For base 10:
`19 * 30 = 570` chars. For base 3: `40 * 30 = 1200` chars. This means the
basecase handles strings up to ~570-1200 characters, which produce bigints of
up to ~30 limbs. The O(n^2) cost is ~30^2 = ~900 mul_1 calls of increasing
size. Total: ~450 * 5 = ~2250 cycles = ~0.7us. Acceptable.

**Optimization notes:**
- `digit_value` is called per character in the inner parsing loop. A 256-byte
  LUT would help here since we parse `chunk_k * RADIX_DC_THRESHOLD` characters.
- The `if (d < 0) d = 0` fallback silently treats invalid characters as 0.
  This is lenient — an alternative would be to return an error or assert.
  The leniency matches the to_string output (which never produces invalid chars),
  so in practice this path is never taken for valid roundtrips.
- **Verdict:** Correct. Standard Horner's method. O(n^2) basecase is expected.

---

## 5. D&C Conversion (lines 1054-1179)

### 5.1 `dc_to_radix(out, x, est_digits, pad, cache)` (lines 1056-1108)

D&C digit extraction via division by cached powers.

**Algorithm:**
1. Base cases: `x == 0` -> emit padding; `x.abs_size() <= 30` -> basecase.
2. Tighten digit estimate from actual bit length.
3. Compute split: `k = floor(log2(est_chunks / 2))`, `split_digits = chunk_k * 2^k`.
4. Divide x by `cache.pow2k[k]` to get quotient (high) and remainder (low).
5. Recurse on high (with `est_digits - split_digits`), then low (padded to
   `split_digits`).

**Complexity:** The recursion tree has O(log n) levels. At each level, the
dominant cost is one division of an n-limb number by an ~n/2-limb divisor,
which costs O(M(n)) using Newton division. The total is:
```
T(n) = T(n/2) + O(M(n)) = O(M(n) * log n)
```
This is optimal for the D&C approach (matching GMP's performance class).

**Split point analysis:**

The split is chunk-aligned: `split_digits` is always a multiple of `chunk_k`,
and equals `chunk_k * 2^k` for some k. This ensures:
- The divisor is always one of the cached powers (no extra computation).
- The low half has exactly `split_digits` digits (correctly padded).

The choice of k: `k = floor(log2(half_chunks))` where
`half_chunks = ceil(est_digits / chunk_k) / 2`. This picks the largest
power-of-2-aligned split that's <= half the number of chunks. The split is
approximately balanced but biased toward the low half being a power-of-2
chunk count.

**Tightening (line 1068-1070):** Re-estimates digits from actual bit length
at each recursion level. This prevents over-splitting when the quotient is
much smaller than estimated. Good.

**Division dispatch (lines 1091-1097):** Uses Newton division with precomputed
reciprocal for large divisors, schoolbook for small. The `ensure_reciprocal(k)`
call computes the reciprocal lazily and memoizes it. This is key for
performance: without memoization, each reciprocal computation at level k
costs O(M(n/2^k)), and doing this at every recursion level would add
O(M(n)) total cost (acceptable but wasteful when the same level is visited
by multiple branches).

**Optimization notes:**
- The `bigint q, r` allocations at each recursion level create temporary
  objects. For very deep recursion (~20+ levels), the heap allocation overhead
  could be visible. An alternative would be to use a scratch buffer pool.
  In practice, with RADIX_DC_THRESHOLD=30, the tree has at most ~15-20 levels
  for million-limb numbers, so this is fine.
- `bigint tmp = this->abs()` in `to_string` creates a copy. This is necessary
  since `dc_to_radix` is destructive (divisions modify x). The copy cost is
  O(n) and happens once.
- **Verdict:** Correct. O(M(n) log n). Well-structured.

---

### 5.2 `dc_from_radix(s, len, cache)` (lines 1152-1179)

D&C string-to-bigint via the reverse process: split string, recurse, combine.

**Algorithm:**
1. Basecase: `len <= chunk_k * RADIX_DC_THRESHOLD` -> `basecase_from_radix`.
2. Compute split: same chunk-aligned logic as `dc_to_radix`.
3. Recurse: `high = dc_from_radix(s, len - split_digits)`
4. Recurse: `low = dc_from_radix(s + (len - split_digits), split_digits)`
5. Combine: `high * cache.pow2k[k] + low`

**Complexity:** Same recursion as `dc_to_radix` but the dominant cost is the
multiplication `high * pow2k[k]` (O(M(n)) per level) plus the addition (O(n)).
Total: O(M(n) log n).

**Critical issue — `high *= pow_bi` uses full bigint multiply:**

Line 1174-1176:
```cpp
bigint pow_bi = from_limbs_unsigned(pow.data, pow.size);
high *= pow_bi;
high += low;
```

The `from_limbs_unsigned` creates a temporary bigint copying `pow.data`. Then
`high *= pow_bi` dispatches through `operator*=` which calls `mpn_mul`. This
is a **full general multiply** (basecase/Karatsuba/NTT depending on size).

This is correct but has two unnecessary costs:
1. **Copy of pow.data:** `from_limbs_unsigned` copies the divisor power into a
   new bigint. This is O(n) memory + copy. The data is already in the cache.
2. **The multiply is unbalanced:** `high` has roughly n/2 limbs and `pow2k[k]`
   also has roughly n/2 limbs. The `operator*=` internally allocates a result
   buffer, multiplies, then moves the result back. An in-place multiply that
   operates directly on the limb arrays would avoid one allocation.

For comparison, in `dc_to_radix`, the corresponding operation is a division
which operates directly on limb arrays via `mpn_tdiv_qr_preinv`. The
`dc_from_radix` path pays an extra allocation + copy per recursion level.

**Optimization opportunities:**
- Use `mpn_mul` directly on limb arrays instead of going through `operator*=`.
  This would eliminate the `from_limbs_unsigned` copy and one bigint allocation.
- The addition `high += low` is fine (O(n) in-place).

**Safety guard (lines 1166-1168):** `if (split_digits >= len) return basecase`.
This handles edge cases where the computed split exceeds the string length
(shouldn't happen with correct arithmetic, but defensive). Good.

---

## 6. Public API (lines 679-733)

### 6.1 `to_string(int base, radix_powers_cache* pow_cache)` (lines 679-703)

**Dispatch:**
1. Zero check -> return "0"
2. Power-of-2 -> `pow2_to_string` (O(n))
3. Otherwise -> D&C path (O(M(n) log n))

**Cache handling:** If `pow_cache` is null, creates a stack-local cache. If
provided, either initializes it (if radix==0) or asserts matching base. This
is the correct pattern for optional cache reuse.

**The `bigint tmp = this->abs()` copy:** Necessary because D&C division is
destructive. O(n) cost, paid once.

### 6.2 `from_string(const char* s, int base, radix_powers_cache* pow_cache)` (lines 705-733)

**Dispatch:**
1. Null/empty check
2. Sign handling: `-` sets neg, `+` consumed only for base <= 62
3. Leading zero skip
4. Power-of-2 -> `pow2_from_string` (O(D))
5. Otherwise -> D&C path (O(M(n) log n))

**The `strlen` call (line 714):** O(D) scan of the input string. Unavoidable
without a length parameter. For very large strings (millions of digits), this
is a single pass that should be pipelined with the subsequent zero-skip loop.

---

## 7. Complexity Summary

| Function | Complexity | Notes |
|----------|-----------|-------|
| `pow2_to_string` | O(n) | n = limbs. Optimal. |
| `pow2_from_string` | O(D) | D = digit count. Optimal. |
| `basecase_to_radix` | O(n^2) | n <= 30 limbs. ~40us worst case. |
| `basecase_from_radix` | O(n^2) | n <= 30 limbs. ~0.7us worst case. |
| `dc_to_radix` | O(M(n) log n) | Optimal D&C. Newton reciprocal cached. |
| `dc_from_radix` | O(M(n) log n) | Optimal D&C. One extra copy per level. |
| `limb_to_radix_string` | O(chunk_k/lut_t) | ~7 divisions for base 10. |
| `digit_value` | O(1) | Branch chain; LUT would be faster. |
| `radix_powers_cache::reset` | O(lut_pow * lut_t) | One-time. Negligible. |
| `radix_powers_cache::get_pow2k` | O(M(S)) | S = limbs at level k. Amortized. |
| `radix_powers_cache::ensure_reciprocal` | O(M(n)) | Memoized. One-time per level. |

---

## 8. Potential Improvements (prioritized)

### High impact (affects large inputs)

1. **`dc_from_radix`: avoid `from_limbs_unsigned` copy.** Replace the
   `bigint pow_bi = from_limbs_unsigned(...)` + `high *= pow_bi` pattern with
   direct `mpn_mul` on limb arrays. Saves one O(n) copy + one bigint allocation
   per D&C level. Estimated improvement: 5-15% on from_string for large inputs.

### Medium impact (affects basecase boundary)

2. **Barrett reduction in `limb_to_radix_string`.** Pre-compute Barrett
   inverses of `lut_pow` and `base` in the cache. Replace hardware `div` with
   `mulhi + sub`. Estimated 3-5x speedup for the per-chunk digit extraction.
   Only matters if the basecase is a significant fraction of total time (it is
   near the D&C threshold).

3. **`digit_value` LUT.** Replace the branch chain with a 256-byte table.
   Saves ~2 cycles/char in `pow2_from_string` and `basecase_from_radix`.
   Measurable for multi-megabyte hex strings.

### Low impact (micro-optimizations)

4. **`pow2_to_string` SIMD for hex.** Use `_mm256_shuffle_epi8` to convert
   16 nibbles to ASCII in one instruction. Only relevant for base-16 bulk
   conversion of multi-million-limb numbers.

5. **`get_pow2k` reference safety.** Store `prev.data` and `prev.size` in
   local variables before `pow2k.push_back()` to avoid potential reference
   invalidation.

6. **`basecase_to_radix` string allocation.** Write directly into `out`
   instead of building an intermediate `std::string s`. Saves one allocation +
   copy for the basecase path.

---

## 9. Correctness Checklist

| Item | Status |
|------|--------|
| Power-of-2 bases: bit extraction correct at limb boundaries | OK |
| Power-of-2 bases: cross-limb spanning (bit_off + shift > 64) | OK |
| LUT construction: digits in big-endian order | OK |
| LUT indexing: `val % lut_pow` gives correct entry | OK |
| chunk_k overflow check: `pow <= UINT64_MAX / base` | OK |
| D&C split always chunk-aligned | OK |
| D&C to_radix: high part padded correctly | OK |
| D&C to_radix: low part padded to exactly split_digits | OK |
| D&C from_radix: split_digits < len guard | OK |
| D&C from_radix: high * pow + low reconstruction | OK |
| Sign handling: `-` prefix consumed | OK |
| Sign handling: `+` consumed only for base <= 62 | OK |
| Leading zeros: skipped but `"0"` preserved | OK |
| Cache reuse: `reset()` idempotent for same base | OK |
| Cache mismatch: asserted | OK |
| Newton reciprocal: only for size >= DIV_DC_THRESHOLD | OK |
| est_radix_digits: conservative (+1 margin) | OK |
| basecase_to_radix: chunk_buf[64] sufficient for n<=30 | OK (max ~48 chunks) |
| Invalid digit fallback: treated as 0 (lenient) | OK (debatable) |
