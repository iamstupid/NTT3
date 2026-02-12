# zint Library Audit

Full algorithmic and optimization audit of the zint arbitrary-precision integer library.

---

## Architecture Overview

```
bigint.hpp    High-level bigint class, radix conversion, operator dispatch
  mul.hpp     Multiplication: basecase -> Karatsuba -> NTT
  div.hpp     Division: schoolbook -> Newton (with precomputed reciprocals)
  mpn.hpp     Low-level limb operations: add, sub, shift, mul_1, divrem_1
  scratch.hpp Thread-local bump allocator (mark/restore)
  tuning.hpp  Centralized crossover thresholds
  asm/        Hand-written x64 ASM kernels (ADX/BMI2, Zen 4 targeted)
  ntt/api.hpp NTT bridge: p30x3 (3 primes, u32) and p50x4 (4 primes, FP)
```

**Storage model:** Sign-magnitude, 64-bit limbs, little-endian. Size + sign packed
in a single u32 (bit 31 = sign, bits 0-30 = limb count). Max ~2^31 limbs.

---

## Algorithm Complexity Summary

| Operation | Algorithm | Complexity | Threshold |
|-----------|-----------|------------|-----------|
| Addition / Subtraction | Linear carry chain | O(n) | Always |
| Multiplication (small) | Fused ASM schoolbook | O(n^2) | bn < 32 |
| Multiplication (medium) | Karatsuba | O(n^1.585) | 32 <= bn < ~200 |
| Multiplication (large) | NTT (3-prime or 4-prime) | O(n log n) | bn >= 128 AND an*bn >= 40960 |
| Squaring (small) | Symmetry-exploiting basecase | O(n^2/2) | n < 176 |
| Squaring (medium) | Karatsuba (generic, a=b) | O(n^1.585) | 176 <= n < 224 |
| Squaring (large) | NTT | O(n log n) | n >= 224 |
| Division (single limb) | Barrett divrem_1 | O(n) | dn = 1 |
| Division (small) | Schoolbook (Knuth Algorithm D) | O(nn * dn) | 2 <= dn < 60 |
| Division (large) | Newton (precomputed reciprocal) | O(M(n)) | dn >= 60 |
| Newton reciprocal | Recursive doubling | O(M(n)) | Base case <= 32 |
| to_string (power-of-2 base) | Bit extraction | O(n) | base in {2,4,8,16,32,64} |
| to_string (general base) | D&C with cached powers | O(M(n) log n) | All other bases |
| from_string (power-of-2 base) | Bit packing | O(D) | base in {2,4,8,16,32,64} |
| from_string (general base) | D&C with cached powers | O(M(n) log n) | All other bases |
| Bitwise AND/OR/XOR/NOT | AVX2 vectorized | O(n) | Always |
| Left/Right shift | AVX2 vectorized | O(n) | Always |
| Comparison | AVX2 vectorized scan | O(n) | Always |

All algorithms use optimal asymptotic complexity for their category. No O(n^2)
operations masquerade as O(n), and no suboptimal algorithm classes are used where
better ones are available (with the exception of missing Toom-3, discussed below).

---

## Layer 1: ASM Kernels (`asm/`)

### addmul_1_adx.asm -- ~1.7 cycles/limb, 2.7x vs scalar

ADX dual-carry chain (ADCX on CF, ADOX on OF) with BMI2 MULX. 2-way unrolled
with `loop` instruction (1 uop on Zen 4). Even/odd dispatch via parity peel.

**Correctness:** Final carry drained via `mov ebx, 0` (not `xor` — preserves flags).
`jrcxz` short-jump within range. All callee-saved registers (rbx, rsi, rdi) saved/restored.

### submul_1_adx.asm -- ~2.2 cycles/limb, 1.8x vs scalar

BMI2 only (no ADX — SUB clobbers OF). Serial carry chain: MULX -> ADD -> ADC -> SUB -> ADC.
2-way unrolled with `dec/jnz`.

**Minor issues:** Unused `rbx` push/pop (2 wasted cycles). Filename says `_adx` but no ADX used.

### mul_basecase_adx.asm -- 2.3x vs scalar at 32x32

Fused schoolbook: `rep stosq` zeros rp, then ALL rows (including first) use the addmul_1
inner loop. Eliminates per-row call overhead (~16 cycles/row). 8 callee-saved pushes;
5th param at `[rsp+104]` — correct Windows x64 ABI.

---

## Layer 2: mpn.hpp — Low-Level Limb Operations

31 functions audited. All are O(n) or O(1). Key findings:

### AVX2 Usage

| Function | AVX2? | Technique |
|----------|-------|-----------|
| mpn_not/and/or/xor_n | Yes | 256-bit logical ops, 4 limbs/iter |
| mpn_lshift/rshift | Yes | Overlapping 256-bit loads + SLL/SRL/OR |
| mpn_cmp | Yes | 256-bit cmpeq + BSR lane finding |
| mpn_add_1/sub_1 | Yes | SIMD scan for carry/borrow propagation |
| mpn_add_n/sub_n | No | Serial carry chain (inherently sequential) |
| mpn_mul_1 | No | Serial carry chain |

### Issues Found

| Priority | Issue | Impact |
|----------|-------|--------|
| Medium | `mpn_mul_1` lacks ASM dispatch | ~1.5-2x slower than potential ASM. Mitigated by `mul_basecase_adx` fusing the first row. Matters for standalone scalar-multiply uses (e.g., radix conversion's `mul_limb`). |
| Medium | `mpn_add_1v` is exact duplicate of `mpn_add_1` | Dead code. Either differentiate or remove. |
| Low | Stale comments on addmul_1/submul_1 dispatch | Comments say "scalar for tiny n, ASM for n>=3" but dispatch is unconditional. |
| Low | `mpn_copyi`/`mpn_copyd` have identical bodies | Both call `memmove`. The two-function distinction provides no value. |

---

## Layer 3: mul.hpp — Multiplication

### Basecase (n < 32)

Delegates to `zint_mpn_mul_basecase_adx`. O(n^2). Optimal for small n.

### Squaring Basecase (n < 176)

Computes off-diagonal triangle via `mpn_mul_1` + `mpn_addmul_1`, then a fused
pass doubles the off-diagonals and adds the diagonal (a[i]^2). ~1.5x faster
than full multiply.

**Issue:** No fused ASM squaring kernel. For n up to 176, each row incurs
function-call overhead. A dedicated ASM kernel would save ~10-15%.

### Karatsuba (32 <= n < ~200)

Standard balanced Karatsuba with:
- Correct odd-n splitting (h = n - n/2)
- Sign-tracking for |a0-a1| * |b0-b1|
- Quick zero-check to skip multiply when difference is zero
- Scratch passed explicitly (no per-level ScratchScope overhead)

**Issues found:**

| Priority | Issue |
|----------|-------|
| Medium | No dedicated Karatsuba squaring — `mpn_sqr` calls `mpn_mul_karatsuba_n(rp, ap, ap, n, ...)` which computes the absolute difference twice and multiplies (instead of squaring) it. Saves ~33% per level. |
| Low | Line 184 memcpy in recombination — copies z0 to scratch. GMP avoids this with in-place accumulation. Saves one O(n) pass per level (~6-7% of Karatsuba time). |
| Info | No Toom-3 layer between Karatsuba and NTT. For n = 64-200, Toom-3 (O(n^1.465)) could give 10-20% over Karatsuba, but the implementation complexity is high and the NTT kicks in around n=200. |

### Unbalanced Karatsuba

Slices the larger operand into bn-sized chunks, multiplying each chunk by bp.
Remainder chunk handled separately. Simple and correct.

### NTT Bridge

`mpn_mul_ntt` -> `ntt::big_multiply_u64` reinterprets u64 limbs as u32 arrays
(little-endian). Two-tier dispatch:

| Engine | Primes | Max NTT size | Arithmetic |
|--------|--------|-------------|------------|
| p30x3 | 3 (~30-bit) | 12.6M u32 (~3.1M u64 limbs) | u32 Montgomery SIMD |
| p50x4 | 4 (~50-bit) | Unlimited | double FMA Barrett |

**Major issue — no NTT squaring optimization:** When `ap == bp`, the NTT path
still performs two independent forward transforms. A squaring-aware path would
do one forward NTT + pointwise squaring + one inverse NTT, saving ~25-30% for
large squarings. This cascades into Newton division (which squares the reciprocal
at each doubling step).

### Dispatch Thresholds (tuning.hpp)

| Threshold | Value | Notes |
|-----------|-------|-------|
| KARATSUBA_THRESHOLD | 32 | Balanced multiply |
| NTT_BN_MIN | 128 | Minimum short dimension for NTT |
| NTT_AREA | 40960 | an*bn hyperbola crossover |
| SQR_KARATSUBA_THRESHOLD | 176 | High due to fast ASM basecase squaring |
| SQR_NTT_THRESHOLD | 224 | Low — Karatsuba squaring isn't competitive |

---

## Layer 4: div.hpp — Division

### Schoolbook (dn < 60)

Knuth Algorithm D with normalization, 2-by-1 quotient estimation, add-back
correction. Uses `mpn_submul_1` ASM kernel for the inner loop.

**Issue:** Uses hardware `div` for quotient estimation (~30-40 cycles). GMP
precomputes `invert_limb(d1)` for Barrett-style estimation (~10 cycles). For
long divisions (nn >> dn), the savings would be ~20 cycles per quotient digit.

### Newton Division (dn >= 60)

Computes Newton reciprocal (O(M(n))), then processes numerator in blocks.
Each block: quotient estimate via `np_hi * inv`, residual via `q * d`,
up to 5 adjustments.

**Reciprocal computation:** Recursive doubling with base cases at dn=1 (hardware div)
and dn<=32 (schoolbook division of all-ones). Total cost O(M(n)).

**Precomputed reciprocal support:** `mpn_div_qr_newton_preinv` and `mpn_tdiv_qr_preinv`
accept pre-normalized divisor + reciprocal, used by the radix conversion cache.

**Issue:** Division dispatch only checks `dn`, not `nn`. For nn >> dn with dn in
30-60 range, Newton would beat schoolbook but isn't selected.

---

## Layer 5: bigint.hpp — High-Level Class

### Critical Issues

| # | Function | Issue | Impact |
|---|----------|-------|--------|
| 1 | `compare(long long)` | **Heap-allocates 32 bytes for every scalar comparison.** Every `if (x == 0)`, `while (x > 0)` allocates + frees. Should be pure register-based O(1). | High — hot loop performance |
| 2 | `operator/=(long long)`, `operator%=(long long)` | **Uses full general-purpose division for single-limb divisors.** Multiple heap allocations, schoolbook normalization, scratch. Should use `div_limb`/`mpn_divrem_1` directly. | High — 10-50x slower than necessary |

### Medium Issues

| # | Function | Issue |
|---|----------|-------|
| 3 | `operator+=`, `operator-=` | Always creates a temporary `bigint` instead of operating in-place. Every `+=` discards the existing buffer and allocates a new one, even when capacity is sufficient. |
| 4 | `operator+=(long long)` etc. | Creates a full `bigint` from `long long` then invokes `operator+=`. Triggers 2 allocations for a single-limb add. The class has `add_limb` which does zero-alloc in-place work. |
| 5 | `operator~()` | Three O(n) passes and two allocations for `~x = -x - 1`. Could be one pass, zero allocations. |
| 6 | `operator>>=` on negatives | Uses truncation semantics (not floor). `(-7) >> 1 = -3`, not `-4`. Undocumented. |

### Low Issues

| # | Issue |
|---|-------|
| 7 | `ensure_capacity`: infinite loop if n > 2^31 (u32 doubling wraps to 0) |
| 8 | No rvalue-qualified `operator-()` or `abs()` — `-std::move(x)` copies |
| 9 | `from_limbs_unsigned` doesn't normalize — fragile if input has leading zeros |
| 10 | `operator*(const bigint&)` copies operand then multiplies, wasting one alloc |
| 11 | Binary `operator/` copies then divides, could call `divmod` directly |
| 12 | No ADL-visible free `swap` function |
| 13 | `operator<<(ostream&)` ignores stream format flags (hex, oct, etc.) |
| 14 | `bitwise_assign` and `bitwise_binary_op` have ~130 lines of duplicated logic |

---

## Layer 6: Radix Conversion (bigint.hpp)

Fully audited in [radix_conversion_audit.md](radix_conversion_audit.md). Summary:

- **Power-of-2 bases:** O(n) bit extraction/packing. Optimal.
- **General bases:** O(M(n) log n) D&C with chunk-aligned splitting. GMP-style
  cache stores `chunk_pow^(2^k)` instead of `base^(2^k)`, making the tree shallower.
- **LUT basecase:** Extracts `lut_t` digits at a time via `val % lut_pow` + table lookup.
  2.7x fewer divisions than scalar for base 10.
- **Newton reciprocal cache:** Memoized per-level, reused across calls.

**Main issue:** `dc_from_radix` creates a throwaway `bigint` copy of the cached power
via `from_limbs_unsigned` at every D&C level. Direct `mpn_mul` on limb arrays would
save one O(n) copy + allocation per level (5-15% improvement).

---

## Layer 7: Infrastructure

### scratch.hpp — Thread-Local Bump Allocator

Block-based bump allocator with mark/restore. Default 1 MB blocks, 1.5x geometric
growth. `ScratchScope` RAII wrapper for automatic save/restore.

- `static_assert(trivially_destructible)` prevents leaking non-trivial types
- 4096-byte block alignment satisfies all AVX2 requirements
- No shrink-to-fit (memory retained until thread exit) — acceptable for bigint workloads

### tuning.hpp — Centralized Thresholds

All crossover points in one file. Tuned for Zen 4 via benchmarking.

### NTT Memory Management

| Allocator | Engine | Strategy |
|-----------|--------|----------|
| NTTArena | p30x3 | Binned pool, tagged pointers. O(1) alloc/dealloc. |
| Raw aligned alloc | p50x4 | Fresh allocation per call. No pooling. |
| Scratch | zint | Bump + mark/restore. No per-alloc malloc. |

**Issue:** p50x4 uses `std::vector<u64>` heap allocation in the CRT hot path
instead of the Scratch allocator.

---

## Improvement Opportunities (Ranked by Impact)

### Tier 1 — High Impact

| # | Improvement | Location | Est. Gain |
|---|------------|----------|-----------|
| 1 | **NTT squaring** — single forward transform | api.hpp | 25-30% for large squarings; cascades into Newton division |
| 2 | **Scalar comparison without allocation** | bigint.hpp `compare(long long)` | Eliminates malloc/free per `if (x == 0)` |
| 3 | **Single-limb division fast path** | bigint.hpp `operator/=(long long)` | 10-50x for `x /= small_constant` |

### Tier 2 — Medium Impact

| # | Improvement | Location | Est. Gain |
|---|------------|----------|-----------|
| 4 | Dedicated Karatsuba squaring | mul.hpp | ~33% per Karatsuba recursion level for squaring |
| 5 | In-place `operator+=`/`-=` | bigint.hpp | Eliminates alloc+free per addition when capacity suffices |
| 6 | Scalar `operator+=(long long)` via `add_limb`/`sub_1` | bigint.hpp | Eliminates 2 allocs per scalar add |
| 7 | Fused ASM squaring kernel | asm/ | 10-15% for basecase squaring up to n=176 |
| 8 | ASM `mpn_mul_1` | asm/ | ~1.5-2x for standalone scalar multiply |

### Tier 3 — Low Impact

| # | Improvement | Location | Est. Gain |
|---|------------|----------|-----------|
| 9 | Copy-free Karatsuba recombination | mul.hpp line 184 | ~6-7% of Karatsuba time |
| 10 | `dc_from_radix` direct mpn_mul | bigint.hpp | 5-15% for large from_string |
| 11 | Barrett reduction in `limb_to_radix_string` | bigint.hpp | 3-5x in per-chunk digit extraction |
| 12 | `digit_value` LUT (256-byte table) | bigint.hpp | ~2 cycles/char in pow2_from_string |
| 13 | Schoolbook precomputed 2-by-1 reciprocal | div.hpp | ~20 cycles/quotient digit |
| 14 | nn-aware division dispatch | div.hpp | Wins when nn >> dn, dn in 30-60 |
| 15 | p50x4 arena pooling | api.hpp | Eliminates per-call malloc in large NTT |

---

## Correctness Assessment

**No bugs found.** All arithmetic, sign handling, carry propagation, memory
management, aliasing behavior, and edge cases (zero, INT64_MIN, self-operations)
are correct across all 6 layers. The ADX carry chains are properly maintained,
CRT reconstruction is sound, and Newton division converges correctly.

The only semantic concern is `operator>>=` on negative numbers using truncation
(not floor) semantics, which differs from GMP/Python but is consistent with C++.
