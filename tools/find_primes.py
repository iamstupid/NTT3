#!/usr/bin/env python3
"""
Find primes of the form  p = c * 2^n * k + 1  with configurable c, n.
For each prime found, outputs: p, factorization of p-1, and a primitive root.

Usage:
    python find_primes.py                        # defaults: c=1, n=50
    python find_primes.py --c=15 --n=46          # primes with p-1 divisible by 15*2^46
    python find_primes.py --c=3 --n=48 --count=8
    python find_primes.py --c=1 --n=50 --min-bits=49 --max-bits=51
"""

import argparse
import math
from sympy import isprime, factorint, primitive_root


def find_primes(c: int, n: int, count: int, min_bits: int, max_bits: int):
    base = c * (1 << n)
    # Determine k range from bit constraints
    # p = base * k + 1, p >= 2^(min_bits-1), p < 2^max_bits
    k_min = max(1, (((1 << (min_bits - 1)) - 1) // base) + 1)
    k_max = ((1 << max_bits) - 1) // base
    if k_max < k_min:
        print(f"No candidates: c={c}, n={n} gives base={base} "
              f"({base.bit_length()} bits), but range [{min_bits}, {max_bits}] "
              f"bits is empty (k_min={k_min}, k_max={k_max})")
        return

    print(f"# Searching primes of form {c} * 2^{n} * k + 1")
    print(f"# base = {c} * 2^{n} = {base}")
    print(f"# k range: [{k_min}, {k_max}]  ({min_bits}-{max_bits} bit primes)")
    print(f"# Requesting {count} primes")
    print()

    # Factor c once (p-1 = c * 2^n * k, so its factors include 2^n and factors of c and k)
    c_factors = factorint(c) if c > 1 else {}

    found = 0
    for k in range(k_min, k_max + 1):
        p = base * k + 1
        if not isprime(p):
            continue

        # Factor p-1 = c * 2^n * k
        # Combine: 2^n, factors of c, factors of k
        pm1_factors = {2: n}
        for f, e in c_factors.items():
            pm1_factors[f] = pm1_factors.get(f, 0) + e
        k_factors = factorint(k) if k > 1 else {}
        for f, e in k_factors.items():
            pm1_factors[f] = pm1_factors.get(f, 0) + e

        g = primitive_root(p)

        # Format factorization
        parts = []
        for f in sorted(pm1_factors.keys()):
            e = pm1_factors[f]
            parts.append(f"{f}^{e}" if e > 1 else str(f))
        factstr = " * ".join(parts)

        # Verify
        check = 1
        for f, e in pm1_factors.items():
            check *= f ** e
        assert check == p - 1, f"Factorization mismatch: {check} != {p-1}"

        print(f"p = {p}")
        print(f"  {p.bit_length()} bits, k = {k}")
        print(f"  p - 1 = {factstr}")
        print(f"  primitive root = {g}")

        # Show orders available for NTT
        v2 = pm1_factors.get(2, 0)
        v3 = pm1_factors.get(3, 0)
        v5 = pm1_factors.get(5, 0)
        sizes = []
        if v2 >= 8:
            sizes.append(f"2^{v2}")
        if v3 >= 1 and v2 >= 8:
            sizes.append(f"3*2^{v2}")
        if v5 >= 1 and v2 >= 8:
            sizes.append(f"5*2^{v2}")
        if sizes:
            print(f"  max NTT sizes: {', '.join(sizes)}")
        print()

        found += 1
        if found >= count:
            break

    print(f"# Found {found} primes")


def main():
    parser = argparse.ArgumentParser(
        description="Find primes of the form c * 2^n * k + 1")
    parser.add_argument("--c", type=int, default=1,
                        help="Cofactor c (default: 1)")
    parser.add_argument("--n", type=int, default=50,
                        help="Power of 2 (default: 50)")
    parser.add_argument("--count", type=int, default=8,
                        help="Number of primes to find (default: 8)")
    parser.add_argument("--min-bits", type=int, default=49,
                        help="Minimum bit-length of primes (default: 49)")
    parser.add_argument("--max-bits", type=int, default=51,
                        help="Maximum bit-length of primes (default: 51)")
    args = parser.parse_args()

    find_primes(args.c, args.n, args.count, args.min_bits, args.max_bits)


if __name__ == "__main__":
    main()
