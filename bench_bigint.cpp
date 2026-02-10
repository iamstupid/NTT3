// bench_bigint.cpp - Performance benchmarks for bi::bigint
#include "bigint/bigint.hpp"
#include "bigint/mul.hpp"
#include "bigint/div.hpp"
#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <cmath>

using bi::bigint;
using bi::limb_t;

static double now_ns() {
    using namespace std::chrono;
    return (double)duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

// ============================================================
// mpn-level benchmarks
// ============================================================

static void bench_mpn_add_n() {
    printf("=== mpn_add_n throughput ===\n");
    printf("%8s %12s %12s\n", "limbs", "ns/call", "cycles/limb");

    for (uint32_t n : {4u, 16u, 64u, 256u, 1024u, 4096u, 16384u, 65536u}) {
        std::vector<limb_t> a(n), b(n), r(n);
        std::mt19937_64 rng(42);
        for (uint32_t i = 0; i < n; i++) { a[i] = rng(); b[i] = rng(); }

        int iters = (int)(2e8 / (double)n); // scale iterations
        if (iters < 10) iters = 10;

        double t0 = now_ns();
        limb_t dummy = 0;
        for (int it = 0; it < iters; it++) {
            dummy += bi::mpn_add_n(r.data(), a.data(), b.data(), n);
        }
        double t1 = now_ns();
        double ns_per_call = (t1 - t0) / iters;
        // Assuming ~4 GHz, 1 cycle ≈ 0.25 ns
        double cycles_per_limb = ns_per_call / n * 4.0; // rough estimate at 4 GHz

        printf("%8u %12.1f %12.2f\n", n, ns_per_call, cycles_per_limb);
        (void)dummy;
    }
}

static void bench_mpn_mul_1() {
    printf("\n=== mpn_mul_1 throughput ===\n");
    printf("%8s %12s %12s\n", "limbs", "ns/call", "cycles/limb");

    for (uint32_t n : {4u, 16u, 64u, 256u, 1024u, 4096u, 16384u}) {
        std::vector<limb_t> a(n), r(n);
        std::mt19937_64 rng(42);
        for (uint32_t i = 0; i < n; i++) a[i] = rng();
        limb_t multiplier = rng();

        int iters = (int)(1e8 / (double)n);
        if (iters < 10) iters = 10;

        double t0 = now_ns();
        limb_t dummy = 0;
        for (int it = 0; it < iters; it++) {
            dummy += bi::mpn_mul_1(r.data(), a.data(), n, multiplier);
        }
        double t1 = now_ns();
        double ns_per_call = (t1 - t0) / iters;
        double cycles_per_limb = ns_per_call / n * 4.0;

        printf("%8u %12.1f %12.2f\n", n, ns_per_call, cycles_per_limb);
        (void)dummy;
    }
}

static void bench_mpn_addmul_1() {
    printf("\n=== mpn_addmul_1 throughput ===\n");
    printf("%8s %12s %12s\n", "limbs", "ns/call", "cycles/limb");

    for (uint32_t n : {4u, 16u, 64u, 256u, 1024u, 4096u, 16384u}) {
        std::vector<limb_t> a(n), r(n);
        std::mt19937_64 rng(42);
        for (uint32_t i = 0; i < n; i++) { a[i] = rng(); r[i] = rng(); }
        limb_t multiplier = rng();

        int iters = (int)(1e8 / (double)n);
        if (iters < 10) iters = 10;

        double t0 = now_ns();
        limb_t dummy = 0;
        for (int it = 0; it < iters; it++) {
            dummy += bi::mpn_addmul_1(r.data(), a.data(), n, multiplier);
        }
        double t1 = now_ns();
        double ns_per_call = (t1 - t0) / iters;
        double cycles_per_limb = ns_per_call / n * 4.0;

        printf("%8u %12.1f %12.2f\n", n, ns_per_call, cycles_per_limb);
        (void)dummy;
    }
}

// ============================================================
// bigint-level benchmarks
// ============================================================

static bigint random_bigint_pos(std::mt19937_64& rng, int n_limbs) {
    bigint result(1LL);
    for (int i = 0; i < n_limbs; i++) {
        result <<= 64;
        // Add a random 64-bit value via mul_limb + add
        limb_t v = rng();
        bigint tmp;
        uint32_t hi = (uint32_t)(v >> 32);
        uint32_t lo = (uint32_t)v;
        tmp = bigint((long long)hi);
        tmp <<= 32;
        tmp += (long long)lo;
        result += tmp;
    }
    return result;
}

static void bench_bigint_add() {
    printf("\n=== bigint addition ===\n");
    printf("%8s %12s %12s\n", "limbs", "ns/call", "GB/s");

    std::mt19937_64 rng(42);

    for (int n : {1, 4, 16, 64, 256, 1024, 4096, 16384}) {
        bigint a = random_bigint_pos(rng, n);
        bigint b = random_bigint_pos(rng, n);

        int iters = (int)(5e7 / (double)n);
        if (iters < 10) iters = 10;

        // Warm up
        bigint r = a + b;
        (void)r;

        double t0 = now_ns();
        for (int it = 0; it < iters; it++) {
            r = a + b;
        }
        double t1 = now_ns();
        double ns_per_call = (t1 - t0) / iters;
        double bytes = (double)n * 8.0 * 2.0; // read 2 operands
        double gb_per_s = bytes / ns_per_call; // bytes/ns = GB/s

        printf("%8d %12.1f %12.2f\n", n, ns_per_call, gb_per_s);
    }
}

static void bench_bigint_shift() {
    printf("\n=== bigint left shift ===\n");
    printf("%8s %12s\n", "limbs", "ns/call");

    std::mt19937_64 rng(42);

    for (int n : {1, 4, 16, 64, 256, 1024, 4096}) {
        bigint a = random_bigint_pos(rng, n);

        int iters = (int)(1e7 / (double)n);
        if (iters < 10) iters = 10;

        double t0 = now_ns();
        for (int it = 0; it < iters; it++) {
            bigint r = a << 17; // shift by 17 bits
            (void)r;
        }
        double t1 = now_ns();
        double ns_per_call = (t1 - t0) / iters;

        printf("%8d %12.1f\n", n, ns_per_call);
    }
}

static void bench_to_string() {
    printf("\n=== bigint to_string (decimal, D&C) ===\n");
    printf("%8s %8s %12s %12s\n", "limbs", "digits", "ms/call", "us/digit");

    std::mt19937_64 rng(42);

    for (int n : {1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}) {
        bigint a = random_bigint_pos(rng, n);

        // Warm up + measure digit count
        std::string s = a.to_string();
        int digits = (int)s.size();

        // Scale iterations: D&C is ~O(M(n)*log(n))
        int iters;
        if (n <= 64) iters = 1000;
        else if (n <= 512) iters = 100;
        else if (n <= 4096) iters = 10;
        else iters = 3;

        double t0 = now_ns();
        for (int it = 0; it < iters; it++) {
            s = a.to_string();
        }
        double t1 = now_ns();
        double ms_per_call = (t1 - t0) / iters / 1e6;
        double us_per_digit = (t1 - t0) / iters / 1e3 / digits;

        printf("%8d %8d %12.3f %12.3f\n", n, digits, ms_per_call, us_per_digit);
    }
}

static void bench_from_string() {
    printf("\n=== bigint from_string (decimal, D&C) ===\n");
    printf("%8s %8s %12s %12s\n", "limbs", "digits", "ms/call", "us/digit");

    std::mt19937_64 rng(77);

    for (int n : {1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}) {
        bigint a = random_bigint_pos(rng, n);
        std::string s = a.to_string();
        int digits = (int)s.size();

        int iters;
        if (n <= 64) iters = 1000;
        else if (n <= 512) iters = 100;
        else if (n <= 4096) iters = 10;
        else iters = 3;

        // Warm up
        bigint b = bigint::from_string(s.c_str());
        (void)b;

        double t0 = now_ns();
        for (int it = 0; it < iters; it++) {
            b = bigint::from_string(s.c_str());
        }
        double t1 = now_ns();
        double ms_per_call = (t1 - t0) / iters / 1e6;
        double us_per_digit = (t1 - t0) / iters / 1e3 / digits;

        printf("%8d %8d %12.3f %12.3f\n", n, digits, ms_per_call, us_per_digit);
    }
}

// ============================================================
// Multiplication benchmarks
// ============================================================

static void bench_mpn_mul() {
    printf("\n=== mpn_mul (balanced n x n) ===\n");
    printf("%8s %12s %12s %12s\n", "limbs", "ns/call", "algo", "ns/limb^2");

    for (uint32_t n : {4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u, 2048u, 4096u, 8192u, 16384u}) {
        std::vector<limb_t> a(n), b(n), r(2 * n);
        std::mt19937_64 rng(42);
        for (uint32_t i = 0; i < n; i++) { a[i] = rng(); b[i] = rng(); }

        const char* algo;
        if (n < bi::KARATSUBA_THRESHOLD) algo = "basecase";
        else if (n < bi::NTT_THRESHOLD) algo = "karatsuba";
        else algo = "NTT";

        // Warm up
        bi::mpn_mul(r.data(), a.data(), n, b.data(), n);

        double total_ns = (n <= 256) ? 5e8 : 2e9;
        int iters = (int)(total_ns / ((double)n * n));
        if (n >= bi::NTT_THRESHOLD) iters = (int)(total_ns / ((double)n * 15));
        if (iters < 3) iters = 3;

        double t0 = now_ns();
        for (int it = 0; it < iters; it++) {
            bi::mpn_mul(r.data(), a.data(), n, b.data(), n);
        }
        double t1 = now_ns();
        double ns_per_call = (t1 - t0) / iters;
        double ns_per_limb2 = ns_per_call / ((double)n * n);

        printf("%8u %12.0f %12s %12.3f\n", n, ns_per_call, algo, ns_per_limb2);
    }
}

static void bench_bigint_mul() {
    printf("\n=== bigint multiply (balanced n x n) ===\n");
    printf("%8s %12s %12s\n", "limbs", "ns/call", "algo");

    std::mt19937_64 rng(99);

    for (int n : {4, 16, 64, 256, 1024, 4096, 16384, 65536}) {
        bigint a = random_bigint_pos(rng, n);
        bigint b = random_bigint_pos(rng, n);

        const char* algo;
        if ((uint32_t)n < bi::KARATSUBA_THRESHOLD) algo = "basecase";
        else if ((uint32_t)n < bi::NTT_THRESHOLD) algo = "karatsuba";
        else algo = "NTT";

        // Warm up
        bigint c = a * b;
        (void)c;

        double total_ns = ((uint32_t)n < bi::NTT_THRESHOLD) ? 5e8 : 2e9;
        int iters;
        if ((uint32_t)n < bi::KARATSUBA_THRESHOLD)
            iters = (int)(total_ns / ((double)n * n));
        else if ((uint32_t)n < bi::NTT_THRESHOLD)
            iters = (int)(total_ns / (double)(n * std::pow(n, 0.585)));
        else
            iters = (int)(total_ns / ((double)n * 20));
        if (iters < 3) iters = 3;

        double t0 = now_ns();
        for (int it = 0; it < iters; it++) {
            c = a * b;
        }
        double t1 = now_ns();
        double ns_per_call = (t1 - t0) / iters;

        printf("%8d %12.0f %12s\n", n, ns_per_call, algo);
    }
}

static void bench_mul_vs_div() {
    printf("\n=== mul(n,n) vs div(2n,n) ===\n");
    printf("%8s %12s %12s %8s\n", "n", "mul ns", "div ns", "ratio");

    for (uint32_t n : {64u, 128u, 256u, 512u, 1024u, 2048u, 4096u, 8192u, 16384u}) {
        std::vector<limb_t> a(n), b(n), r(2 * n);
        std::mt19937_64 rng(42);
        for (uint32_t i = 0; i < n; i++) { a[i] = rng(); b[i] = rng(); }
        b[n - 1] |= (limb_t)1 << 63; // ensure divisor is "large"

        // --- Benchmark mul(n,n) ---
        bi::mpn_mul(r.data(), a.data(), n, b.data(), n);  // warmup

        int mul_iters;
        if (n <= 256) mul_iters = (int)(5e8 / ((double)n * n));
        else if (n < 1024) mul_iters = (int)(5e8 / ((double)n * std::pow(n, 0.585)));
        else mul_iters = (int)(2e9 / ((double)n * 15));
        if (mul_iters < 3) mul_iters = 3;

        double t0 = now_ns();
        for (int it = 0; it < mul_iters; it++) {
            bi::mpn_mul(r.data(), a.data(), n, b.data(), n);
        }
        double t1 = now_ns();
        double mul_ns = (t1 - t0) / mul_iters;

        // --- Benchmark div(2n, n) ---
        // Set up a 2n-limb numerator and n-limb divisor
        std::vector<limb_t> num(2 * n + 1), d(n), q(n + 1);
        for (uint32_t i = 0; i < 2 * n; i++) num[i] = rng();
        for (uint32_t i = 0; i < n; i++) d[i] = rng();
        d[n - 1] |= (limb_t)1 << 63;
        // Ensure np[nn-1] < dp[dn-1] by clearing top bit of numerator
        num[2 * n - 1] &= ~((limb_t)1 << 63);
        num[2 * n] = 0;

        // warmup
        {
            std::vector<limb_t> tmp(num.begin(), num.end());
            bi::mpn_div_qr(q.data(), tmp.data(), 2 * n, d.data(), n);
        }

        int div_iters = mul_iters / 2;
        if (div_iters < 3) div_iters = 3;

        t0 = now_ns();
        for (int it = 0; it < div_iters; it++) {
            std::vector<limb_t> tmp(num.begin(), num.end());
            bi::mpn_div_qr(q.data(), tmp.data(), 2 * n, d.data(), n);
        }
        t1 = now_ns();
        double div_ns = (t1 - t0) / div_iters;

        printf("%8u %12.0f %12.0f %8.2f\n", n, mul_ns, div_ns, div_ns / mul_ns);
    }
}

int main() {
    printf("BigInt Performance Benchmarks\n");
    printf("=============================\n\n");

    bench_mul_vs_div();
    bench_to_string();

    printf("\nDone.\n");
    return 0;
}
