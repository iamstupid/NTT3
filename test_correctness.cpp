#if defined(_MSC_VER)
#pragma warning(disable: 4324)  // structure was padded due to alignment specifier
#endif

#ifdef __GNUC__
#pragma GCC target("avx2")
#endif

#include "ntt/api.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <fstream>
#include <sstream>

using namespace ntt;

// ════════════════════════════════════════════════
// Test 1: Small cases (<1000 limbs) via Python ground truth
// ════════════════════════════════════════════════

// Naive schoolbook multiply for small validation
static std::vector<u32> schoolbook_multiply(const std::vector<u32>& a, const std::vector<u32>& b) {
    if (a.empty() || b.empty()) return {};
    idt n = a.size() + b.size();
    std::vector<u32> result(n, 0);
    for (idt i = 0; i < a.size(); ++i) {
        u64 carry = 0;
        for (idt j = 0; j < b.size(); ++j) {
            u64 prod = u64(a[i]) * b[j] + result[i + j] + carry;
            result[i + j] = (u32)prod;
            carry = prod >> 32;
        }
        idt k = i + b.size();
        while (carry && k < n) {
            u64 s = u64(result[k]) + carry;
            result[k] = (u32)s;
            carry = s >> 32;
            ++k;
        }
    }
    // Trim leading zeros
    while (result.size() > 1 && result.back() == 0) result.pop_back();
    return result;
}

static bool test_small_cases() {
    std::cout << "=== Test 1: Small cases (schoolbook reference) ===" << std::endl;
    std::mt19937 rng(42);
    bool all_pass = true;

    // Test various sizes
    int sizes[] = {1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 100, 128, 255, 256, 500, 512, 999};

    for (int na : sizes) {
        for (int trial = 0; trial < 3; ++trial) {
            int nb = na;
            if (trial == 1) nb = std::max(1, na / 2);
            if (trial == 2) nb = std::min(999, na + na / 3);

            std::vector<u32> a(na), b(nb);
            for (auto& x : a) x = rng();
            for (auto& x : b) x = rng();

            // Schoolbook reference
            auto expected = schoolbook_multiply(a, b);

            // NTT result
            idt out_len = na + nb;
            std::vector<u32> result(out_len, 0);
            big_multiply(result.data(), out_len, a.data(), na, b.data(), nb);

            // Trim
            while (result.size() > 1 && result.back() == 0) result.pop_back();

            if (result != expected) {
                std::cout << "  FAIL: na=" << na << " nb=" << nb << std::endl;
                // Print first diff
                idt diff_pos = 0;
                while (diff_pos < std::min(result.size(), expected.size()) &&
                       result[diff_pos] == expected[diff_pos]) ++diff_pos;
                std::cout << "    First diff at position " << diff_pos << std::endl;
                if (diff_pos < result.size())
                    std::cout << "    Got:      " << result[diff_pos] << std::endl;
                if (diff_pos < expected.size())
                    std::cout << "    Expected: " << expected[diff_pos] << std::endl;
                all_pass = false;
            } else {
                std::cout << "  PASS: na=" << na << " nb=" << nb << std::endl;
            }
        }
    }
    return all_pass;
}

// ════════════════════════════════════════════════
// Test 2: Large cases (>1000 limbs) via modular verification
// ════════════════════════════════════════════════

// Compute sum(a[i] * base^i) mod p for a small prime p
static u64 bignum_mod(const u32* a, idt len, u64 p) {
    u64 result = 0;
    u64 base_pow = 1;  // (2^32)^0 mod p
    u64 base = (u64(1) << 32) % p;
    for (idt i = 0; i < len; ++i) {
        result = (result + (u64)(a[i] % p) * base_pow) % p;
        base_pow = (base_pow * base) % p;
    }
    return result;
}

// Compute (a_mod * b_mod) mod p
static u64 mulmod(u64 a, u64 b, u64 p) {
    // For primes < 2^32, this won't overflow u64
    // Use __int128 for safety
#if defined(_MSC_VER)
    u64 hi;
    u64 lo = _umul128(a, b, &hi);
    u64 rem;
    _udiv128(hi, lo, p, &rem);
    return rem;
#else
    return (unsigned __int128)a * b % p;
#endif
}

static bool test_large_cases() {
    std::cout << "\n=== Test 2: Large cases (modular verification) ===" << std::endl;
    std::mt19937 rng(123);
    bool all_pass = true;

    // Small verification primes
    u64 test_primes[] = {
        1000000007ULL,
        998244353ULL,
        1000000009ULL,
        999999937ULL,
        104729ULL
    };

    int sizes[] = {1000, 2000, 5000, 10000, 50000, 100000};

    for (int na : sizes) {
        int nb = na;
        std::vector<u32> a(na), b(nb);
        for (auto& x : a) x = rng();
        for (auto& x : b) x = rng();

        idt out_len = na + nb;
        std::vector<u32> result(out_len, 0);
        big_multiply(result.data(), out_len, a.data(), na, b.data(), nb);

        // Verify against all test primes
        bool pass = true;
        for (u64 p : test_primes) {
            u64 a_mod = bignum_mod(a.data(), na, p);
            u64 b_mod = bignum_mod(b.data(), nb, p);
            u64 expected_mod = mulmod(a_mod, b_mod, p);

            // Trim trailing zeros for result
            idt rlen = out_len;
            while (rlen > 1 && result[rlen - 1] == 0) --rlen;
            u64 result_mod = bignum_mod(result.data(), rlen, p);

            if (result_mod != expected_mod) {
                std::cout << "  FAIL: na=" << na << " nb=" << nb
                          << " prime=" << p
                          << " got=" << result_mod
                          << " expected=" << expected_mod << std::endl;
                pass = false;
            }
        }

        if (pass) {
            std::cout << "  PASS: na=" << na << " nb=" << nb
                      << " (verified mod " << (sizeof(test_primes)/sizeof(test_primes[0]))
                      << " primes)" << std::endl;
        }
        all_pass = all_pass && pass;
    }

    return all_pass;
}

int main() {
    std::cout << "NTT Big Integer Multiplication - Correctness Tests" << std::endl;
    std::cout << "===================================================\n" << std::endl;

    bool pass1 = test_small_cases();
    bool pass2 = test_large_cases();

    std::cout << "\n===================================================" << std::endl;
    if (pass1 && pass2) {
        std::cout << "ALL TESTS PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED" << std::endl;
        return 1;
    }
}
