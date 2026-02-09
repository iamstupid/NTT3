// test_sd_ntt_integrated.cpp - Integration tests for ntt::big_multiply_u64()
//
// Tests the u64 4-prime sd_ntt path via the public API.

#include "ntt/api.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

using u64 = std::uint64_t;

// Schoolbook multiply for reference (u64 limbs)
static void schoolbook_mul(u64* out, std::size_t out_len,
                           const u64* a, std::size_t na,
                           const u64* b, std::size_t nb) {
    std::memset(out, 0, out_len * sizeof(u64));
    for (std::size_t i = 0; i < na; ++i) {
        u64 carry = 0;
        for (std::size_t j = 0; j < nb && i + j < out_len; ++j) {
#if defined(_MSC_VER) && !defined(__clang__)
            u64 hi;
            u64 lo = _umul128(a[i], b[j], &hi);
            unsigned char cf, cf2;
            cf = _addcarry_u64(0, lo, carry, &lo);
            cf2 = _addcarry_u64(0, out[i+j], lo, &out[i+j]);
            carry = hi + cf + cf2;
#else
            unsigned __int128 prod = (unsigned __int128)a[i] * b[j] + out[i+j] + carry;
            out[i+j] = (u64)prod;
            carry = (u64)(prod >> 64);
#endif
        }
        if (i + nb < out_len)
            out[i + nb] += carry;
    }
}

static bool test_small_known() {
    printf("  small known values... ");

    // 3 * 7 = 21
    u64 a = 3, b = 7;
    u64 out[2] = {};
    ntt::big_multiply_u64(out, 2, &a, 1, &b, 1);
    if (out[0] != 21 || out[1] != 0) {
        printf("FAIL: 3*7 = %llu,%llu (expected 21,0)\n",
               (unsigned long long)out[0], (unsigned long long)out[1]);
        return false;
    }

    // (2^64-1) * (2^64-1) = 2^128 - 2^65 + 1
    u64 max64 = ~u64(0);
    u64 out2[2] = {};
    ntt::big_multiply_u64(out2, 2, &max64, 1, &max64, 1);
    // Expected: lo=1, hi=0xFFFFFFFFFFFFFFFE
    if (out2[0] != 1 || out2[1] != (max64 - 1)) {
        printf("FAIL: max64^2 = %llx,%llx (expected 1,%llx)\n",
               (unsigned long long)out2[0], (unsigned long long)out2[1],
               (unsigned long long)(max64 - 1));
        return false;
    }

    printf("OK\n");
    return true;
}

static bool test_vs_schoolbook(std::size_t na, std::size_t nb, unsigned seed) {
    printf("  %zu x %zu limbs (seed=%u)... ", na, nb, seed);

    std::mt19937_64 rng(seed);
    std::vector<u64> a(na), b(nb);
    for (auto& v : a) v = rng();
    for (auto& v : b) v = rng();

    std::size_t out_len = na + nb;
    std::vector<u64> out_ntt(out_len, 0);
    std::vector<u64> out_ref(out_len, 0);

    ntt::big_multiply_u64(out_ntt.data(), out_len, a.data(), na, b.data(), nb);
    schoolbook_mul(out_ref.data(), out_len, a.data(), na, b.data(), nb);

    for (std::size_t i = 0; i < out_len; ++i) {
        if (out_ntt[i] != out_ref[i]) {
            printf("FAIL at limb %zu: ntt=%llx ref=%llx\n",
                   i, (unsigned long long)out_ntt[i], (unsigned long long)out_ref[i]);
            return false;
        }
    }

    printf("OK\n");
    return true;
}

static bool test_symmetric() {
    printf("  symmetry (a*b == b*a)... ");

    std::mt19937_64 rng(999);
    std::size_t na = 100, nb = 200;
    std::vector<u64> a(na), b(nb);
    for (auto& v : a) v = rng();
    for (auto& v : b) v = rng();

    std::size_t out_len = na + nb;
    std::vector<u64> out_ab(out_len, 0), out_ba(out_len, 0);

    ntt::big_multiply_u64(out_ab.data(), out_len, a.data(), na, b.data(), nb);
    ntt::big_multiply_u64(out_ba.data(), out_len, b.data(), nb, a.data(), na);

    if (out_ab != out_ba) {
        printf("FAIL: a*b != b*a\n");
        return false;
    }

    printf("OK\n");
    return true;
}

int main() {
    printf("=== ntt::big_multiply_u64 integration tests ===\n\n");

    bool all_pass = true;

    all_pass &= test_small_known();
    all_pass &= test_symmetric();

    // Small sizes (schoolbook reference)
    all_pass &= test_vs_schoolbook(1, 1, 1);
    all_pass &= test_vs_schoolbook(2, 2, 2);
    all_pass &= test_vs_schoolbook(4, 4, 3);
    all_pass &= test_vs_schoolbook(10, 10, 4);
    all_pass &= test_vs_schoolbook(50, 50, 5);
    all_pass &= test_vs_schoolbook(100, 100, 6);
    all_pass &= test_vs_schoolbook(200, 200, 7);

    // Asymmetric
    all_pass &= test_vs_schoolbook(1, 100, 8);
    all_pass &= test_vs_schoolbook(50, 200, 9);

    // Larger sizes (still small enough for schoolbook in reasonable time)
    all_pass &= test_vs_schoolbook(500, 500, 10);
    all_pass &= test_vs_schoolbook(1000, 1000, 11);

    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
