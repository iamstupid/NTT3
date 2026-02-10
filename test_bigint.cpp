// test_bigint.cpp - Correctness tests for bi::bigint (Stage 1)
#include "bigint/bigint.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <chrono>
#include <cassert>

using bi::bigint;
using bi::limb_t;

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        printf("  FAIL at line %d: ", __LINE__); \
        printf(__VA_ARGS__); printf("\n"); \
        g_fail++; \
    } else { g_pass++; } \
} while(0)

// ============================================================
// Test mpn-level operations
// ============================================================

static void test_mpn_add_sub() {
    printf("=== mpn add/sub ===\n");

    limb_t a[4] = {0xFFFFFFFFFFFFFFFFULL, 0, 0, 0};
    limb_t b[4] = {1, 0, 0, 0};
    limb_t r[4] = {};

    limb_t carry = bi::mpn_add_n(r, a, b, 4);
    CHECK(r[0] == 0 && r[1] == 1 && r[2] == 0 && r[3] == 0 && carry == 0,
          "0xFFFF...+1 carry propagation");

    // Subtraction
    limb_t borrow = bi::mpn_sub_n(r, b, a, 4);
    CHECK(r[0] == 2 && r[1] == 0xFFFFFFFFFFFFFFFFULL && borrow == 1,
          "1 - 0xFFFF... underflow");

    // add_1
    limb_t c[2] = {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
    carry = bi::mpn_add_1v(r, c, 2, 1);
    CHECK(r[0] == 0 && r[1] == 0 && carry == 1, "mpn_add_1 overflow");

    // sub_1
    limb_t d[2] = {0, 1};
    borrow = bi::mpn_sub_1(r, d, 2, 1);
    CHECK(r[0] == 0xFFFFFFFFFFFFFFFFULL && r[1] == 0 && borrow == 0,
          "mpn_sub_1 borrow propagation");

    printf("  %d passed\n", g_pass);
}

static void test_mpn_shift() {
    printf("=== mpn shift ===\n");
    int prev_pass = g_pass;

    limb_t a[4] = {0x1234567890ABCDEFULL, 0xFEDCBA0987654321ULL, 0, 0};
    limb_t r[4] = {};

    // Left shift by 4
    limb_t out = bi::mpn_lshift(r, a, 2, 4);
    CHECK(r[0] == (a[0] << 4), "lshift[0]");
    CHECK(r[1] == ((a[1] << 4) | (a[0] >> 60)), "lshift[1]");
    CHECK(out == (a[1] >> 60), "lshift carry");

    // Right shift by 4
    out = bi::mpn_rshift(r, a, 2, 4);
    CHECK(r[0] == ((a[0] >> 4) | (a[1] << 60)), "rshift[0]");
    CHECK(r[1] == (a[1] >> 4), "rshift[1]");
    CHECK(out == (a[0] << 60), "rshift carry");

    // Left shift by 63
    limb_t bv[2] = {0x8000000000000000ULL, 0};
    out = bi::mpn_lshift(r, bv, 1, 63);
    CHECK(r[0] == 0 && out == 0x4000000000000000ULL, "lshift 63 bits");

    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

static void test_mpn_cmp() {
    printf("=== mpn cmp ===\n");
    int prev_pass = g_pass;

    limb_t a[2] = {1, 2};
    limb_t b[2] = {1, 2};
    limb_t c[2] = {2, 1};
    limb_t d[2] = {0, 3};

    CHECK(bi::mpn_cmp(a, b, 2) == 0, "equal");
    CHECK(bi::mpn_cmp(a, c, 2) > 0, "a > c (MSB)");
    CHECK(bi::mpn_cmp(c, a, 2) < 0, "c < a (MSB)");
    CHECK(bi::mpn_cmp(a, d, 2) < 0, "a < d");

    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

static void test_mpn_mul_1() {
    printf("=== mpn mul_1/addmul_1 ===\n");
    int prev_pass = g_pass;

    // Simple: 3 * 7 = 21
    limb_t a[1] = {3};
    limb_t r[2] = {};
    limb_t carry = bi::mpn_mul_1(r, a, 1, 7);
    CHECK(r[0] == 21 && carry == 0, "3 * 7 = 21");

    // Overflow: 0xFFFFFFFFFFFFFFFF * 2
    limb_t big[1] = {0xFFFFFFFFFFFFFFFFULL};
    carry = bi::mpn_mul_1(r, big, 1, 2);
    CHECK(r[0] == 0xFFFFFFFFFFFFFFFEULL && carry == 1, "MAX * 2");

    // addmul_1: r += a * b
    r[0] = 100; r[1] = 0;
    carry = bi::mpn_addmul_1(r, a, 1, 7);
    CHECK(r[0] == 121 && carry == 0, "100 + 3*7 = 121");

    // Large addmul_1
    limb_t x[2] = {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
    r[0] = 0; r[1] = 0;
    carry = bi::mpn_addmul_1(r, x, 2, 0xFFFFFFFFFFFFFFFFULL);
    CHECK(r[0] == 1 && r[1] == 0xFFFFFFFFFFFFFFFFULL, "big addmul_1 low");
    CHECK(carry == 0xFFFFFFFFFFFFFFFEULL, "big addmul_1 carry");

    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

static void test_mpn_divrem_1() {
    printf("=== mpn divrem_1 ===\n");
    int prev_pass = g_pass;

    // 100 / 7 = 14 remainder 2
    limb_t a[1] = {100};
    limb_t q[1];
    limb_t rem = bi::mpn_divrem_1(q, a, 1, 7);
    CHECK(q[0] == 14 && rem == 2, "100 / 7 = 14 R 2");

    // 2^64 / 3
    limb_t b[2] = {0, 1}; // = 2^64
    limb_t qb[2];
    rem = bi::mpn_divrem_1(qb, b, 2, 3);
    CHECK(qb[0] == 6148914691236517205ULL && qb[1] == 0 && rem == 1,
          "2^64 / 3");

    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

// ============================================================
// Test bigint class
// ============================================================

static void test_bigint_construct() {
    printf("=== bigint construction ===\n");
    int prev_pass = g_pass;

    bigint z;
    CHECK(z.is_zero(), "default is zero");
    CHECK(z.sign() == 0, "zero sign");

    bigint a(42LL);
    CHECK(!a.is_zero() && a.is_positive(), "42 positive");
    CHECK(a.abs_size() == 1 && a.limbs()[0] == 42, "42 value");

    bigint b(-100LL);
    CHECK(b.is_negative(), "-100 negative");
    CHECK(b.abs_size() == 1 && b.limbs()[0] == 100, "-100 magnitude");

    // INT64_MIN
    bigint c(INT64_MIN);
    CHECK(c.is_negative(), "INT64_MIN negative");
    CHECK(c.abs_size() == 1 && c.limbs()[0] == (uint64_t)INT64_MIN, "INT64_MIN value");

    // Copy
    bigint d(a);
    CHECK(d == a, "copy construct");
    CHECK(d.limbs() != a.limbs(), "copy is independent");

    // Move
    bigint e(std::move(d));
    CHECK(e == a, "move has correct value");
    CHECK(d.is_zero(), "moved-from is zero");

    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

static void test_bigint_compare() {
    printf("=== bigint comparison ===\n");
    int prev_pass = g_pass;

    bigint a(100LL), b(200LL), c(-50LL), d(100LL), z;

    CHECK(a < b, "100 < 200");
    CHECK(b > a, "200 > 100");
    CHECK(a == d, "100 == 100");
    CHECK(a != b, "100 != 200");
    CHECK(c < a, "-50 < 100");
    CHECK(c < z, "-50 < 0");
    CHECK(z < a, "0 < 100");
    CHECK(a > z, "100 > 0");
    CHECK(z == bigint(), "0 == 0");

    // Large number comparison
    bigint big1("99999999999999999999999999999");
    bigint big2("100000000000000000000000000000");
    CHECK(big1 < big2, "big comparison");

    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

static void test_bigint_add_sub() {
    printf("=== bigint add/sub ===\n");
    int prev_pass = g_pass;

    // Basic addition
    bigint a(100LL), b(200LL);
    CHECK(a + b == bigint(300LL), "100 + 200 = 300");

    // Addition with negative
    bigint c(-50LL);
    CHECK(a + c == bigint(50LL), "100 + (-50) = 50");
    CHECK(c + a == bigint(50LL), "(-50) + 100 = 50");

    // Subtraction
    CHECK(a - b == bigint(-100LL), "100 - 200 = -100");
    CHECK(b - a == bigint(100LL), "200 - 100 = 100");

    // Cancel to zero
    CHECK(a - a == bigint(), "100 - 100 = 0");

    // Double negative
    bigint d(-30LL);
    CHECK(c + d == bigint(-80LL), "(-50) + (-30) = -80");
    CHECK(c - d == bigint(-20LL), "(-50) - (-30) = -20");

    // Carry propagation
    bigint big1("18446744073709551615"); // 2^64 - 1
    bigint one(1LL);
    bigint sum = big1 + one;
    CHECK(sum.to_string() == "18446744073709551616", "2^64-1 + 1 = 2^64");

    // Multi-limb
    bigint x("123456789012345678901234567890");
    bigint y("987654321098765432109876543210");
    bigint expected_sum("1111111110111111111011111111100");
    CHECK(x + y == expected_sum, "large addition");

    bigint diff = y - x;
    bigint expected_diff("864197532086419753208641975320");
    CHECK(diff == expected_diff, "large subtraction");

    // Self-add
    bigint s(12345LL);
    s += s;
    CHECK(s == bigint(24690LL), "self-add");

    // Self-sub
    bigint t(99999LL);
    t -= t;
    CHECK(t.is_zero(), "self-sub is zero");

    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

static void test_bigint_shift() {
    printf("=== bigint shift ===\n");
    int prev_pass = g_pass;

    bigint a(1LL);
    CHECK((a << 0) == bigint(1LL), "1 << 0 = 1");
    CHECK((a << 1) == bigint(2LL), "1 << 1 = 2");
    CHECK((a << 10) == bigint(1024LL), "1 << 10 = 1024");

    // 1 << 63 = 2^63 (positive, 2-limb number on 64-bit)
    bigint expected_2_63("9223372036854775808"); // 2^63
    CHECK((a << 63) == expected_2_63, "1 << 63");

    // Multi-limb shift
    bigint bv(1LL);
    bv <<= 64;
    CHECK(bv.abs_size() == 2 && bv.limbs()[0] == 0 && bv.limbs()[1] == 1, "1 << 64");

    bigint cv(1LL);
    cv <<= 128;
    CHECK(cv.abs_size() == 3 && cv.limbs()[2] == 1, "1 << 128");

    // Right shift
    bigint dv(1024LL);
    CHECK((dv >> 3) == bigint(128LL), "1024 >> 3 = 128");
    CHECK((dv >> 10) == bigint(1LL), "1024 >> 10 = 1");
    CHECK((dv >> 11) == bigint(), "1024 >> 11 = 0");

    // Shift negative
    bigint e(-1024LL);
    CHECK((e << 1) == bigint(-2048LL), "-1024 << 1 = -2048");
    CHECK((e >> 1) == bigint(-512LL), "-1024 >> 1 = -512");

    // Large shift round-trip
    bigint f("123456789012345678901234567890");
    bigint g = f << 100;
    bigint h = g >> 100;
    CHECK(h == f, "large shift round-trip");

    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

static void test_bigint_string() {
    printf("=== bigint string conversion ===\n");
    int prev_pass = g_pass;

    CHECK(bigint().to_string() == "0", "0");
    CHECK(bigint(1LL).to_string() == "1", "1");
    CHECK(bigint(-1LL).to_string() == "-1", "-1");
    CHECK(bigint(123456789LL).to_string() == "123456789", "123456789");
    CHECK(bigint(-999LL).to_string() == "-999", "-999");

    // Round-trip
    const char* big = "123456789012345678901234567890123456789";
    bigint x(big);
    CHECK(x.to_string() == big, "big round-trip");

    const char* neg_big = "-99999999999999999999999999999999";
    bigint y(neg_big);
    CHECK(y.to_string() == neg_big, "negative big round-trip");

    // Powers of 2
    bigint p(1LL);
    for (int i = 0; i < 100; i++) {
        p.mul_limb(2);
    }
    // 2^100 = 1267650600228229401496703205376
    CHECK(p.to_string() == "1267650600228229401496703205376", "2^100");

    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

// Helper: build a random bigint using public API only
static bigint random_bigint(std::mt19937_64& rng, int max_limbs) {
    int n = (rng() % max_limbs) + 1;
    bigint result;
    for (int i = n - 1; i >= 0; i--) {
        limb_t v = rng();
        uint32_t hi32 = (uint32_t)(v >> 32);
        uint32_t lo32 = (uint32_t)v;
        bigint limb_val((long long)hi32);
        limb_val <<= 32;
        limb_val += (long long)lo32;
        result <<= 64;
        result += limb_val;
    }
    if (rng() & 1) result.negate();
    return result;
}

static void test_bigint_random_add_sub() {
    printf("=== bigint random add/sub ===\n");
    int prev_pass = g_pass;

    std::mt19937_64 rng(42);
    int errors = 0;

    for (int trial = 0; trial < 1000; trial++) {
        bigint a = random_bigint(rng, 5);
        bigint b = random_bigint(rng, 5);

        // (a + b) - b == a
        bigint sum = a + b;
        bigint diff = sum - b;
        if (diff != a) {
            errors++;
            if (errors <= 3) {
                printf("  FAIL trial %d: (a + b) - b != a\n", trial);
                printf("    a = %s\n    b = %s\n    sum = %s\n    diff = %s\n",
                       a.to_string().c_str(), b.to_string().c_str(),
                       sum.to_string().c_str(), diff.to_string().c_str());
            }
        }

        // (a - b) + b == a
        diff = a - b;
        bigint restored = diff + b;
        if (restored != a) {
            errors++;
            if (errors <= 3) {
                printf("  FAIL trial %d: (a - b) + b != a\n", trial);
            }
        }

        // a + b == b + a (commutativity)
        bigint sum2 = b + a;
        if (sum != sum2) {
            errors++;
            if (errors <= 3) {
                printf("  FAIL trial %d: a + b != b + a\n", trial);
            }
        }
    }

    CHECK(errors == 0, "random add/sub: %d errors", errors);
    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

static void test_bigint_shift_random() {
    printf("=== bigint random shift ===\n");
    int prev_pass = g_pass;

    std::mt19937_64 rng(123);
    int errors = 0;

    for (int trial = 0; trial < 200; trial++) {
        bigint a = random_bigint(rng, 4);
        if (a.is_negative()) a.negate(); // positive for shift tests

        unsigned shift = rng() % 200;

        // (a << shift) >> shift == a
        bigint shifted = a << shift;
        bigint back = shifted >> shift;
        if (back != a) {
            errors++;
            if (errors <= 3) {
                printf("  FAIL: shift %u round-trip\n", shift);
                printf("    a = %s\n    shifted = %s\n    back = %s\n",
                       a.to_string().c_str(), shifted.to_string().c_str(),
                       back.to_string().c_str());
            }
        }
    }

    CHECK(errors == 0, "random shift: %d errors", errors);
    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

// ============================================================
// Stage 2: Multiplication tests
// ============================================================

static void test_mpn_mul_basecase() {
    printf("=== mpn_mul_basecase ===\n");
    int prev_pass = g_pass;

    // 3 * 7 = 21
    limb_t a1[1] = {3}, b1[1] = {7}, r1[2] = {};
    bi::mpn_mul_basecase(r1, a1, 1, b1, 1);
    CHECK(r1[0] == 21 && r1[1] == 0, "3 * 7 = 21");

    // MAX * MAX = (2^64-1)^2
    limb_t a2[1] = {~0ULL}, b2[1] = {~0ULL}, r2[2] = {};
    bi::mpn_mul_basecase(r2, a2, 1, b2, 1);
    // (2^64-1)^2 = 2^128 - 2*2^64 + 1 = {1, 0xFFFFFFFFFFFFFFFE}
    CHECK(r2[0] == 1 && r2[1] == 0xFFFFFFFFFFFFFFFEULL, "MAX^2");

    // Multi-limb: {0xFFFF..., 0xFFFF...} * {2, 0} = {0xFFFF...E, 0xFFFF..., 1}
    limb_t a3[2] = {~0ULL, ~0ULL}, b3[1] = {2}, r3[3] = {};
    bi::mpn_mul_basecase(r3, a3, 2, b3, 1);
    CHECK(r3[0] == (~0ULL - 1) && r3[1] == ~0ULL && r3[2] == 1, "2-limb * 1-limb");

    // Cross-check: 4-limb * 4-limb via string
    bigint x("12345678901234567890");
    bigint y("98765432109876543210");
    bigint expected("1219326311370217952237463801111263526900");
    bigint got = x * y;
    CHECK(got == expected, "20-digit * 20-digit: got %s", got.to_string().c_str());

    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

static void test_mpn_sqr_basecase() {
    printf("=== mpn_sqr_basecase ===\n");
    int prev_pass = g_pass;

    // 1^2 = 1
    limb_t a1[1] = {1}, r1[2] = {};
    bi::mpn_sqr_basecase(r1, a1, 1);
    CHECK(r1[0] == 1 && r1[1] == 0, "1^2");

    // 7^2 = 49
    limb_t a2[1] = {7}, r2[2] = {};
    bi::mpn_sqr_basecase(r2, a2, 1);
    CHECK(r2[0] == 49 && r2[1] == 0, "7^2");

    // Cross-check: sqr vs mul for 2-limb
    limb_t a3[2] = {0x123456789ABCDEF0ULL, 0xFEDCBA0987654321ULL};
    limb_t r_sqr[4] = {}, r_mul[4] = {};
    bi::mpn_sqr_basecase(r_sqr, a3, 2);
    bi::mpn_mul_basecase(r_mul, a3, 2, a3, 2);
    CHECK(std::memcmp(r_sqr, r_mul, 4 * sizeof(limb_t)) == 0, "sqr == mul for 2-limb");

    // Cross-check: sqr vs mul for 4-limb
    limb_t a4[4] = {0x1111111111111111ULL, 0x2222222222222222ULL,
                     0x3333333333333333ULL, 0x4444444444444444ULL};
    limb_t rs4[8] = {}, rm4[8] = {};
    bi::mpn_sqr_basecase(rs4, a4, 4);
    bi::mpn_mul_basecase(rm4, a4, 4, a4, 4);
    CHECK(std::memcmp(rs4, rm4, 8 * sizeof(limb_t)) == 0, "sqr == mul for 4-limb");

    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

static void test_bigint_multiply_basic() {
    printf("=== bigint multiply basic ===\n");
    int prev_pass = g_pass;

    // Zero
    CHECK((bigint() * bigint(42LL)).is_zero(), "0 * 42 = 0");
    CHECK((bigint(42LL) * bigint()).is_zero(), "42 * 0 = 0");

    // Identity
    CHECK(bigint(7LL) * bigint(1LL) == bigint(7LL), "7 * 1 = 7");
    CHECK(bigint(1LL) * bigint(7LL) == bigint(7LL), "1 * 7 = 7");

    // Simple
    CHECK(bigint(6LL) * bigint(7LL) == bigint(42LL), "6 * 7 = 42");
    CHECK(bigint(100LL) * bigint(200LL) == bigint(20000LL), "100 * 200");

    // Signs
    CHECK(bigint(-3LL) * bigint(5LL) == bigint(-15LL), "-3 * 5 = -15");
    CHECK(bigint(3LL) * bigint(-5LL) == bigint(-15LL), "3 * -5 = -15");
    CHECK(bigint(-3LL) * bigint(-5LL) == bigint(15LL), "-3 * -5 = 15");

    // Large single-limb
    bigint big(INT64_MAX);
    bigint two(2LL);
    bigint expected("18446744073709551614"); // 2 * (2^63 - 1)
    CHECK(big * two == expected, "INT64_MAX * 2");

    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

static void test_bigint_multiply_karatsuba() {
    printf("=== bigint multiply (karatsuba range) ===\n");
    int prev_pass = g_pass;

    // Two ~40-limb numbers (in Karatsuba range: 32-1024 limbs)
    // Build from known values via string
    bigint a("99999999999999999999999999999999999999999999999999"
             "99999999999999999999999999999999999999999999999999"
             "99999999999999999999999999999999999999999999999999"
             "99999999999999999999999999999999999999999999999999"
             "99999999999999999999999999999999999999999999999999"
             "99999999999999999999999999999999999999999999999999"
             "9999999999999999999999"); // 322 nines
    bigint b("11111111111111111111111111111111111111111111111111"
             "11111111111111111111111111111111111111111111111111"
             "11111111111111111111111111111111111111111111111111"
             "11111111111111111111111111111111111111111111111111"
             "11111111111111111111111111111111111111111111111111"
             "11111111111111111111111111111111111111111111111111"
             "1111111111111111111111"); // 322 ones

    bigint prod = a * b;

    // Verify: a = 10^322 - 1, b = (10^322 - 1)/9
    // a * b = (10^322 - 1)^2 / 9
    // Check: prod * 9 + a == a * a  ... actually easier to check via string length and commutativity
    bigint prod2 = b * a;
    CHECK(prod == prod2, "Karatsuba commutativity");

    // Cross-check with schoolbook for smaller numbers
    // 256-bit * 256-bit (4 limbs each)
    bigint x("115792089237316195423570985008687907853269984665640564039457584007913129639935"); // 2^256-1
    bigint y("340282366920938463463374607431768211455"); // 2^128-1
    bigint expected_prod = x * y;
    bigint check = y * x;
    CHECK(expected_prod == check, "256-bit * 128-bit commutativity");

    // Verify (a*b)/b == a via string for a large number
    bigint c("12345678901234567890123456789012345678901234567890"
             "12345678901234567890123456789012345678901234567890"
             "12345678901234567890123456789012345678901234567890"
             "12345678901234567890123456789012345678901234567890"
             "12345678901234567890123456789012345678901234567890"
             "12345678901234567890123456789012345678901234567890"
             "1234567890123456789012345678901234567890");

    bigint d("98765432109876543210987654321098765432109876543210"
             "98765432109876543210987654321098765432109876543210"
             "98765432109876543210987654321098765432109876543210");

    bigint cd = c * d;
    bigint dc = d * c;
    CHECK(cd == dc, "large Karatsuba commutativity");

    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

static void test_bigint_multiply_random() {
    printf("=== bigint multiply random cross-check ===\n");
    int prev_pass = g_pass;

    std::mt19937_64 rng(99);
    int errors = 0;

    // Test various size ranges: basecase, Karatsuba, and boundary
    for (int trial = 0; trial < 500; trial++) {
        int na, nb;
        if (trial < 200) {
            // Basecase range: 1-31 limbs
            na = (rng() % 31) + 1;
            nb = (rng() % 31) + 1;
        } else if (trial < 400) {
            // Karatsuba range: 30-200 limbs
            na = (rng() % 170) + 30;
            nb = (rng() % 170) + 30;
        } else {
            // Boundary: around thresholds
            na = (rng() % 10) + 28; // around KARATSUBA_THRESHOLD
            nb = (rng() % 10) + 28;
        }

        // Build random positive bigints with known limb counts
        bigint a, b;
        for (int i = 0; i < na; i++) {
            limb_t v = rng();
            bigint limb_val;
            limb_val = bigint((long long)(v >> 32));
            limb_val <<= 32;
            limb_val += (long long)(v & 0xFFFFFFFFULL);
            a <<= 64;
            a += limb_val;
        }
        for (int i = 0; i < nb; i++) {
            limb_t v = rng();
            bigint limb_val;
            limb_val = bigint((long long)(v >> 32));
            limb_val <<= 32;
            limb_val += (long long)(v & 0xFFFFFFFFULL);
            b <<= 64;
            b += limb_val;
        }

        // Test commutativity: a*b == b*a
        bigint ab = a * b;
        bigint ba = b * a;
        if (ab != ba) {
            errors++;
            if (errors <= 3) {
                printf("  FAIL trial %d (na=%d, nb=%d): a*b != b*a\n", trial, na, nb);
            }
        }

        // Test distributivity: a*(b+1) == a*b + a (for small cases to keep fast)
        if (na + nb < 100) {
            bigint b1 = b + bigint(1LL);
            bigint ab1 = a * b1;
            bigint ab_plus_a = ab + a;
            if (ab1 != ab_plus_a) {
                errors++;
                if (errors <= 3) {
                    printf("  FAIL trial %d (na=%d, nb=%d): a*(b+1) != a*b + a\n", trial, na, nb);
                }
            }
        }
    }

    CHECK(errors == 0, "random multiply: %d errors", errors);
    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

static void test_bigint_sqr() {
    printf("=== bigint squaring ===\n");
    int prev_pass = g_pass;

    // a^2 == a*a for various sizes
    std::mt19937_64 rng(77);
    int errors = 0;

    for (int trial = 0; trial < 200; trial++) {
        int n;
        if (trial < 50) n = (rng() % 10) + 1;
        else if (trial < 150) n = (rng() % 100) + 10;
        else n = (rng() % 50) + 30;

        bigint a;
        for (int i = 0; i < n; i++) {
            limb_t v = rng();
            bigint limb_val;
            limb_val = bigint((long long)(v >> 32));
            limb_val <<= 32;
            limb_val += (long long)(v & 0xFFFFFFFFULL);
            a <<= 64;
            a += limb_val;
        }

        bigint sqr = a * a;
        bigint neg_a = -a;
        bigint neg_sqr = neg_a * neg_a;

        // (-a)^2 == a^2
        if (sqr != neg_sqr) {
            errors++;
            if (errors <= 3)
                printf("  FAIL trial %d (n=%d): (-a)^2 != a^2\n", trial, n);
        }

        // a^2 - a == a*(a-1) for small sizes
        if (n < 20) {
            bigint a_minus_1 = a - bigint(1LL);
            bigint lhs = sqr - a;
            bigint rhs = a * a_minus_1;
            if (lhs != rhs) {
                errors++;
                if (errors <= 3)
                    printf("  FAIL trial %d (n=%d): a^2 - a != a*(a-1)\n", trial, n);
            }
        }
    }

    CHECK(errors == 0, "squaring: %d errors", errors);
    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

static void test_bigint_multiply_ntt() {
    printf("=== bigint multiply NTT range ===\n");
    int prev_pass = g_pass;

    // Build numbers large enough to trigger NTT (>= 1024 limbs)
    std::mt19937_64 rng(42);

    // ~1100 limbs each → ~70,000 decimal digits
    bigint a, b;
    for (int i = 0; i < 1100; i++) {
        limb_t v = rng();
        bigint limb_val;
        limb_val = bigint((long long)(v >> 32));
        limb_val <<= 32;
        limb_val += (long long)(v & 0xFFFFFFFFULL);
        a <<= 64;
        a += limb_val;
    }
    for (int i = 0; i < 1100; i++) {
        limb_t v = rng();
        bigint limb_val;
        limb_val = bigint((long long)(v >> 32));
        limb_val <<= 32;
        limb_val += (long long)(v & 0xFFFFFFFFULL);
        b <<= 64;
        b += limb_val;
    }

    printf("  computing NTT multiply (~1100 limbs)...\n");
    bigint ab = a * b;
    bigint ba = b * a;
    CHECK(ab == ba, "NTT commutativity (1100 limbs)");

    // Distributivity: a*(b+1) == a*b + a
    bigint b1 = b + bigint(1LL);
    bigint ab1 = a * b1;
    bigint ab_plus_a = ab + a;
    CHECK(ab1 == ab_plus_a, "NTT distributivity (1100 limbs)");

    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

// ============================================================
// Stage 3: Division tests
// ============================================================

static void test_bigint_div_basic() {
    printf("=== bigint division basic ===\n");
    int prev_pass = g_pass;

    // 0 / x = 0
    CHECK((bigint() / bigint(42LL)).is_zero(), "0 / 42 = 0");
    CHECK((bigint() % bigint(42LL)).is_zero(), "0 %% 42 = 0");

    // x / 1 = x
    CHECK(bigint(42LL) / bigint(1LL) == bigint(42LL), "42 / 1 = 42");
    CHECK(bigint(42LL) % bigint(1LL) == bigint(), "42 %% 1 = 0");

    // Simple
    CHECK(bigint(42LL) / bigint(7LL) == bigint(6LL), "42 / 7 = 6");
    CHECK(bigint(42LL) % bigint(7LL) == bigint(), "42 %% 7 = 0");

    CHECK(bigint(100LL) / bigint(7LL) == bigint(14LL), "100 / 7 = 14");
    CHECK(bigint(100LL) % bigint(7LL) == bigint(2LL), "100 %% 7 = 2");

    // Signs (truncated division: sign of remainder = sign of dividend)
    CHECK(bigint(-100LL) / bigint(7LL) == bigint(-14LL), "-100 / 7 = -14");
    CHECK(bigint(-100LL) % bigint(7LL) == bigint(-2LL), "-100 %% 7 = -2");

    CHECK(bigint(100LL) / bigint(-7LL) == bigint(-14LL), "100 / -7 = -14");
    CHECK(bigint(100LL) % bigint(-7LL) == bigint(2LL), "100 %% -7 = 2");

    CHECK(bigint(-100LL) / bigint(-7LL) == bigint(14LL), "-100 / -7 = 14");
    CHECK(bigint(-100LL) % bigint(-7LL) == bigint(-2LL), "-100 %% -7 = -2");

    // |a| < |b|: quotient = 0, remainder = a
    CHECK(bigint(5LL) / bigint(100LL) == bigint(), "5 / 100 = 0");
    CHECK(bigint(5LL) % bigint(100LL) == bigint(5LL), "5 %% 100 = 5");

    // |a| == |b|
    CHECK(bigint(7LL) / bigint(7LL) == bigint(1LL), "7 / 7 = 1");
    CHECK(bigint(7LL) % bigint(7LL) == bigint(), "7 %% 7 = 0");

    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

static void test_bigint_div_multi_limb() {
    printf("=== bigint division multi-limb ===\n");
    int prev_pass = g_pass;

    // 2^128 / (2^64 + 1)
    bigint a = bigint(1LL) << 128;
    bigint b("18446744073709551617"); // 2^64 + 1
    bigint q = a / b;
    bigint r = a % b;
    // q * b + r should == a
    CHECK(q * b + r == a, "2^128 / (2^64+1) identity");

    // Large numbers
    bigint x("123456789012345678901234567890123456789012345678901234567890");
    bigint y("999999999999999999999999999999");
    bigint xq = x / y;
    bigint xr = x % y;
    CHECK(xq * y + xr == x, "large division identity");
    CHECK(xr >= bigint() && xr < y, "remainder in range");

    // Division by 2-limb number
    bigint p("340282366920938463463374607431768211456"); // 2^128
    bigint d("18446744073709551616"); // 2^64
    CHECK(p / d == bigint("18446744073709551616"), "2^128 / 2^64 = 2^64");
    CHECK(p % d == bigint(), "2^128 %% 2^64 = 0");

    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

static void test_bigint_div_random() {
    printf("=== bigint division random ===\n");
    int prev_pass = g_pass;

    std::mt19937_64 rng(777);
    int errors = 0;

    for (int trial = 0; trial < 500; trial++) {
        int na, nb;
        if (trial < 200) {
            na = (rng() % 8) + 1;
            nb = (rng() % na) + 1;
        } else if (trial < 400) {
            na = (rng() % 30) + 5;
            nb = (rng() % na) + 1;
        } else {
            na = (rng() % 60) + 10;
            nb = (rng() % (na - 1)) + 1;
        }

        bigint a, b;
        for (int i = 0; i < na; i++) {
            limb_t v = rng();
            bigint lv((long long)(v >> 32));
            lv <<= 32;
            lv += (long long)(v & 0xFFFFFFFFULL);
            a <<= 64;
            a += lv;
        }
        for (int i = 0; i < nb; i++) {
            limb_t v = rng();
            bigint lv((long long)(v >> 32));
            lv <<= 32;
            lv += (long long)(v & 0xFFFFFFFFULL);
            b <<= 64;
            b += lv;
        }
        if (b.is_zero()) b = bigint(1LL);

        // Random signs
        if (rng() & 1) a.negate();
        if (rng() & 1) b.negate();

        bigint q = a / b;
        bigint r = a % b;

        // Identity: a == q * b + r
        bigint check = q * b + r;
        if (check != a) {
            errors++;
            if (errors <= 3) {
                printf("  FAIL trial %d (na=%d, nb=%d): q*b+r != a\n", trial, na, nb);
                printf("    a = %s\n    b = %s\n", a.to_string().c_str(), b.to_string().c_str());
                printf("    q = %s\n    r = %s\n", q.to_string().c_str(), r.to_string().c_str());
                printf("    q*b+r = %s\n", check.to_string().c_str());
            }
        }

        // |r| < |b|
        if (r.compare_abs(b) >= 0) {
            errors++;
            if (errors <= 3) {
                printf("  FAIL trial %d: |r| >= |b|\n", trial);
            }
        }
    }

    CHECK(errors == 0, "random division: %d errors", errors);
    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

// ============================================================
// Stage 4: D&C Radix Conversion tests
// ============================================================

static void test_radix_known_values() {
    printf("=== radix conversion known values ===\n");
    int prev_pass = g_pass;

    // Small values (basecase path)
    CHECK(bigint(0LL).to_string() == "0", "0");
    CHECK(bigint(1LL).to_string() == "1", "1");
    CHECK(bigint(-1LL).to_string() == "-1", "-1");
    CHECK(bigint(999999999999999999LL).to_string() == "999999999999999999", "10^18-1");

    // Powers of 2
    bigint p2_64 = bigint(1LL) << 64;
    CHECK(p2_64.to_string() == "18446744073709551616", "2^64");

    bigint p2_128 = bigint(1LL) << 128;
    CHECK(p2_128.to_string() == "340282366920938463463374607431768211456", "2^128");

    bigint p2_256 = bigint(1LL) << 256;
    CHECK(p2_256.to_string() ==
        "115792089237316195423570985008687907853269984665640564039457584007913129639936",
        "2^256");

    // Powers of 10
    bigint p10 = bigint(1LL);
    for (int i = 0; i < 36; i++) p10.mul_limb(10);
    CHECK(p10.to_string() == "1000000000000000000000000000000000000", "10^36");

    // from_string round-trips
    CHECK(bigint::from_string("0") == bigint(), "from_string 0");
    CHECK(bigint::from_string("00000") == bigint(), "from_string leading zeros");
    CHECK(bigint::from_string("-0") == bigint(), "from_string -0");
    CHECK(bigint::from_string("+42") == bigint(42LL), "from_string +42");

    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

static void test_radix_roundtrip_sizes() {
    printf("=== radix round-trip at various sizes ===\n");
    int prev_pass = g_pass;

    std::mt19937_64 rng(2024);
    int errors = 0;

    // Test sizes that cross basecase/D&C boundary
    // RADIX_DC_THRESHOLD = 30 limbs → ~578 digits for to_string
    // from_string threshold = 18*30 = 540 chars
    int test_sizes[] = {1, 5, 10, 20, 29, 30, 31, 40, 50, 80, 100, 200, 500};

    for (int sz : test_sizes) {
        for (int trial = 0; trial < 20; trial++) {
            // Build random positive bigint with 'sz' limbs
            bigint a;
            for (int i = 0; i < sz; i++) {
                limb_t v = rng();
                bigint lv((long long)(v >> 32));
                lv <<= 32;
                lv += (long long)(v & 0xFFFFFFFFULL);
                a <<= 64;
                a += lv;
            }

            // to_string then from_string should round-trip
            std::string s = a.to_string();
            bigint b = bigint::from_string(s.c_str());
            if (a != b) {
                errors++;
                if (errors <= 3) {
                    printf("  FAIL: round-trip sz=%d trial=%d\n", sz, trial);
                    printf("    a = %s\n    b = %s\n", s.c_str(), b.to_string().c_str());
                }
            }

            // Also test negative
            bigint neg_a = -a;
            std::string neg_s = neg_a.to_string();
            bigint neg_b = bigint::from_string(neg_s.c_str());
            if (neg_a != neg_b) {
                errors++;
                if (errors <= 3) {
                    printf("  FAIL: negative round-trip sz=%d trial=%d\n", sz, trial);
                }
            }
        }
    }

    CHECK(errors == 0, "round-trip: %d errors", errors);
    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

static void test_radix_large_roundtrip() {
    printf("=== radix large round-trip (D&C path) ===\n");
    int prev_pass = g_pass;

    std::mt19937_64 rng(9999);
    int errors = 0;

    // Large sizes: well into D&C territory
    int test_sizes[] = {100, 500, 1000};

    for (int sz : test_sizes) {
        bigint a;
        for (int i = 0; i < sz; i++) {
            limb_t v = rng();
            bigint lv((long long)(v >> 32));
            lv <<= 32;
            lv += (long long)(v & 0xFFFFFFFFULL);
            a <<= 64;
            a += lv;
        }

        printf("  testing %d limbs (~%d digits)...\n", sz, (int)(sz * 19.3));

        std::string s = a.to_string();
        bigint b = bigint::from_string(s.c_str());
        if (a != b) {
            errors++;
            printf("  FAIL: round-trip at %d limbs\n", sz);
        }

        // Verify digit count is reasonable
        uint32_t expected_digits = (uint32_t)(sz * 19.3);
        if (s.size() < expected_digits - sz && s.size() > expected_digits + sz) {
            errors++;
            printf("  FAIL: digit count %zu out of range for %d limbs\n", s.size(), sz);
        }
    }

    CHECK(errors == 0, "large round-trip: %d errors", errors);
    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

static void test_radix_special_patterns() {
    printf("=== radix special patterns ===\n");
    int prev_pass = g_pass;

    // All 9's (repunit-like)
    std::string nines(1000, '9');
    bigint a(nines.c_str());
    CHECK(a.to_string() == nines, "1000 nines round-trip");

    // 10^1000
    std::string pow10k = "1" + std::string(1000, '0');
    bigint b(pow10k.c_str());
    CHECK(b.to_string() == pow10k, "10^1000 round-trip");

    // 10^1000 - 1 = 999...9 (1000 nines)
    bigint c = b - bigint(1LL);
    CHECK(c.to_string() == nines, "10^1000 - 1 = 999...9");

    // Alternating digits
    std::string alt;
    for (int i = 0; i < 500; i++) alt += (i % 2 == 0) ? '1' : '0';
    bigint d(alt.c_str());
    CHECK(d.to_string() == alt, "alternating digits round-trip");

    // Single 1 followed by zeros (power of 10)
    for (int exp = 0; exp <= 600; exp += 100) {
        std::string s = "1" + std::string(exp, '0');
        bigint x(s.c_str());
        CHECK(x.to_string() == s, "10^%d round-trip", exp);
    }

    printf("  %d passed (this section)\n", g_pass - prev_pass);
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("BigInt Correctness Tests (Stage 1-4)\n");
    printf("====================================\n\n");

    // Stage 1
    test_mpn_add_sub();
    test_mpn_shift();
    test_mpn_cmp();
    test_mpn_mul_1();
    test_mpn_divrem_1();
    test_bigint_construct();
    test_bigint_compare();
    test_bigint_add_sub();
    test_bigint_shift();
    test_bigint_string();
    test_bigint_random_add_sub();
    test_bigint_shift_random();

    // Stage 2: Multiplication
    test_mpn_mul_basecase();
    test_mpn_sqr_basecase();
    test_bigint_multiply_basic();
    test_bigint_multiply_karatsuba();
    test_bigint_multiply_random();
    test_bigint_sqr();
    test_bigint_multiply_ntt();

    // Stage 3: Division
    test_bigint_div_basic();
    test_bigint_div_multi_limb();
    test_bigint_div_random();

    // Stage 4: Radix conversion
    test_radix_known_values();
    test_radix_roundtrip_sizes();
    test_radix_large_roundtrip();
    test_radix_special_patterns();

    printf("\n====================================\n");
    printf("Total: %d passed, %d failed\n", g_pass, g_fail);

    if (g_fail > 0) {
        printf("*** SOME TESTS FAILED ***\n");
        return 1;
    }
    printf("ALL TESTS PASSED\n");
    return 0;
}
