// bench_vs_gmp.cpp - Comprehensive benchmark: zint vs GMP
//
// Build (MSYS2 ucrt64):
//   g++ -std=c++17 -O2 -mavx2 -mbmi2 -madx -mfma -I. bench_vs_gmp.cpp
//       zint/asm/addmul_1_adx.S zint/asm/submul_1_adx.S zint/asm/mul_basecase_adx.S
//       -lgmp -o bench_vs_gmp.exe

// Include GMP first, then wrap its functions before zint macros clash.
#include <gmp.h>

// GMP mpn_ macros stomp on identically-named zint functions.
// Save the GMP function pointers we need, then undef macros.
namespace gmp {
    inline mp_limb_t addmul_1(mp_limb_t* rp, const mp_limb_t* ap, mp_size_t n, mp_limb_t b) {
        return __gmpn_addmul_1(rp, ap, n, b);
    }
    inline void mul_n(mp_limb_t* rp, const mp_limb_t* ap, const mp_limb_t* bp, mp_size_t n) {
        __gmpn_mul_n(rp, ap, bp, n);
    }
    inline mp_limb_t mul(mp_limb_t* rp, const mp_limb_t* ap, mp_size_t an,
                         const mp_limb_t* bp, mp_size_t bn) {
        return __gmpn_mul(rp, ap, an, bp, bn);
    }
    inline void sqr(mp_limb_t* rp, const mp_limb_t* ap, mp_size_t n) {
        __gmpn_sqr(rp, ap, n);
    }
    inline void tdiv_qr(mp_limb_t* qp, mp_limb_t* rp, mp_size_t qxn,
                         const mp_limb_t* np, mp_size_t nn,
                         const mp_limb_t* dp, mp_size_t dn) {
        __gmpn_tdiv_qr(qp, rp, qxn, np, nn, dp, dn);
    }
}

// Undef GMP macros that collide with zint names
#undef mpn_addmul_1
#undef mpn_submul_1
#undef mpn_mul_1
#undef mpn_mul
#undef mpn_mul_n
#undef mpn_sqr
#undef mpn_tdiv_qr
#undef mpn_add_n
#undef mpn_sub_n
#undef mpn_add_1
#undef mpn_sub_1
#undef mpn_add
#undef mpn_sub
#undef mpn_cmp
#undef mpn_copyi
#undef mpn_zero
#undef mpn_lshift
#undef mpn_rshift
#undef mpn_divrem_1

#include "zint/bigint.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

// ---- Timing utilities ----

static inline double now_ns() {
    using clk = std::chrono::high_resolution_clock;
    return (double)clk::now().time_since_epoch().count();
}

// Returns median nanoseconds per call, running at least min_iters or min_ns total.
template<typename F>
double bench(F&& fn, int min_iters = 3, double min_ns = 50e6) {
    std::vector<double> times;
    double total = 0;
    for (int i = 0; i < 200 && (i < min_iters || total < min_ns); ++i) {
        double t0 = now_ns();
        fn();
        double t1 = now_ns();
        double dt = t1 - t0;
        times.push_back(dt);
        total += dt;
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2]; // median
}

// ---- Random data ----

static std::mt19937_64 rng(42);

static void fill_random(uint64_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i)
        p[i] = rng();
}

static void fill_random_limbs(mp_limb_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i)
        p[i] = rng();
    if (n > 0) p[n-1] |= 1ULL; // ensure nonzero top
}

// ---- Formatting ----

static const char* fmt_ns(double ns) {
    static char buf[4][32];
    static int idx = 0;
    char* b = buf[idx++ & 3];
    if (ns < 1e3) snprintf(b, 32, "%.0f ns", ns);
    else if (ns < 1e6) snprintf(b, 32, "%.1f us", ns / 1e3);
    else if (ns < 1e9) snprintf(b, 32, "%.2f ms", ns / 1e6);
    else snprintf(b, 32, "%.3f s", ns / 1e9);
    return b;
}

static const char* fmt_limbs(size_t n) {
    static char buf[4][32];
    static int idx = 0;
    char* b = buf[idx++ & 3];
    if (n < 1024) snprintf(b, 32, "%zu", n);
    else if (n < 1024*1024) snprintf(b, 32, "%zuK", n / 1024);
    else snprintf(b, 32, "%zuM", n / (1024*1024));
    return b;
}

// ---- Print helpers ----

static void print_header(const char* name) {
    printf("\n=== %s ===\n", name);
    printf("%-12s  %12s  %12s  %8s\n", "Size", "zint", "GMP", "Ratio");
    printf("%-12s  %12s  %12s  %8s\n", "----", "----", "---", "-----");
}

static void print_row(const char* label, double zint_ns, double gmp_ns) {
    double ratio = zint_ns / gmp_ns;
    printf("%-12s  %12s  %12s  %7.2fx\n",
           label, fmt_ns(zint_ns), fmt_ns(gmp_ns), ratio);
    fflush(stdout);
}

// ============================================================
// Benchmarks
// ============================================================

static void bench_addmul_1(const std::vector<size_t>& sizes) {
    print_header("addmul_1 (rp[n] += ap[n] * scalar)");
    for (size_t n : sizes) {
        std::vector<uint64_t> ap(n), rp_z(n+1), rp_g(n+1);
        fill_random(ap.data(), n);
        uint64_t scalar = rng();

        double t_zint = bench([&]{
            memcpy(rp_z.data(), rp_g.data(), n * 8);
            rp_z[n] = zint::mpn_addmul_1(rp_z.data(), ap.data(), (uint32_t)n, scalar);
        });

        double t_gmp = bench([&]{
            memcpy(rp_g.data(), rp_z.data(), n * 8);
            rp_g[n] = gmp::addmul_1(rp_g.data(), ap.data(), n, scalar);
        });

        print_row(fmt_limbs(n), t_zint, t_gmp);
    }
}

static void bench_mul(const std::vector<size_t>& sizes) {
    print_header("Multiply (balanced: n x n limbs)");
    for (size_t n : sizes) {
        std::vector<uint64_t> ap(n), bp(n), rp_z(2*n), rp_g(2*n);
        fill_random(ap.data(), n);
        fill_random(bp.data(), n);

        double t_zint = bench([&]{
            zint::mpn_mul(rp_z.data(), ap.data(), (uint32_t)n, bp.data(), (uint32_t)n);
        }, 3, 30e6);

        double t_gmp = bench([&]{
            gmp::mul_n(rp_g.data(), ap.data(), bp.data(), n);
        }, 3, 30e6);

        print_row(fmt_limbs(n), t_zint, t_gmp);
    }
}

static void bench_mul_unbalanced() {
    print_header("Multiply (unbalanced: large x small)");
    struct Case { size_t an, bn; };
    Case cases[] = {
        {1024, 32}, {4096, 64}, {16384, 128}, {65536, 256},
        {4096, 1024}, {16384, 4096},
    };
    for (auto [an, bn] : cases) {
        std::vector<uint64_t> ap(an), bp(bn), rp_z(an+bn), rp_g(an+bn);
        fill_random(ap.data(), an);
        fill_random(bp.data(), bn);

        double t_zint = bench([&]{
            zint::mpn_mul(rp_z.data(), ap.data(), (uint32_t)an, bp.data(), (uint32_t)bn);
        }, 3, 30e6);

        double t_gmp = bench([&]{
            gmp::mul(rp_g.data(), ap.data(), an, bp.data(), bn);
        }, 3, 30e6);

        char label[32];
        snprintf(label, sizeof(label), "%sx%s", fmt_limbs(an), fmt_limbs(bn));
        print_row(label, t_zint, t_gmp);
    }
}

static void bench_sqr(const std::vector<size_t>& sizes) {
    print_header("Squaring (n limbs)");
    for (size_t n : sizes) {
        std::vector<uint64_t> ap(n), rp_z(2*n), rp_g(2*n);
        fill_random(ap.data(), n);

        double t_zint = bench([&]{
            zint::mpn_sqr(rp_z.data(), ap.data(), (uint32_t)n);
        }, 3, 30e6);

        double t_gmp = bench([&]{
            gmp::sqr(rp_g.data(), ap.data(), n);
        }, 3, 30e6);

        print_row(fmt_limbs(n), t_zint, t_gmp);
    }
}

static void bench_div(const std::vector<size_t>& sizes_dn) {
    print_header("Division (2*dn / dn limbs)");
    for (size_t dn : sizes_dn) {
        size_t nn = 2 * dn;
        std::vector<uint64_t> np_z(nn + 1), np_g(nn + 1), dp(dn), qp_z(nn), qp_g(nn);
        fill_random(np_z.data(), nn);
        fill_random(dp.data(), dn);
        dp[dn-1] |= (1ULL << 63); // ensure top bit set for division
        // Ensure np < dp * B^(nn-dn), i.e., np[nn-1] < dp[dn-1]
        np_z[nn-1] = dp[dn-1] - 1;
        np_z[nn] = 0;

        double t_zint = bench([&]{
            memcpy(np_z.data(), np_g.data(), nn * 8); // restore
            np_z[nn] = 0;
            zint::mpn_div_qr(qp_z.data(), np_z.data(), (uint32_t)nn, dp.data(), (uint32_t)dn);
        }, 3, 30e6);

        double t_gmp = bench([&]{
            memcpy(np_g.data(), np_z.data(), nn * 8); // restore
            np_g[nn] = 0;
            gmp::tdiv_qr(qp_g.data(), np_g.data(), 0, np_g.data(), nn, dp.data(), dn);
        }, 3, 30e6);

        print_row(fmt_limbs(dn), t_zint, t_gmp);
    }
}

static void bench_bigint_mul(const std::vector<size_t>& sizes) {
    print_header("BigInt multiply (full stack: alloc + mul + string)");
    for (size_t n : sizes) {
        // Create random bigints with n limbs
        std::vector<uint64_t> ad(n), bd(n);
        fill_random(ad.data(), n);
        fill_random(bd.data(), n);

        zint::bigint a = zint::bigint::from_limbs(ad.data(), (uint32_t)n, false);
        zint::bigint b = zint::bigint::from_limbs(bd.data(), (uint32_t)n, false);

        mpz_t ga, gb, gc;
        mpz_init(ga); mpz_init(gb); mpz_init(gc);
        mpz_import(ga, n, -1, 8, 0, 0, ad.data());
        mpz_import(gb, n, -1, 8, 0, 0, bd.data());

        double t_zint = bench([&]{
            auto c = a * b;
            (void)c;
        }, 3, 30e6);

        double t_gmp = bench([&]{
            mpz_mul(gc, ga, gb);
        }, 3, 30e6);

        print_row(fmt_limbs(n), t_zint, t_gmp);

        mpz_clear(ga); mpz_clear(gb); mpz_clear(gc);
    }
}

static void bench_to_string() {
    print_header("to_string (decimal conversion)");
    size_t sizes[] = {10, 100, 1000, 4000, 16000};
    for (size_t n : sizes) {
        std::vector<uint64_t> ad(n);
        fill_random(ad.data(), n);

        zint::bigint a = zint::bigint::from_limbs(ad.data(), (uint32_t)n, false);

        mpz_t ga;
        mpz_init(ga);
        mpz_import(ga, n, -1, 8, 0, 0, ad.data());

        std::string zstr;
        double t_zint = bench([&]{
            zstr = a.to_string();
        }, 3, 30e6);

        char* gstr = nullptr;
        double t_gmp = bench([&]{
            if (gstr) free(gstr);
            gstr = mpz_get_str(nullptr, 10, ga);
        }, 3, 30e6);
        if (gstr) free(gstr);

        char label[32];
        snprintf(label, sizeof(label), "%s (~%zuK digits)", fmt_limbs(n), zstr.size() / 1000);
        print_row(label, t_zint, t_gmp);

        mpz_clear(ga);
    }
}

static void bench_from_string() {
    print_header("from_string (decimal parsing)");
    size_t sizes[] = {10, 100, 1000, 4000, 16000};
    for (size_t n : sizes) {
        std::vector<uint64_t> ad(n);
        fill_random(ad.data(), n);

        zint::bigint a = zint::bigint::from_limbs(ad.data(), (uint32_t)n, false);
        std::string dec = a.to_string();

        mpz_t ga;
        mpz_init(ga);

        double t_zint = bench([&]{
            auto r = zint::bigint::from_string(dec.c_str());
            (void)r;
        }, 3, 30e6);

        double t_gmp = bench([&]{
            mpz_set_str(ga, dec.c_str(), 10);
        }, 3, 30e6);

        char label[32];
        snprintf(label, sizeof(label), "%s (~%zuK digits)", fmt_limbs(n), dec.size() / 1000);
        print_row(label, t_zint, t_gmp);

        mpz_clear(ga);
    }
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("zint vs GMP Benchmark\n");
    printf("=====================\n");
    printf("Ratio < 1.00 = zint faster, > 1.00 = GMP faster\n");
    fflush(stdout);

    // Size ranges for different ops
    std::vector<size_t> small_sizes   = {4, 8, 16, 32};
    std::vector<size_t> kara_sizes    = {32, 64, 128, 256, 512};
    std::vector<size_t> ntt_sizes     = {1024, 2048, 4096, 8192, 16384, 32768};
    std::vector<size_t> all_mul_sizes = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    std::vector<size_t> div_sizes     = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    std::vector<size_t> addmul_sizes  = {4, 8, 16, 32, 64, 128, 256, 1024, 4096};

    printf("\nProgress: addmul_1...\n"); fflush(stdout);
    bench_addmul_1(addmul_sizes);

    printf("\nProgress: balanced multiply...\n"); fflush(stdout);
    bench_mul(all_mul_sizes);

    printf("\nProgress: unbalanced multiply...\n"); fflush(stdout);
    bench_mul_unbalanced();

    printf("\nProgress: squaring...\n"); fflush(stdout);
    bench_sqr(all_mul_sizes);

    printf("\nProgress: division...\n"); fflush(stdout);
    bench_div(div_sizes);

    printf("\nProgress: bigint multiply (full stack)...\n"); fflush(stdout);
    bench_bigint_mul(all_mul_sizes);

    printf("\nProgress: to_string...\n"); fflush(stdout);
    bench_to_string();

    printf("\nProgress: from_string...\n"); fflush(stdout);
    bench_from_string();

    printf("\n=== DONE ===\n");
    return 0;
}
