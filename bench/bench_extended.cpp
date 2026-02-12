// bench_extended.cpp - Extended benchmark: zint vs GMP, CSV output
//
// Geometric progression of sizes up to 1M limbs, unbalanced multiplies,
// all major operations. Outputs CSV for plotting.
//
// Build (MSYS2 ucrt64):
//   g++ -std=c++17 -O2 -mavx2 -mbmi2 -madx -mfma -I. -static
//       bench_extended.cpp zint/asm/*.obj -lgmp -o bench_extended.exe

#include <gmp.h>
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
#include <string>

// ---- Timing ----

static inline double now_ns() {
    using clk = std::chrono::high_resolution_clock;
    return (double)clk::now().time_since_epoch().count();
}

// Median of multiple runs. For small ops, batches internally.
template<typename F>
double bench(F&& fn, int min_iters = 5, double min_ns = 100e6) {
    std::vector<double> times;
    double total = 0;
    for (int i = 0; i < 300 && (i < min_iters || total < min_ns); ++i) {
        double t0 = now_ns();
        fn();
        double t1 = now_ns();
        double dt = t1 - t0;
        times.push_back(dt);
        total += dt;
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

// Batched bench for very small ops (< ~500ns). Returns ns per single call.
template<typename F>
double bench_batched(F&& fn, int batch) {
    auto t = bench([&]{ for (int i = 0; i < batch; ++i) fn(); }, 5, 100e6);
    return t / batch;
}

// ---- Random data ----

static std::mt19937_64 rng(42);

static void fill_random(uint64_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i) p[i] = rng();
}

// ---- Size generation: geometric progression ----
// Generates sizes from min_n to max_n with ~steps_per_octave steps per doubling.
// E.g., steps_per_octave=3 gives: 4, 6, 8, 12, 16, 24, 32, 48, ...
static std::vector<size_t> gen_sizes(size_t min_n, size_t max_n, int steps_per_octave = 3) {
    std::vector<size_t> sizes;
    double ratio = std::pow(2.0, 1.0 / steps_per_octave);
    double x = (double)min_n;
    size_t prev = 0;
    while ((size_t)x <= max_n) {
        size_t s = (size_t)(x + 0.5);
        if (s < min_n) s = min_n;
        // Round to multiple of 4 for alignment
        s = (s + 3) & ~(size_t)3;
        if (s != prev && s <= max_n) {
            sizes.push_back(s);
            prev = s;
        }
        x *= ratio;
    }
    if (sizes.empty() || sizes.back() < max_n) {
        size_t s = (max_n + 3) & ~(size_t)3;
        if (sizes.empty() || sizes.back() != s)
            sizes.push_back(s);
    }
    return sizes;
}

// ---- CSV output ----

static FILE* csv_file = nullptr;

static void csv_row(const char* benchmark, size_t n1, size_t n2,
                    double zint_ns, double gmp_ns) {
    if (!csv_file) return;
    fprintf(csv_file, "%s,%zu,%zu,%.1f,%.1f,%.4f\n",
            benchmark, n1, n2, zint_ns, gmp_ns, zint_ns / gmp_ns);
}

// ---- Console output ----

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

static void print_row(const char* label, double zint_ns, double gmp_ns) {
    double ratio = zint_ns / gmp_ns;
    printf("  %-20s  %12s  %12s  %7.2fx\n",
           label, fmt_ns(zint_ns), fmt_ns(gmp_ns), ratio);
    fflush(stdout);
}

static void print_header(const char* name) {
    printf("\n=== %s ===\n", name);
    printf("  %-20s  %12s  %12s  %8s\n", "Size", "zint", "GMP", "Ratio");
    printf("  %-20s  %12s  %12s  %8s\n", "----", "----", "---", "-----");
    fflush(stdout);
}

// ============================================================
// Benchmarks
// ============================================================

static void bench_addmul_1() {
    print_header("addmul_1 (rp[n] += ap[n] * scalar)");
    auto sizes = gen_sizes(4, 16384, 2);

    for (size_t n : sizes) {
        std::vector<uint64_t> ap(n), rp_z(n + 1), rp_g(n + 1);
        fill_random(ap.data(), n);
        fill_random(rp_z.data(), n);
        memcpy(rp_g.data(), rp_z.data(), n * 8);
        uint64_t scalar = rng();

        int batch = (n < 64) ? 100 : (n < 1024) ? 10 : 1;

        double t_zint, t_gmp;
        if (batch > 1) {
            t_zint = bench_batched([&]{
                rp_z[n] = zint::mpn_addmul_1(rp_z.data(), ap.data(), (uint32_t)n, scalar);
            }, batch);
            t_gmp = bench_batched([&]{
                rp_g[n] = gmp::addmul_1(rp_g.data(), ap.data(), n, scalar);
            }, batch);
        } else {
            t_zint = bench([&]{
                rp_z[n] = zint::mpn_addmul_1(rp_z.data(), ap.data(), (uint32_t)n, scalar);
            });
            t_gmp = bench([&]{
                rp_g[n] = gmp::addmul_1(rp_g.data(), ap.data(), n, scalar);
            });
        }

        char label[32]; snprintf(label, sizeof(label), "%zu", n);
        print_row(label, t_zint, t_gmp);
        csv_row("addmul_1", n, n, t_zint, t_gmp);
    }
}

static void bench_balanced_mul() {
    print_header("Balanced Multiply (n x n)");
    auto sizes = gen_sizes(4, 1048576, 3);

    for (size_t n : sizes) {
        printf("  benchmarking %zu x %zu...\r", n, n); fflush(stdout);
        std::vector<uint64_t> ap(n), bp(n), rp_z(2 * n), rp_g(2 * n);
        fill_random(ap.data(), n);
        fill_random(bp.data(), n);

        int batch = (n < 32) ? 100 : (n < 256) ? 10 : 1;

        double t_zint, t_gmp;
        if (batch > 1) {
            t_zint = bench_batched([&]{
                zint::mpn_mul(rp_z.data(), ap.data(), (uint32_t)n, bp.data(), (uint32_t)n);
            }, batch);
            t_gmp = bench_batched([&]{
                gmp::mul_n(rp_g.data(), ap.data(), bp.data(), n);
            }, batch);
        } else {
            t_zint = bench([&]{
                zint::mpn_mul(rp_z.data(), ap.data(), (uint32_t)n, bp.data(), (uint32_t)n);
            });
            t_gmp = bench([&]{
                gmp::mul_n(rp_g.data(), ap.data(), bp.data(), n);
            });
        }

        char label[32]; snprintf(label, sizeof(label), "%zu", n);
        print_row(label, t_zint, t_gmp);
        csv_row("mul_balanced", n, n, t_zint, t_gmp);
    }
}

static void bench_unbalanced_mul() {
    print_header("Unbalanced Multiply (an x bn)");

    // For each large size, test various ratios
    size_t large_sizes[] = {256, 1024, 4096, 16384, 65536, 262144, 1048576};
    int ratios[] = {2, 4, 8, 16, 32, 64};

    for (size_t an : large_sizes) {
        for (int r : ratios) {
            size_t bn = an / r;
            if (bn < 4) continue;
            // Skip if this would take forever
            if (an > 262144 && r <= 2) continue;

            printf("  benchmarking %zu x %zu...\r", an, bn); fflush(stdout);

            std::vector<uint64_t> ap(an), bp(bn), rp_z(an + bn), rp_g(an + bn);
            fill_random(ap.data(), an);
            fill_random(bp.data(), bn);

            double t_zint = bench([&]{
                zint::mpn_mul(rp_z.data(), ap.data(), (uint32_t)an, bp.data(), (uint32_t)bn);
            }, 3, 50e6);

            double t_gmp = bench([&]{
                gmp::mul(rp_g.data(), ap.data(), an, bp.data(), bn);
            }, 3, 50e6);

            char label[48];
            snprintf(label, sizeof(label), "%zu x %zu (1:%d)", an, bn, r);
            print_row(label, t_zint, t_gmp);
            csv_row("mul_unbalanced", an, bn, t_zint, t_gmp);
        }
    }
}

static void bench_squaring() {
    print_header("Squaring (n limbs)");
    auto sizes = gen_sizes(4, 1048576, 3);

    for (size_t n : sizes) {
        printf("  benchmarking sqr(%zu)...\r", n); fflush(stdout);
        std::vector<uint64_t> ap(n), rp_z(2 * n), rp_g(2 * n);
        fill_random(ap.data(), n);

        int batch = (n < 32) ? 100 : (n < 256) ? 10 : 1;

        double t_zint, t_gmp;
        if (batch > 1) {
            t_zint = bench_batched([&]{
                zint::mpn_sqr(rp_z.data(), ap.data(), (uint32_t)n);
            }, batch);
            t_gmp = bench_batched([&]{
                gmp::sqr(rp_g.data(), ap.data(), n);
            }, batch);
        } else {
            t_zint = bench([&]{
                zint::mpn_sqr(rp_z.data(), ap.data(), (uint32_t)n);
            });
            t_gmp = bench([&]{
                gmp::sqr(rp_g.data(), ap.data(), n);
            });
        }

        char label[32]; snprintf(label, sizeof(label), "%zu", n);
        print_row(label, t_zint, t_gmp);
        csv_row("sqr", n, n, t_zint, t_gmp);
    }
}

static void bench_division() {
    print_header("Division (2*dn / dn limbs)");
    auto sizes = gen_sizes(4, 32768, 3);

    for (size_t dn : sizes) {
        size_t nn = 2 * dn;
        printf("  benchmarking div(%zu / %zu)...\r", nn, dn); fflush(stdout);

        std::vector<uint64_t> np_z(nn + 1), np_g(nn + 1), dp(dn), qp_z(nn), qp_g(nn), rp_g(dn);
        fill_random(np_z.data(), nn);
        fill_random(dp.data(), dn);
        dp[dn - 1] |= (1ULL << 63);
        np_z[nn - 1] = dp[dn - 1] - 1;
        np_z[nn] = 0;
        memcpy(np_g.data(), np_z.data(), (nn + 1) * 8);

        int batch = (dn < 32) ? 100 : (dn < 256) ? 10 : 1;

        double t_zint, t_gmp;
        if (batch > 1) {
            t_zint = bench_batched([&]{
                memcpy(np_z.data(), np_g.data(), nn * 8);
                np_z[nn] = 0;
                zint::mpn_div_qr(qp_z.data(), np_z.data(), (uint32_t)nn, dp.data(), (uint32_t)dn);
            }, batch);
            t_gmp = bench_batched([&]{
                memcpy(np_g.data(), np_z.data(), nn * 8);
                np_g[nn] = 0;
                gmp::tdiv_qr(qp_g.data(), rp_g.data(), 0, np_g.data(), nn, dp.data(), dn);
            }, batch);
        } else {
            t_zint = bench([&]{
                memcpy(np_z.data(), np_g.data(), nn * 8);
                np_z[nn] = 0;
                zint::mpn_div_qr(qp_z.data(), np_z.data(), (uint32_t)nn, dp.data(), (uint32_t)dn);
            });
            t_gmp = bench([&]{
                memcpy(np_g.data(), np_z.data(), nn * 8);
                np_g[nn] = 0;
                gmp::tdiv_qr(qp_g.data(), rp_g.data(), 0, np_g.data(), nn, dp.data(), dn);
            });
        }

        char label[32]; snprintf(label, sizeof(label), "%zu", dn);
        print_row(label, t_zint, t_gmp);
        csv_row("div", dn, dn, t_zint, t_gmp);
    }
}

static void bench_bigint_mul() {
    print_header("BigInt Multiply (full-stack, balanced n x n)");
    auto sizes = gen_sizes(4, 1048576, 3);

    for (size_t n : sizes) {
        printf("  benchmarking bigint_mul(%zu)...\r", n); fflush(stdout);

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
        }, 3, 50e6);

        double t_gmp = bench([&]{
            mpz_mul(gc, ga, gb);
        }, 3, 50e6);

        char label[32]; snprintf(label, sizeof(label), "%zu", n);
        print_row(label, t_zint, t_gmp);
        csv_row("bigint_mul", n, n, t_zint, t_gmp);

        mpz_clear(ga); mpz_clear(gb); mpz_clear(gc);
    }
}

static void bench_to_string() {
    print_header("to_string (decimal)");
    auto sizes = gen_sizes(8, 65536, 2);

    for (size_t n : sizes) {
        printf("  benchmarking to_string(%zu limbs)...\r", n); fflush(stdout);

        std::vector<uint64_t> ad(n);
        fill_random(ad.data(), n);

        zint::bigint a = zint::bigint::from_limbs(ad.data(), (uint32_t)n, false);

        mpz_t ga; mpz_init(ga);
        mpz_import(ga, n, -1, 8, 0, 0, ad.data());

        std::string zstr;
        double t_zint = bench([&]{ zstr = a.to_string(); }, 3, 50e6);

        char* gstr = nullptr;
        double t_gmp = bench([&]{
            if (gstr) free(gstr);
            gstr = mpz_get_str(nullptr, 10, ga);
        }, 3, 50e6);
        if (gstr) free(gstr);

        char label[48];
        snprintf(label, sizeof(label), "%zu (%zuK dig)", n, zstr.size() / 1000);
        print_row(label, t_zint, t_gmp);
        csv_row("to_string", n, zstr.size(), t_zint, t_gmp);

        mpz_clear(ga);
    }
}

static void bench_from_string() {
    print_header("from_string (decimal)");
    auto sizes = gen_sizes(8, 65536, 2);

    for (size_t n : sizes) {
        printf("  benchmarking from_string(%zu limbs)...\r", n); fflush(stdout);

        std::vector<uint64_t> ad(n);
        fill_random(ad.data(), n);

        zint::bigint a = zint::bigint::from_limbs(ad.data(), (uint32_t)n, false);
        std::string dec = a.to_string();

        mpz_t ga; mpz_init(ga);

        double t_zint = bench([&]{
            auto r = zint::bigint::from_string(dec.c_str());
            (void)r;
        }, 3, 50e6);

        double t_gmp = bench([&]{
            mpz_set_str(ga, dec.c_str(), 10);
        }, 3, 50e6);

        char label[48];
        snprintf(label, sizeof(label), "%zu (%zuK dig)", n, dec.size() / 1000);
        print_row(label, t_zint, t_gmp);
        csv_row("from_string", n, dec.size(), t_zint, t_gmp);

        mpz_clear(ga);
    }
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    const char* csv_path = "bench_results.csv";
    if (argc > 1) csv_path = argv[1];

    csv_file = fopen(csv_path, "w");
    if (!csv_file) {
        fprintf(stderr, "Cannot open %s for writing\n", csv_path);
        return 1;
    }
    fprintf(csv_file, "benchmark,n1,n2,zint_ns,gmp_ns,ratio\n");

    printf("zint vs GMP Extended Benchmark\n");
    printf("==============================\n");
    printf("Ratio < 1.00 = zint faster, > 1.00 = GMP faster\n");
    printf("CSV output: %s\n", csv_path);
    fflush(stdout);

    printf("\n[1/8] addmul_1...\n"); fflush(stdout);
    bench_addmul_1();

    printf("\n[2/8] Balanced multiply...\n"); fflush(stdout);
    bench_balanced_mul();

    printf("\n[3/8] Unbalanced multiply...\n"); fflush(stdout);
    bench_unbalanced_mul();

    printf("\n[4/8] Squaring...\n"); fflush(stdout);
    bench_squaring();

    printf("\n[5/8] Division...\n"); fflush(stdout);
    bench_division();

    printf("\n[6/8] BigInt multiply (full stack)...\n"); fflush(stdout);
    bench_bigint_mul();

    printf("\n[7/8] to_string...\n"); fflush(stdout);
    bench_to_string();

    printf("\n[8/8] from_string...\n"); fflush(stdout);
    bench_from_string();

    fclose(csv_file);
    printf("\n=== DONE === Results in %s\n", csv_path);
    return 0;
}
