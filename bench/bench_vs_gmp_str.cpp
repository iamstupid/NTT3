// bench_vs_gmp_str.cpp - String conversion benchmarks: zint vs GMP
#include <gmp.h>
namespace gmp {} // keep namespace

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
#include <string>

static std::mt19937_64 rng(42);

static void fill_random(uint64_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i) p[i] = rng();
}

static inline double now_ns() {
    using clk = std::chrono::high_resolution_clock;
    return (double)clk::now().time_since_epoch().count();
}

template<typename F>
double bench(F&& fn, int min_iters = 3, double min_ns = 100e6) {
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
    return times[times.size() / 2];
}

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
    printf("%-24s  %12s  %12s  %7.2fx\n",
           label, fmt_ns(zint_ns), fmt_ns(gmp_ns), ratio);
    fflush(stdout);
}

int main() {
    printf("zint vs GMP: String Conversion\n");
    printf("==============================\n");
    printf("Ratio < 1.00 = zint faster, > 1.00 = GMP faster\n\n");

    printf("=== to_string (decimal) ===\n");
    printf("%-24s  %12s  %12s  %8s\n", "Size", "zint", "GMP", "Ratio");
    printf("%-24s  %12s  %12s  %8s\n", "----", "----", "---", "-----");

    size_t sizes[] = {10, 100, 500, 1000, 2000, 4000, 8000, 16000};
    for (size_t n : sizes) {
        printf("  benchmarking %zu limbs...\n", n); fflush(stdout);

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
        snprintf(label, sizeof(label), "%zu limbs (~%zuK dig)", n, zstr.size() / 1000);
        print_row(label, t_zint, t_gmp);
        mpz_clear(ga);
    }

    printf("\n=== from_string (decimal) ===\n");
    printf("%-24s  %12s  %12s  %8s\n", "Size", "zint", "GMP", "Ratio");
    printf("%-24s  %12s  %12s  %8s\n", "----", "----", "---", "-----");

    for (size_t n : sizes) {
        printf("  benchmarking %zu limbs...\n", n); fflush(stdout);

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
        snprintf(label, sizeof(label), "%zu limbs (~%zuK dig)", n, dec.size() / 1000);
        print_row(label, t_zint, t_gmp);
        mpz_clear(ga);
    }

    printf("\n=== DONE ===\n");
    return 0;
}
