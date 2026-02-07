#if defined(_MSC_VER)
#pragma warning(disable: 4324)
#endif
#ifdef __GNUC__
#pragma GCC target("avx2")
#endif

#include "ntt/api.hpp"
#include <vector>
#include <random>
#include <chrono>
#include <cstdio>
#include <algorithm>

using namespace ntt;

int main() {
    // Test sizes from 1000 to 3000000 limbs
    std::vector<idt> sizes;
    for (idt s = 1000; s <= 3000000; ) {
        sizes.push_back(s);
        // Roughly 1.3x geometric progression for smooth curve
        idt next = (idt)(s * 1.3);
        if (next == s) next = s + 1;
        s = next;
    }
    sizes.push_back(3000000);

    std::mt19937 rng(42);

    // Warmup: trigger JIT/page faults on first call
    {
        std::vector<u32> wa(64, 1), wb(64, 1), wo(128);
        ntt::big_multiply(wo.data(), 128, wa.data(), 64, wb.data(), 64);
    }

    printf("# limbs,time_ms\n");
    for (idt n : sizes) {
        std::vector<u32> a(n), b(n);
        for (auto& x : a) x = rng();
        for (auto& x : b) x = rng();
        idt out_len = 2 * n;
        std::vector<u32> out(out_len);

        // Run multiple iterations for small sizes, fewer for large
        int iters = 1;
        if (n <= 10000) iters = 20;
        else if (n <= 100000) iters = 5;
        else if (n <= 500000) iters = 3;

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            ntt::big_multiply(out.data(), out_len, a.data(), n, b.data(), n);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        printf("%zu,%.4f\n", n, ms);
        fflush(stdout);
    }
    return 0;
}
