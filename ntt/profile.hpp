#pragma once
#include "common.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace ntt {

using profile_clock = std::chrono::high_resolution_clock;

struct ProfileCounters {
    u64 api_total_ns = 0;
    u64 api_reduce_pad_ns = 0;
    u64 api_forward_ns = 0;
    u64 api_freqmul_ns = 0;
    u64 api_inverse_ns = 0;
    u64 api_crt_ns = 0;

    u64 sched_fwd_direct_ns = 0;
    u64 sched_fwd_bailey_ns = 0;
    u64 sched_bailey_prep_ns = 0;
    u64 sched_bailey_stage1_ns = 0;
    u64 sched_bailey_twiddle_ns = 0;
    u64 sched_bailey_transpose_ns = 0;
    u64 sched_bailey_stage2_ns = 0;
    u64 sched_bailey_copy_ns = 0;
    u64 sched_inv_ns = 0;
    u64 sched_freqmul_ns = 0;
};

inline ProfileCounters& profile_counters() {
    static ProfileCounters c{};
    return c;
}

inline bool profile_enabled() {
    static const bool enabled = []() {
        const char* p = std::getenv("NTT_PROFILE");
        return p && p[0] == '1';
    }();
    return enabled;
}

struct ProfileScope {
    u64* bucket;
    profile_clock::time_point t0;
    bool active;

    explicit ProfileScope(u64* b)
        : bucket(b), t0(profile_clock::now()), active(profile_enabled()) {}

    ~ProfileScope() {
        if (!active) return;
        const auto t1 = profile_clock::now();
        *bucket += (u64)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    }
};

inline void profile_reset() {
    profile_counters() = ProfileCounters{};
}

inline double ns_to_ms(u64 ns) {
    return double(ns) / 1.0e6;
}

inline void profile_dump(FILE* out = stdout) {
    const auto& c = profile_counters();
    if (!out) return;

    std::fprintf(out, "=== NTT Profile (ms) ===\n");
    std::fprintf(out, "api_total       : %.3f\n", ns_to_ms(c.api_total_ns));
    std::fprintf(out, "  reduce_pad    : %.3f\n", ns_to_ms(c.api_reduce_pad_ns));
    std::fprintf(out, "  forward       : %.3f\n", ns_to_ms(c.api_forward_ns));
    std::fprintf(out, "  freq_multiply : %.3f\n", ns_to_ms(c.api_freqmul_ns));
    std::fprintf(out, "  inverse       : %.3f\n", ns_to_ms(c.api_inverse_ns));
    std::fprintf(out, "  crt+carry     : %.3f\n", ns_to_ms(c.api_crt_ns));

    std::fprintf(out, "scheduler\n");
    std::fprintf(out, "  fwd_direct    : %.3f\n", ns_to_ms(c.sched_fwd_direct_ns));
    std::fprintf(out, "  fwd_bailey    : %.3f\n", ns_to_ms(c.sched_fwd_bailey_ns));
    std::fprintf(out, "    prep        : %.3f\n", ns_to_ms(c.sched_bailey_prep_ns));
    std::fprintf(out, "    stage1      : %.3f\n", ns_to_ms(c.sched_bailey_stage1_ns));
    std::fprintf(out, "    twiddle     : %.3f\n", ns_to_ms(c.sched_bailey_twiddle_ns));
    std::fprintf(out, "    transpose   : %.3f\n", ns_to_ms(c.sched_bailey_transpose_ns));
    std::fprintf(out, "    stage2      : %.3f\n", ns_to_ms(c.sched_bailey_stage2_ns));
    std::fprintf(out, "    copy        : %.3f\n", ns_to_ms(c.sched_bailey_copy_ns));
    std::fprintf(out, "  inverse       : %.3f\n", ns_to_ms(c.sched_inv_ns));
    std::fprintf(out, "  freq_multiply : %.3f\n", ns_to_ms(c.sched_freqmul_ns));
}

} // namespace ntt
