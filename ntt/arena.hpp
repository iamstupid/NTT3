#pragma once
#include "common.hpp"
#include <vector>

namespace ntt {

// Pool allocator for NTT scratch buffers.
// Sizes are {1,3,5} * 2^k. Bins are assigned sequential indices
// in sorted size order so that bin[i+1] > bin[i]:
//   ..., 2^k, 5*2^(k-2), 3*2^(k-1), 2^(k+1), ...
//
// Alloc may return a slightly larger recycled buffer (up to ~2x).
// Two tag bits in bits 62-63 of the pointer record the bin offset,
// so dealloc can return the buffer to its correct bin.
struct NTTArena {
    // Sorted index formula (sequential for count >= 4):
    //   m=1: 3*k        where count = 2^k
    //   m=5: 3*(k+2)+1  where count = 5*2^k
    //   m=3: 3*(k+1)+2  where count = 3*2^k
    static constexpr int NUM_BINS = 3 * (MAX_LOG + 1);

    std::vector<void*> bins[NUM_BINS];

    ~NTTArena() {
        for (auto& bin : bins)
            for (void* p : bin)
                aligned_free_array<char>(static_cast<char*>(p));
    }

    // Map count = {1,3,5}*2^k to sequential sorted index.
    static int sorted_index(idt count) {
        int k = ntt_ctzll(static_cast<unsigned long long>(count));
        idt m = count >> k;
        if (m == 1) return 3 * k;
        if (m == 5) return 3 * (k + 2) + 1;
        return 3 * (k + 1) + 2;  // m == 3
    }

    // Pointer tagging: 2 bits in bits 62-63
    static constexpr int TAG_SHIFT = 62;
    static constexpr uintptr_t TAG_MASK  = uintptr_t(3) << TAG_SHIFT;
    static constexpr uintptr_t ADDR_MASK = ~TAG_MASK;

    template<typename T>
    static T* tag_ptr(T* p, int offset) {
        return reinterpret_cast<T*>(
            reinterpret_cast<uintptr_t>(p) | (uintptr_t(offset) << TAG_SHIFT));
    }

    static int get_tag(const void* p) {
        return int((reinterpret_cast<uintptr_t>(p) >> TAG_SHIFT) & 3);
    }

    // Strip tag bits to get usable pointer.
    template<typename T>
    static T* raw(T* p) {
        return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(p) & ADDR_MASK);
    }

    // Allocate count elements of type T.  Returns a TAGGED pointer.
    // Use raw() for the usable address.  Pass the tagged pointer to dealloc().
    template<typename T>
    T* alloc(idt count) {
        int idx = sorted_index(count);
        for (int off = 0; off < 4 && idx + off < NUM_BINS; ++off) {
            auto& bin = bins[idx + off];
            if (!bin.empty()) {
                void* p = bin.back();
                bin.pop_back();
                return tag_ptr(static_cast<T*>(p), off);
            }
        }
        return aligned_alloc_array<T, 64>(count);  // tag=0
    }

    // Return a (possibly tagged) pointer to its correct bin.
    template<typename T>
    void dealloc(T* p, idt requested_count) {
        int off = get_tag(p);
        int actual_idx = sorted_index(requested_count) + off;
        bins[actual_idx].push_back(raw(p));
    }

    static NTTArena& instance() {
        static thread_local NTTArena arena;
        return arena;
    }
};

} // namespace ntt
