#include <cstdlib>
#include <iostream>
#include <vector>
#include <x86intrin.h>

#include <oneapi/tbb/cache_aligned_allocator.h>

#include "AlignedAlloc.h"
#include "Ticktock.h"

constexpr size_t n = 1 << 29;

void alloc_original() {
    {
        TICK(first_alloc_original)
        std::vector<int> arr(n);
        TOCK(first_alloc_original)
    }

    {
        TICK(second_alloc_original)
        std::vector<int> arr(n);
        TOCK(second_alloc_original)
    }

    std::cout << std::endl;
}

void alloc_tbb() {
    {
        TICK(first_alloc_tbb)
        std::vector<int, tbb::cache_aligned_allocator<int>> arr(n);
        TOCK(first_alloc_tbb)
    }

    {
        TICK(second_alloc_tbb)
        std::vector<int, oneapi::tbb::cache_aligned_allocator<int>> arr(n);
        TOCK(second_alloc_tbb)
    }

    std::cout << std::endl;
}

void is_aligned_64B() {
    std::cout << std::boolalpha;
    for (int i = 0; i < 5; ++i) {
        std::vector<int> arr(n);
        bool is_aligned = (uintptr_t) arr.data() % 64 == 0;
        std::cout << "std64: " << is_aligned << std::endl;
    }

    for (int i = 0; i < 5; ++i) {
        std::vector<int, oneapi::tbb::cache_aligned_allocator<int>> arr(n);
        bool is_aligned = (uintptr_t) arr.data() % 64 == 0;
        std::cout << "tbb64: " << is_aligned << std::endl;
    }

    std::cout << std::endl;
}

void is_aligned_16B() {
    std::cout << std::boolalpha;
    for (int i = 0; i < 5; i++) {
        std::vector<int> arr(n);
        bool is_aligned = (uintptr_t) arr.data() % 16 == 0;
        std::cout << "std16: " << is_aligned << std::endl;
    }

    for (int i = 0; i < 5; i++) {
        auto arr = (int *) malloc(n * sizeof(int));
        bool is_aligned = (uintptr_t) arr % 16 == 0;
        std::cout << "malloc16: " << is_aligned << std::endl;
        free(arr);
    }

    std::cout << std::endl;
}

void force_aligned_4KB() {
    std::cout << std::boolalpha;
    for (int i = 0; i < 5; ++i) {
        auto arr = (int *) _mm_malloc(n * sizeof(int), 4096);
        bool is_aligned = (uintptr_t) arr % 4096 == 0;
        std::cout << "_mm_malloc: " << is_aligned << std::endl;
        _mm_free(arr);
    }

    for (int i = 0; i < 5; ++i) {
        auto arr = (int *) aligned_alloc(4096, n * sizeof(int));
        bool is_aligned = (uintptr_t) arr % 4096 == 0;
        std::cout << "aligned_alloc: " << is_aligned << std::endl;
        free(arr);
    }

    std::cout << std::endl;
}

void aligned_allocator() {
    std::cout << std::boolalpha;
    for (int i = 0; i < 5; ++i) {
        std::vector<int, AlignedAllocator<int>> arr(n);
        bool is_aligned = (uintptr_t) arr.data() % 64 == 0;
        std::cout << "aligned_alloc 64: " << is_aligned << std::endl;
    }

    for (int i = 0; i < 5; i++) {
        std::vector<int, AlignedAllocator<int, 4096>> arr(n);
        bool is_aligned = (uintptr_t) arr.data() % 4096 == 0;
        std::cout << "aligned_alloc 4096: " << is_aligned << std::endl;
    }

    std::cout << std::endl;
}

float testing_func_0(int sz) {
    std::vector<float> tmp;
    for (int i = 0; i < n; i++) {
        tmp.push_back(i / 15 * 2.71828f);
    }
    std::reverse(tmp.begin(), tmp.end());
    float ret = tmp[32];
    return ret;
}

float testing_func_pooling(int sz) {
    static thread_local std::vector<float> tmp;
    for (int i = 0; i < n; i++) {
        tmp.push_back(i / 15 * 2.71828f);
    }
    std::reverse(tmp.begin(), tmp.end());
    float ret = tmp[32];
    return ret;
}

void temp_unoptimized() {
    const size_t sz = 1 << 25;
    TICK(first_call_original)
    std::cout << "Unoptimized: " << testing_func_0(sz) << std::endl;
    TOCK(first_call_original)

    TICK(second_call_original)
    std::cout << "Unoptimized: " << testing_func_0(sz - 1) << std::endl;
    TOCK(second_call_original)

    std::cout << std::endl;
}

void temp_pooling() {
    const size_t sz = 1 << 25;
    TICK(first_call_pool)
    std::cout << "Pooling: " << testing_func_pooling(sz) << std::endl;
    TOCK(first_call_pool)

    TICK(second_call_pool)
    std::cout << "Pooling: " << testing_func_pooling(sz - 1) << std::endl;
    TOCK(second_call_pool)

    std::cout << std::endl;
}

int main() {
    alloc_original();
    alloc_tbb();

    is_aligned_64B();
    is_aligned_16B();
    force_aligned_4KB();
    aligned_allocator();

    temp_unoptimized();
    temp_pooling();

    return 0;
}
