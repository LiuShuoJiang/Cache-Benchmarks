#include <cstring>
#include <vector>
#include <x86intrin.h>

#include <benchmark/benchmark.h>

constexpr size_t n = 1 << 27;

std::vector<float> a(n);

void BM_ordered(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            benchmark::DoNotOptimize(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ordered);

static uint32_t randomize(uint32_t i) {
    i = (i ^ 61) ^ (i >> 16);
    i *= 9;
    i ^= i << 4;
    i *= 0x27d4eb2d;
    i ^= i >> 15;
    return i;
}

void BM_random(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            size_t r = randomize(i) % n;
            benchmark::DoNotOptimize(a[r]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_random);

void BM_random_64B(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n / 16; i++) {
            size_t r = randomize(i) % (n / 16);
            // random access among different $ lines
            // sequential access within a $ line
            for (size_t j = 0; j < 16; j++) {
                benchmark::DoNotOptimize(a[r * 16 + j]);
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_random_64B);

void BM_random_64B_prefetch(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n / 16; ++i) {
            size_t next_r = randomize(i + 64) % (n / 16);
            _mm_prefetch(&a[next_r * 16], _MM_HINT_T0);

            size_t r = randomize(i) % (n / 16);
            for (size_t j = 0; j < 16; ++j) {
                benchmark::DoNotOptimize(a[r * 16 + j]);
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_random_64B_prefetch);

void BM_random_4KB(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n / 1024; i++) {
            size_t r = randomize(i) % (n / 1024);
            for (size_t j = 0; j < 1024; j++) {
                benchmark::DoNotOptimize(a[r * 1024 + j]);
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_random_4KB);

void BM_random_4KB_aligned(benchmark::State &bm) {
    float *arr = (float *) _mm_malloc(n * sizeof(float), 4096);
    std::memset(arr, 0, sizeof(float));

    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n / 1024; ++i) {
            size_t r = randomize(i) % (n / 1024);
            for (size_t j = 0; j < 1024; ++j) {
                benchmark::DoNotOptimize(arr[r * 1024 + j]);
            }
        }
        benchmark::DoNotOptimize(arr);
    }
    _mm_free(arr);
}
BENCHMARK(BM_random_4KB_aligned);

BENCHMARK_MAIN();
