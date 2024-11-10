#include <vector>

#include <benchmark/benchmark.h>

constexpr size_t n = 1 << 27;
std::vector<float> a(n);

void BM_original(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = a[i] * 2;
        }
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = a[i] + 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_original);

void BM_optimized(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = a[i] * 2;
            a[i] = a[i] + 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_optimized);

BENCHMARK_MAIN();
