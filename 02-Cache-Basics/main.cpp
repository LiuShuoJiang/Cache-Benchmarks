#include <vector>

// #include <omp.h>
#include <benchmark/benchmark.h>

constexpr size_t n = 1 << 28;

std::vector<float> a(n);

void BM_fill1GB(benchmark::State &bm) {
    for (auto _: bm) {
        for (size_t i = 0; i < (1 << 28); ++i) {
            a[i] = 1;
        }
    }
}
BENCHMARK(BM_fill1GB);

void BM_fill128MB(benchmark::State &bm) {
    for (auto _: bm) {
        for (size_t i = 0; i < (1 << 25); i++) {
            a[i] = 1;
        }
    }
}
BENCHMARK(BM_fill128MB);

void BM_fill16MB(benchmark::State &bm) {
    for (auto _: bm) {
        for (size_t i = 0; i < (1 << 22); i++) {
            a[i] = 1;
        }
    }
}
BENCHMARK(BM_fill16MB);

void BM_fill1MB(benchmark::State &bm) {
    for (auto _: bm) {
        for (size_t i = 0; i < (1 << 18); i++) {
            a[i] = 1;
        }
    }
}
BENCHMARK(BM_fill1MB);

void BM_fill128KB(benchmark::State &bm) {
    for (auto _: bm) {
        for (size_t i = 0; i < (1 << 15); i++) {
            a[i] = 1;
        }
    }
}
BENCHMARK(BM_fill128KB);

void BM_fill16KB(benchmark::State &bm) {
    for (auto _: bm) {
        for (size_t i = 0; i < (1 << 12); i++) {
            a[i] = 1;
        }
    }
}
BENCHMARK(BM_fill16KB);

void BM_skip1(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip1);

void BM_skip2(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i += 2) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip2);

void BM_skip4(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i += 4) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip4);

void BM_skip8(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i += 8) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip8);

void BM_skip16(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i += 16) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip16);

void BM_skip32(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i += 32) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip32);

void BM_skip64(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i += 64) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip64);

void BM_skip128(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i += 128) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip128);

BENCHMARK_MAIN();
