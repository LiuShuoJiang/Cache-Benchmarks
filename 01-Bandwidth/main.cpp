#include <cmath>
#include <vector>

#include <benchmark/benchmark.h>
#include <omp.h>


constexpr size_t n = 1 << 26;

std::vector<float> a(n);

static float funcB(float x) {
    return x * (x * sqrtf(x) * (x + 1) + x * 3.14f - 1 / (x + sqrtf(x - 2) + 1)) + 42 / (2.718f - x * sqrtf(x));
}

void BM_fill(benchmark::State &bm) {
    for (auto _: bm) {
        for (size_t i = 0; i < n; ++i) {
            a[i] = 1;
        }
    }
}
BENCHMARK(BM_fill);

void BM_parallel_fill(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            a[i] = 1;
        }
    }
}
BENCHMARK(BM_parallel_fill);

void BM_sine(benchmark::State &bm) {
    for (auto _: bm) {
        for (size_t i = 0; i < n; i++) {
            a[i] = std::sin(i);
        }
    }
}
BENCHMARK(BM_sine);

void BM_parallel_sine(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            a[i] = std::sin(i);
        }
    }
}
BENCHMARK(BM_parallel_sine);

void BM_serial_add(benchmark::State &bm) {
    for (auto _: bm) {
        for (size_t i = 0; i < n; ++i) {
            a[i] = a[i] + 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_serial_add);

void BM_parallel_add(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = a[i] + 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_parallel_add);

void BM_serial_funcB(benchmark::State &bm) {
    for (auto _: bm) {
        for (size_t i = 0; i < n; i++) {
            a[i] = funcB(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_serial_funcB);

void BM_1funcB(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = funcB(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_1funcB);

void BM_2funcB(benchmark::State &bm) {
    for (auto _: bm) {
        omp_set_num_threads(2);
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = funcB(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_2funcB);

void BM_4funcB(benchmark::State &bm) {
    for (auto _: bm) {
        omp_set_num_threads(4);
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = funcB(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_4funcB);

void BM_6funcB(benchmark::State &bm) {
    for (auto _: bm) {
        omp_set_num_threads(6);
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = funcB(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_6funcB);

void BM_8funcB(benchmark::State &bm) {
    for (auto _: bm) {
        omp_set_num_threads(8);
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = funcB(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_8funcB);

void BM_10funcB(benchmark::State &bm) {
    for (auto _: bm) {
        omp_set_num_threads(10);
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = funcB(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_10funcB);

BENCHMARK_MAIN();
