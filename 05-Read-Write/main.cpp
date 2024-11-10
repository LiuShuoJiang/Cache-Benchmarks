#include <vector>
#include <x86intrin.h>

#include <benchmark/benchmark.h>


constexpr size_t n = 1 << 27;
std::vector<float> a(n);

static uint32_t randomize(uint32_t i) {
    i = (i ^ 61) ^ (i >> 16);
    i *= 9;
    i ^= i << 4;
    i *= 0x27d4eb2d;
    i ^= i >> 15;
    return i;
}

void BM_read(benchmark::State &bm) {
    for (auto _: bm) {
        float ret = 0;
#pragma omp parallel for reduction(+ : ret)
        for (size_t i = 0; i < n; i++) {
            ret += a[i];
        }
        benchmark::DoNotOptimize(ret);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_read);

void BM_write(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_write);

void BM_write_streamed(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            float value = 1;
            _mm_stream_si32((int *) &a[i], *(int *) &value);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_write_streamed);

void BM_write_stream_then_read(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            float value = 1;
            _mm_stream_si32((int *) &a[i], *(int *) &value);
            // cannot optimize!!!
            benchmark::DoNotOptimize(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_write_stream_then_read);

void BM_write_streamed_ps(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i += 4) {
            _mm_stream_ps(&a[i], _mm_set1_ps(1.f));
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_write_streamed_ps);

void BM_write_streamed_ps_skipped(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i += 8) {// cannot optimize!!!
            _mm_stream_ps(&a[i], _mm_set1_ps(1.f));
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_write_streamed_ps_skipped);

void BM_read_and_write(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = a[i] + 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_read_and_write);

std::vector<int> b(n);

void BM_write0(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            b[i] = 0;
        }
        benchmark::DoNotOptimize(b);
    }
}
BENCHMARK(BM_write0);

void BM_write1(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            b[i] = 1;
        }
        benchmark::DoNotOptimize(b);
    }
}
BENCHMARK(BM_write1);

void BM_write1_streamed(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            _mm_stream_si32(&b[i], 1);
        }
        benchmark::DoNotOptimize(b);
    }
}
BENCHMARK(BM_write1_streamed);

BENCHMARK_MAIN();
