#include "NdArray.h"
#include <benchmark/benchmark.h>
#include <vector>
#include <x86intrin.h>


constexpr size_t Rows = 1 << 13;
constexpr size_t Cols = 1 << 12;
constexpr int nblur = 8;

ndarray<2, float, 16> a(Cols, Rows);
ndarray<2, float> b(Cols, Rows);

void BM_copy(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int row = 0; row < Rows; ++row) {
            for (int col = 0; col < Cols; ++col) {
                b(col, row) = a(col, row);
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_copy);

void BM_copy_streamed(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int row = 0; row < Rows; ++row) {
            for (int col = 0; col < Cols; col += 4) {
                _mm_stream_ps(&b(col, row), _mm_load_ps(&a(col, row)));
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_copy_streamed);

void BM_col_blur(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int row = 0; row < Rows; ++row) {
            for (int col = 0; col < Cols; ++col) {
                float res = 0;
                for (int t = -nblur; t <= nblur; ++t) {
                    res += a(col + t, row);
                }
                b(col, row) = res;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_col_blur);

void BM_col_blur_prefetched(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int row = 0; row < Rows; ++row) {
            for (int col = 0; col < Cols; ++col) {
                _mm_prefetch(&a(col + 16, row), _MM_HINT_T0);
                float res = 0;
                for (int t = -nblur; t <= nblur; ++t) {
                    res += a(col + t, row);
                }
                b(col, row) = res;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_col_blur_prefetched);

void BM_col_blur_tiled_prefetched(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int row = 0; row < Rows; ++row) {
            for (int colBase = 0; colBase < Cols; colBase += 16) {
                _mm_prefetch(&a(colBase + 16, row), _MM_HINT_T0);

                for (int col = colBase; col < colBase + 16; ++col) {
                    float res = 0;
                    for (int t = -nblur; t <= nblur; ++t) {
                        res += a(col + t, row);
                    }
                    b(col, row) = res;
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_col_blur_tiled_prefetched);

void BM_col_blur_tiled_prefetched_streamed(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int row = 0; row < Rows; ++row) {
            for (int colBase = 0; colBase < Cols; colBase += 16) {
                _mm_prefetch(&a(colBase + 16, row), _MM_HINT_T0);

                for (int col = colBase; col < colBase + 16; col += 4) {
                    __m128 res = _mm_setzero_ps();
                    for (int t = -nblur; t <= nblur; ++t) {
                        res = _mm_add_ps(res, _mm_loadu_ps(&a(col + t, row)));
                    }
                    _mm_stream_ps(&b(col, row), res);
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_col_blur_tiled_prefetched_streamed);

BENCHMARK_MAIN();
