#include "NdArray.h"
#include <benchmark/benchmark.h>
#include <vector>
#include <x86intrin.h>

constexpr size_t n = 1 << 9;
constexpr size_t Rows = 1 << 13;
constexpr size_t Cols = 1 << 12;

std::vector<float> arr(Rows *Cols);

static uint32_t randomize(uint32_t i) {
    i = (i ^ 61) ^ (i >> 16);
    i *= 9;
    i ^= i << 4;
    i *= 0x27d4eb2d;
    i ^= i >> 15;
    return i;
}

void BM_terrible_alloc(benchmark::State &bm) {
    for (auto _: bm) {
        std::vector<std::vector<std::vector<float>>> a(n, std::vector<std::vector<float>>(n, std::vector<float>(n)));
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_terrible_alloc);

void BM_flatten_alloc(benchmark::State &bm) {
    for (auto _: bm) {
        std::vector<float> a(n * n * n);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_flatten_alloc);

void BM_terrible(benchmark::State &bm) {
    std::vector<std::vector<std::vector<float>>> a(n, std::vector<std::vector<float>>(n, std::vector<float>(n)));
    for (auto _: bm) {
        for (int i = 0; i < n * n * n; i++) {
            uint32_t x = randomize(i) % n;
            uint32_t y = randomize(i ^ x) % n;
            uint32_t z = randomize(i ^ y) % n;
            a[x][y][z] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_terrible);

void BM_flatten(benchmark::State &bm) {
    std::vector<float> a(n * n * n);
    for (auto _: bm) {
        for (int i = 0; i < n * n * n; i++) {
            uint32_t x = randomize(i) % n;
            uint32_t y = randomize(i ^ x) % n;
            uint32_t z = randomize(i ^ y) % n;
            a[(x * n + y) * n + z] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_flatten);

void BM_row_col_loop_row_col_array(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (size_t row = 0; row < Rows; ++row) {
            for (size_t col = 0; col < Cols; ++col) {
                arr[row * Cols + col] = 1;
            }
        }
        benchmark::DoNotOptimize(arr);
    }
}
BENCHMARK(BM_row_col_loop_row_col_array);

void BM_col_row_loop_row_col_array(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (size_t col = 0; col < Cols; ++col) {
            for (size_t row = 0; row < Rows; ++row) {
                arr[row * Cols + col] = 1;
            }
        }
        benchmark::DoNotOptimize(arr);
    }
}
BENCHMARK(BM_col_row_loop_row_col_array);

void BM_row_col_loop_col_row_array(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (size_t row = 0; row < Rows; ++row) {
            for (size_t col = 0; col < Cols; ++col) {
                arr[row + col * Rows] = 1;
            }
        }
        benchmark::DoNotOptimize(arr);
    }
}
BENCHMARK(BM_row_col_loop_col_row_array);

void BM_col_row_loop_col_row_array(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (size_t col = 0; col < Cols; ++col) {
            for (size_t row = 0; row < Rows; ++row) {
                arr[row + col * Rows] = 1;
            }
        }
        benchmark::DoNotOptimize(arr);
    }
}
BENCHMARK(BM_col_row_loop_col_row_array);

void BM_stdvector(benchmark::State &bm) {
    std::vector<float> a(Rows * Cols);

    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int row = 0; row < Rows; ++row) {
            for (int col = 0; col < Cols; ++col) {
                a[row * Cols + col] = 1;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_stdvector);

void BM_ndarray(benchmark::State &bm) {
    ndarray<2, float> a(Cols, Rows);

    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int row = 0; row < Rows; ++row) {
            for (int col = 0; col < Cols; ++col) {
                a(col, row) = 1;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ndarray);

void BM_ndarray_aligned(benchmark::State &bm) {
    ndarray<2, float> a(Cols, Rows);

    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int row = 0; row < Rows; ++row) {
            for (int col = 0; col < Cols; col += 8) {
                _mm256_stream_ps(&a(col, row), _mm256_set1_ps(1));
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ndarray_aligned);

void BM_x_blur(benchmark::State &bm) {
    constexpr int nblur = 8;
    ndarray<2, float, nblur> a(Cols, Rows);
    ndarray<2, float> b(Cols, Rows);

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
BENCHMARK(BM_x_blur);


BENCHMARK_MAIN();
