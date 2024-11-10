#include <iostream>
#include <vector>
#include <x86intrin.h>

#include <benchmark/benchmark.h>

#include "NdArray.h"


constexpr size_t Rows = 1 << 13;
constexpr size_t Cols = 1 << 12;
constexpr int nblur = 8;

ndarray<2, float, 16> a(Cols, Rows);
ndarray<2, float> b(Cols, Rows);

void BM_col_blur_reference(benchmark::State &bm) {
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
BENCHMARK(BM_col_blur_reference);

void BM_row_blur_original(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int row = 0; row < Rows; ++row) {
            for (int col = 0; col < Cols; ++col) {
                float res = 0;
                for (int t = -nblur; t <= nblur; ++t) {
                    res += a(col, row + t);
                }
                b(col, row) = res;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_row_blur_original);

void BM_row_blur_tiled(benchmark::State &bm) {
    constexpr int blockSize = 32;

    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int rowBase = 0; rowBase < Rows; rowBase += blockSize) {
            for (int colBase = 0; colBase < Cols; colBase += blockSize) {
                for (int row = rowBase; row < rowBase + blockSize; ++row) {
                    for (int col = colBase; col < colBase + blockSize; ++col) {
                        float res = 0;
                        for (int t = -nblur; t <= nblur; ++t) {
                            res += a(col, row + t);
                        }
                        b(col, row) = res;
                    }
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_row_blur_tiled);

void BM_row_blur_tiled_only_col(benchmark::State &bm) {
    constexpr int blockSize = 32;

    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int colBase = 0; colBase < Cols; colBase += blockSize) {
            for (int row = 0; row < Rows; ++row) {
                for (int col = colBase; col < colBase + blockSize; ++col) {
                    float res = 0;
                    for (int t = -nblur; t <= nblur; ++t) {
                        res += a(col, row + t);
                    }
                    b(col, row) = res;
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_row_blur_tiled_only_col);

void BM_row_blur_tiled_only_col_prefetched(benchmark::State &bm) {
    constexpr int blockSize = 32;

    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int colBase = 0; colBase < Cols; colBase += blockSize) {
            for (int row = 0; row < Rows; ++row) {
                for (int colTemp = colBase; colTemp < colBase + blockSize; colTemp += 16) {
                    _mm_prefetch(&a(colTemp, row + nblur), _MM_HINT_T0);

                    for (int col = colTemp; col < colTemp + 16; ++col) {
                        float res = 0;
                        for (int t = -nblur; t <= nblur; ++t) {
                            res += a(col, row + t);
                        }
                        b(col, row) = res;
                    }
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_row_blur_tiled_only_col_prefetched);

void BM_row_blur_tiled_only_col_prefetched_streamed(benchmark::State &bm) {
    constexpr int blockSize = 32;

    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int colBase = 0; colBase < Cols; colBase += blockSize) {
            for (int row = 0; row < Rows; ++row) {
                for (int colTemp = colBase; colTemp < colBase + blockSize; colTemp += 16) {
                    _mm_prefetch(&a(colTemp, row + nblur), _MM_HINT_T0);

                    for (int col = colTemp; col < colTemp + 16; ++col) {
                        float res = 0;
                        for (int t = -nblur; t <= nblur; ++t) {
                            res += a(col, row + t);
                        }
                        _mm_stream_si32((int *) &b(col, row), (int &) res);
                    }
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_row_blur_tiled_only_col_prefetched_streamed);

void BM_row_blur_tiled_only_col_prefetched_streamed_merged(benchmark::State &bm) {
    constexpr int blockSize = 32;

    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int colBase = 0; colBase < Cols; colBase += blockSize) {
            for (int row = 0; row < Rows; ++row) {
                for (int col = colBase; col < colBase + blockSize; col += 16) {
                    _mm_prefetch(&a(col, row + nblur), _MM_HINT_T0);

                    float res[16];
                    for (int offset = 0; offset < 16; ++offset) {
                        res[offset] = 0;
                        for (int t = -nblur; t <= nblur; ++t) {
                            res[offset] += a(col + offset, row + t);
                        }
                    }

                    for (int offset = 0; offset < 16; ++offset) {
                        _mm_stream_si32((int *) &b(col + offset, row), (int &) res);
                    }
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_row_blur_tiled_only_col_prefetched_streamed_merged);

void BM_row_blur_tiled_only_col_prefetched_streamed_merged_vector(benchmark::State &bm) {
    constexpr int blockSize = 32;

    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int colBase = 0; colBase < Cols; colBase += blockSize) {
            for (int row = 0; row < Rows; ++row) {
                for (int col = colBase; col < colBase + blockSize; col += 16) {
                    _mm_prefetch(&a(col, row + nblur), _MM_HINT_T0);

                    __m128 res[4];
                    for (int offset = 0; offset < 4; ++offset) {
                        res[offset] = _mm_setzero_ps();
                        for (int t = -nblur; t <= nblur; ++t) {
                            res[offset] = _mm_add_ps(res[offset], _mm_load_ps(&a(col + offset * 4, row + t)));
                        }
                    }

                    for (int offset = 0; offset < 4; ++offset) {
                        _mm_stream_ps(&b(col + offset * 4, row), res[offset]);
                    }
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_row_blur_tiled_only_col_prefetched_streamed_merged_vector);

void BM_row_blur_tiled_only_col_prefetched_streamed_merged_vector_interchanged(benchmark::State &bm) {
    constexpr int blockSize = 32;

    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int colBase = 0; colBase < Cols; colBase += blockSize) {
            for (int row = 0; row < Rows; ++row) {
                for (int col = colBase; col < colBase + blockSize; col += 16) {
                    _mm_prefetch(&a(col, row + nblur), _MM_HINT_T0);

                    __m128 res[4];
                    for (int offset = 0; offset < 4; ++offset) {
                        res[offset] = _mm_setzero_ps();
                    }
                    for (int t = -nblur; t <= nblur; ++t) {
                        for (int offset = 0; offset < 4; ++offset) {
                            res[offset] = _mm_add_ps(res[offset], _mm_load_ps(&a(col + offset * 4, row + t)));
                        }
                    }

                    for (int offset = 0; offset < 4; ++offset) {
                        _mm_stream_ps(&b(col + offset * 4, row), res[offset]);
                    }
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_row_blur_tiled_only_col_prefetched_streamed_merged_vector_interchanged);

void BM_row_blur_tiled_only_col_prefetched_streamed_merged_vector_interchanged_unrolled(benchmark::State &bm) {
    constexpr int blockSize = 32;

    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int colBase = 0; colBase < Cols; colBase += blockSize) {
            for (int row = 0; row < Rows; ++row) {
                for (int col = colBase; col < colBase + blockSize; col += 16) {
                    _mm_prefetch(&a(col, row + nblur), _MM_HINT_T0);

                    __m128 res[4];
#pragma GCC unroll 4
                    for (int offset = 0; offset < 4; ++offset) {
                        res[offset] = _mm_setzero_ps();
                    }
                    for (int t = -nblur; t <= nblur; ++t) {
#pragma GCC unroll 4
                        for (int offset = 0; offset < 4; ++offset) {
                            res[offset] = _mm_add_ps(res[offset], _mm_load_ps(&a(col + offset * 4, row + t)));
                        }
                    }

#pragma GCC unroll 4
                    for (int offset = 0; offset < 4; ++offset) {
                        _mm_stream_ps(&b(col + offset * 4, row), res[offset]);
                    }
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_row_blur_tiled_only_col_prefetched_streamed_merged_vector_interchanged_unrolled);

void BM_row_blur_tiled_only_col_prefetched_streamed_merged_vector_interchanged_unrolled_avx(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int col = 0; col < Cols; col += 32) {
            for (int row = 0; row < Rows; ++row) {
                _mm_prefetch(&a(col, row + nblur), _MM_HINT_T0);
                _mm_prefetch(&a(col + 16, row + nblur), _MM_HINT_T0);

                __m256 res[4];

#pragma GCC unroll 4
                for (int offset = 0; offset < 4; ++offset) {
                    res[offset] = _mm256_setzero_ps();
                }

                for (int t = -nblur; t <= nblur; ++t) {
#pragma GCC unroll 4
                    for (int offset = 0; offset < 4; ++offset) {
                        res[offset] = _mm256_add_ps(res[offset], _mm256_load_ps(&a(col + offset * 8, row + t)));
                    }
                }

#pragma GCC unroll 4
                for (int offset = 0; offset < 4; ++offset) {
                    _mm256_stream_ps(&b(col + offset * 8, row), res[offset]);
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_row_blur_tiled_only_col_prefetched_streamed_merged_vector_interchanged_unrolled_avx);

void BM_row_blur_tiled_only_col_prefetched_streamed_merged_vector_interchanged_unrolled_avx_forward(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int col = 0; col < Cols; col += 32) {
            for (int row = 0; row < Rows; ++row) {
                _mm_prefetch(&a(col, row + nblur + 40), _MM_HINT_T0);
                _mm_prefetch(&a(col + 16, row + nblur + 40), _MM_HINT_T0);

                __m256 res[4];

#pragma GCC unroll 4
                for (int offset = 0; offset < 4; ++offset) {
                    res[offset] = _mm256_setzero_ps();
                }

                for (int t = -nblur; t <= nblur; ++t) {
#pragma GCC unroll 4
                    for (int offset = 0; offset < 4; ++offset) {
                        res[offset] = _mm256_add_ps(res[offset], _mm256_load_ps(&a(col + offset * 8, row + t)));
                    }
                }

#pragma GCC unroll 4
                for (int offset = 0; offset < 4; ++offset) {
                    _mm256_stream_ps(&b(col + offset * 8, row), res[offset]);
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_row_blur_tiled_only_col_prefetched_streamed_merged_vector_interchanged_unrolled_avx_forward);

BENCHMARK_MAIN();
