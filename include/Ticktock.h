#ifndef CACHE_BENCHMARKS_TICKTOCK_H
#define CACHE_BENCHMARKS_TICKTOCK_H

#include <oneapi/tbb/tick_count.h>

#define TICK(x) auto bench_##x = oneapi::tbb::tick_count::now();

#define TOCK(x) std::cout << #x ": " << (oneapi::tbb::tick_count::now() - bench_##x).seconds() << "s" << std::endl;

#endif// CACHE_BENCHMARKS_TICKTOCK_H
