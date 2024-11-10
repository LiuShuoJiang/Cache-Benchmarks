cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)

project(Cache_Benchmarks LANGUAGES CXX)

include(cmake/CPM.cmake)

CPMAddPackage(
        NAME benchmark
        GITHUB_REPOSITORY google/benchmark
        VERSION 1.9.0
        OPTIONS
        "BENCHMARK_ENABLE_TESTING Off"
        # "BENCHMARK_USE_LIBCXX ON"
)

if (benchmark_ADDED)
    # patch benchmark target
    set_target_properties(benchmark PROPERTIES CXX_STANDARD 20)
endif()

find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(include)

add_subdirectory(01-Bandwidth)
add_subdirectory(02-Cache-Basics)
add_subdirectory(03-AOS-SOA)
add_subdirectory(04-Prefetch)
add_subdirectory(05-Read-Write)
add_subdirectory(06-Fusion)
add_subdirectory(07-Jacobi)
add_subdirectory(08-Malloc)
add_subdirectory(09-Efficient-Malloc)
add_subdirectory(10-NdArray)
add_subdirectory(11-Stencil)
add_subdirectory(12-Stencil-Row)
