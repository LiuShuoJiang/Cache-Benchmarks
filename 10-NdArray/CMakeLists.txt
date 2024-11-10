add_executable(10_ndarray main.cpp)

target_link_libraries(10_ndarray PUBLIC benchmark::benchmark)
target_link_libraries(10_ndarray PUBLIC OpenMP::OpenMP_CXX)

target_compile_options(10_ndarray PUBLIC -ffast-math -march=native)
