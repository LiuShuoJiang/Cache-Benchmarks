add_executable(04_prefetch main.cpp)

target_link_libraries(04_prefetch PUBLIC benchmark::benchmark)
target_link_libraries(04_prefetch PUBLIC OpenMP::OpenMP_CXX)

target_compile_options(04_prefetch PUBLIC -ffast-math -march=native)
