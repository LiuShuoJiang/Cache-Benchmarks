add_executable(02_cache_basics main.cpp)

target_link_libraries(02_cache_basics PUBLIC benchmark::benchmark)
target_link_libraries(02_cache_basics PUBLIC OpenMP::OpenMP_CXX)

target_compile_options(02_cache_basics PUBLIC -ffast-math -march=native)
