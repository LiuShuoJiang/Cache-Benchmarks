add_executable(03_aos_soa main.cpp)

target_link_libraries(03_aos_soa PUBLIC benchmark::benchmark)
target_link_libraries(03_aos_soa PUBLIC OpenMP::OpenMP_CXX)

target_compile_options(03_aos_soa PUBLIC -ffast-math -march=native)
