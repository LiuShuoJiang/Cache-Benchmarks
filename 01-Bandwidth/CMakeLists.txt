add_executable(01_bandwidth main.cpp)

target_link_libraries(01_bandwidth PUBLIC benchmark::benchmark)
# target_link_libraries(01_bandwidth PUBLIC TBB::tbb)
target_link_libraries(01_bandwidth PUBLIC OpenMP::OpenMP_CXX)

target_compile_options(01_bandwidth PUBLIC -ffast-math -march=native)
