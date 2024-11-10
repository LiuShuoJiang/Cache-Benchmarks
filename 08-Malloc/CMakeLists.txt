add_executable(08_malloc main.cpp)

target_link_libraries(08_malloc PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(08_malloc PUBLIC TBB::tbb)

target_compile_options(08_malloc PUBLIC -ffast-math -march=native)
