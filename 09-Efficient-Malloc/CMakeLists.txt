add_executable(09_malloc_efficient main.cpp)

target_link_libraries(09_malloc_efficient PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(09_malloc_efficient PUBLIC TBB::tbb)

target_compile_options(09_malloc_efficient PUBLIC -ffast-math -march=native)
