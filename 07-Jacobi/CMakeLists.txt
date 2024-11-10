add_executable(07_jacobi main.cpp)

target_link_libraries(07_jacobi PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(07_jacobi PUBLIC TBB::tbb)

target_compile_options(07_jacobi PUBLIC -ffast-math -march=native)
