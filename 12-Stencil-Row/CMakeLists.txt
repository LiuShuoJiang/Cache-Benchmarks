add_executable(12_stencil_row main.cpp)

target_link_libraries(12_stencil_row PUBLIC benchmark::benchmark)
target_link_libraries(12_stencil_row PUBLIC OpenMP::OpenMP_CXX)

target_compile_options(12_stencil_row PUBLIC -ffast-math -march=native)
