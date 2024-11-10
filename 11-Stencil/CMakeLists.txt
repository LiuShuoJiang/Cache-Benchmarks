add_executable(11_stencil main.cpp)

target_link_libraries(11_stencil PUBLIC benchmark::benchmark)
target_link_libraries(11_stencil PUBLIC OpenMP::OpenMP_CXX)

target_compile_options(11_stencil PUBLIC -ffast-math -march=native)
