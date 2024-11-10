add_executable(06_fusion main.cpp)

target_link_libraries(06_fusion PUBLIC benchmark::benchmark)
target_link_libraries(06_fusion PUBLIC OpenMP::OpenMP_CXX)

target_compile_options(06_fusion PUBLIC -ffast-math -march=native)
