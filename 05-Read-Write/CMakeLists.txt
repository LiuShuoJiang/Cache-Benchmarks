add_executable(05_read_write main.cpp)

target_link_libraries(05_read_write PUBLIC benchmark::benchmark)
target_link_libraries(05_read_write PUBLIC OpenMP::OpenMP_CXX)

target_compile_options(05_read_write PUBLIC -ffast-math -march=native)
