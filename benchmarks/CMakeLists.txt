include(FetchContent)

set(BENCHMARK_ENABLE_ASSEMBLY_TESTS OFF)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.4
)
FetchContent_MakeAvailable(googlebenchmark)

include_directories(${googlebenchmark_install_SOURCE_DIR}/include)

# add_executable(bench_analysis
#   bench_main.cpp
#   #bench_scc.cpp
#   #bench_regions.cpp
#   bench_structure.cpp
# )
# target_link_libraries(bench_analysis PRIVATE benchmark::benchmark psOff_recompiler)
# add_dependencies(bench_analysis psOff_recompiler)


add_executable(bench_compiler
  bench_main.cpp
  bench_mlir_structure.cpp
)
target_link_libraries(bench_compiler PRIVATE benchmark::benchmark psOff_compiler)
add_dependencies(bench_compiler psOff_compiler)
