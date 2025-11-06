include(FetchContent)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.4
)
FetchContent_MakeAvailable(googlebenchmark)

include_directories(${googlebenchmark_install_SOURCE_DIR}/include)

add_executable(bench_analysis

  # bench_scc.cpp
  bench_regions.cpp
)
target_link_libraries(bench_analysis PRIVATE benchmark::benchmark psOff_recompiler)
add_dependencies(bench_analysis psOff_recompiler)