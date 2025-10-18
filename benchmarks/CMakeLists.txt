include(FetchContent)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.4
)
FetchContent_MakeAvailable(googlebenchmark)

include_directories(${googlebenchmark_install_SOURCE_DIR}/include)

add_executable(SCCBench bench_scc.cpp)
target_link_libraries(SCCBench PRIVATE benchmark::benchmark psOff_recompiler)
add_dependencies(SCCBench psOff_recompiler)