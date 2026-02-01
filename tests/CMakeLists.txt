include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.x
)

FetchContent_MakeAvailable(googletest)

include_directories(${googletest_install_SOURCE_DIR}/googletest/include)
include_directories(${googletest_install_SOURCE_DIR}/googlemock/include)

add_executable(analysis_test
  test_scc.cpp
  test_regions.cpp
  test_dom.cpp
  test_restructure_to_scf.cpp
  test_alloc.cpp
)

target_link_libraries(analysis_test
  PRIVATE gtest gtest_main psOff_recompiler
)

add_dependencies(analysis_test psOff_recompiler)

add_executable(compiler_test
  test_mlir_cf_to_scf.cpp
  test_rgtocfg.cpp
)

target_link_libraries(compiler_test
  PRIVATE gtest gtest_main psOff_compiler
)

add_dependencies(compiler_test psOff_compiler)
