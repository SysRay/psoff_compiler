include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.x
)

FetchContent_MakeAvailable(googletest)

include_directories(${googletest_install_SOURCE_DIR}/googletest/include)
include_directories(${googletest_install_SOURCE_DIR}/googlemock/include)

add_executable(test_analysis
  test_scc.cpp
  test_regions.cpp
  test_dom.cpp
  test_restructure_to_scf.cpp
  test_alloc.cpp
)

target_link_libraries(test_analysis
  PRIVATE gtest gtest_main psOff_recompiler
)

add_dependencies(test_analysis psOff_recompiler)