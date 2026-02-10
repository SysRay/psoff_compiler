include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.x
)

FetchContent_MakeAvailable(googletest)

include_directories(${googletest_install_SOURCE_DIR}/googletest/include)
include_directories(${googletest_install_SOURCE_DIR}/googlemock/include)

add_definitions(-fno-rtti)

add_executable(compiler_test
  test_mlir_cf_to_scf.cpp
  test_cfg.cpp
  test_promote_regs.cpp
)

target_link_libraries(compiler_test
  PRIVATE gtest gtest_main psOff_compiler
)

add_dependencies(compiler_test psOff_compiler)

add_executable(mlir-opt-psoff
  mlir-opt.cpp
)

target_link_libraries(mlir-opt-psoff
  PRIVATE  psOff_compiler   MLIROptLib
)

add_dependencies(mlir-opt-psoff psOff_compiler)
