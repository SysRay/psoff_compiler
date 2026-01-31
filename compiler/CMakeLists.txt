include(FetchContent)

option(ENABLE_DEFAULT_LOGGING "printf logs" OFF)

#todo add to llvm
# -DLLVM_ENABLE_ASSERTIONS=OFF
# -DLLVM_ENABLE_ABI_BREAKING_CHECKS=OFF

if(USE_LOCAL_LLVM)
  set(LLVM_MLIR_ROOT "D:/Source/llvm-project/build/.install")

  set(LLVM_DIR "${LLVM_MLIR_ROOT}/lib/cmake/llvm")
  set(MLIR_DIR "${LLVM_MLIR_ROOT}/lib/cmake/mlir")
else()
  # Download LLVM
  set(LLVM_DOWN_VERSION "2025Sep27")
  if(WIN32)
    set(LLVM_DOWN_TYPE "-windows")
    set(LLVM_DOWN_HASH "2dafa5436ad309d09d63e6fe53a42408de37663722433280e832c997ce461359")
  else()
    set(LLVM_DOWN_TYPE "-ubuntu")
    set(LLVM_DOWN_HASH "d28adff1c24efb27d85b72ee0cb59bdb4f2cb350338446a6d33d3cad22c9a6fe")
  endif()

  FetchContent_Declare(llvm_install
    URL "https://github.com/SysRay/llvm-project/releases/download/${LLVM_DOWN_VERSION}/release_bin${LLVM_DOWN_TYPE}.zip"
    URL_HASH SHA256=${LLVM_DOWN_HASH}
    DOWNLOAD_EXTRACT_TIMESTAMP True
  )
  FetchContent_MakeAvailable(llvm_install)

  list(APPEND CMAKE_PREFIX_PATH ${llvm_install_SOURCE_DIR})
endif()

set(LLVM_DIR
  "${llvm_install_SOURCE_DIR}/lib/cmake/llvm"
  CACHE PATH "LLVM CMake config directory" FORCE
)

set(MLIR_DIR
  "${llvm_install_SOURCE_DIR}/lib/cmake/mlir"
  CACHE PATH "MLIR CMake config directory" FORCE
)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

add_definitions(${LLVM_DEFINITIONS})
add_compile_definitions(_DEPRECATE_NONFLOATING_COMPLEX _SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING)


add_library(psOff_compiler STATIC
  builder.cpp
  frontend/shader_input.cpp
  frontend/shader_types.cpp
)

if(ENABLE_DEFAULT_LOGGING)
  target_sources(psOff_compiler PRIVATE logging_default.cpp)
endif()

target_include_directories(psOff_compiler
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/alpaca/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fixed-containers/include
  ${llvm_install_SOURCE_DIR}/include
)

target_link_directories(psOff_compiler
  PUBLIC
  ${llvm_install_SOURCE_DIR}/lib
)

target_link_libraries(
  psOff_compiler PUBLIC
  MLIRIR
  MLIRSupport
  MLIRArithDialect
  MLIRArithUtils
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRControlFlowDialect
  MLIRControlFlowToSCF
  MLIRControlFlowToSPIRV
  MLIRDialectUtils
  MLIRVectorUtils
  MLIRRewrite
  MLIRFuncDialect
  MLIRUBDialect
  MLIRSCFDialect
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRIndexDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRMemorySlotInterfaces
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRVectorDialect
  MLIRComplexDialect
  MLIRFunctionInterfaces
  MLIRCallInterfaces
  MLIRVectorInterfaces
  MLIRMaskableOpInterface
  MLIRMaskingOpInterface
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRSubsetOpInterface
  MLIRAffineTransformOps
  MLIRAffineTransforms
  MLIRVectorTransformOps
  MLIRVectorTransforms
  MLIRValueBoundsOpInterface
  MLIRDataLayoutInterfaces
  MLIRCastInterfaces
  MLIRShapedOpInterfaces
  MLIRSideEffectInterfaces
  MLIRDestinationStyleOpInterface
  MLIRParallelCombiningOpInterface
  MLIRLoopLikeInterface
  MLIRInferIntRangeInterface
  MLIRInferIntRangeCommon
  MLIRTensorTransforms
  MLIRTensorDialect
  MLIRViewLikeInterface
  MLIRPDLToPDLInterp
  MLIRRewritePDL
  MLIRAnalysis
  MLIRAffineAnalysis
  MLIRPresburger
  MLIRSPIRVDialect
  MLIRSPIRVImageInterfaces
  MLIRSPIRVConversion
  MLIRSPIRVUtils
  MLIRSPIRVTransforms
  MLIRSCFToSPIRV
  MLIRVectorToSPIRV
  MLIRArithToSPIRV
  MLIRFuncToSPIRV
  MLIRMathToSPIRV
  MLIRMemRefToSPIRV
  MLIRIndexToSPIRV
  MLIRSPIRVSerialization
  LLVMSupport
)
