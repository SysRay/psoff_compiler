# ---- paths ----
set(GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")
set(TEMPLATE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/templates")

set(MLIR_SRC_DIR "${llvm_install_SOURCE_DIR}/include")
set(MLIR_TBLGEN "${llvm_install_SOURCE_DIR}/bin/mlir-tblgen")

# ---- input ----
set(BASE_FILE "psOff.td")
set(INPUT_FILE "${TEMPLATE_DIR}/${BASE_FILE}")

file(MAKE_DIRECTORY "${GENERATED_DIR}")

# ---- outputs ----
set(OUTPUTS
  "${GENERATED_DIR}/${BASE_FILE}.h.inc"
  "${GENERATED_DIR}/${BASE_FILE}.cpp.inc"

  "${GENERATED_DIR}/${BASE_FILE}.attr.h.inc"
  "${GENERATED_DIR}/${BASE_FILE}.attr.cpp.inc"

  "${GENERATED_DIR}/${BASE_FILE}.enum.h.inc"
  "${GENERATED_DIR}/${BASE_FILE}.enum.cpp.inc"

  "${GENERATED_DIR}/${BASE_FILE}.op.h.inc"
  "${GENERATED_DIR}/${BASE_FILE}.op.cpp.inc"

  "${GENERATED_DIR}/${BASE_FILE}.rewrite.h.inc"
  "${GENERATED_DIR}/${BASE_FILE}.pass.h.inc"
)

set(TBLGEN_INCLUDES
  -I "${MLIR_SRC_DIR}"
  -I "${TEMPLATE_DIR}"
)

# ---- tablegen command ----
add_custom_command(
  OUTPUT ${OUTPUTS}
  COMMAND "${MLIR_TBLGEN}" -gen-dialect-decls "${INPUT_FILE}"
  -o "${GENERATED_DIR}/${BASE_FILE}.h.inc"
  ${TBLGEN_INCLUDES}

  COMMAND "${MLIR_TBLGEN}" -gen-dialect-defs "${INPUT_FILE}"
  -o "${GENERATED_DIR}/${BASE_FILE}.cpp.inc"
  ${TBLGEN_INCLUDES}

  COMMAND "${MLIR_TBLGEN}" -gen-attrdef-decls "${INPUT_FILE}"
  -o "${GENERATED_DIR}/${BASE_FILE}.attr.h.inc"
  ${TBLGEN_INCLUDES}

  COMMAND "${MLIR_TBLGEN}" -gen-attrdef-defs "${INPUT_FILE}"
  -o "${GENERATED_DIR}/${BASE_FILE}.attr.cpp.inc"
  ${TBLGEN_INCLUDES}

  COMMAND "${MLIR_TBLGEN}" -gen-enum-decls "${INPUT_FILE}"
  -o "${GENERATED_DIR}/${BASE_FILE}.enum.h.inc"
  ${TBLGEN_INCLUDES}

  COMMAND "${MLIR_TBLGEN}" -gen-enum-defs "${INPUT_FILE}"
  -o "${GENERATED_DIR}/${BASE_FILE}.enum.cpp.inc"
  ${TBLGEN_INCLUDES}

  COMMAND "${MLIR_TBLGEN}" -gen-op-decls "${INPUT_FILE}"
  -o "${GENERATED_DIR}/${BASE_FILE}.op.h.inc"
  ${TBLGEN_INCLUDES}

  COMMAND "${MLIR_TBLGEN}" -gen-op-defs "${INPUT_FILE}"
  -o "${GENERATED_DIR}/${BASE_FILE}.op.cpp.inc"
  ${TBLGEN_INCLUDES}

  COMMAND "${MLIR_TBLGEN}" -gen-rewriters "${INPUT_FILE}"
  -o "${GENERATED_DIR}/${BASE_FILE}.rewrite.h.inc"
  ${TBLGEN_INCLUDES}

  COMMAND "${MLIR_TBLGEN}" -gen-pass-decls "${INPUT_FILE}"
  -o "${GENERATED_DIR}/${BASE_FILE}.pass.h.inc"
  ${TBLGEN_INCLUDES}

  DEPENDS "${INPUT_FILE}"
  COMMENT "Generating MLIR TableGen files for ${BASE_FILE}"
  VERBATIM
)

# ---- target to hook into build ----
add_custom_target(psOffGpu-tablegen
  DEPENDS ${OUTPUTS}
)

add_definitions(-fno-rtti)

add_library(mlir_custom STATIC wrapper.cpp)
add_dependencies(mlir_custom psOffGpu-tablegen)

target_include_directories(mlir_custom PUBLIC ${CMAKE_BINARY_DIR}/generated ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
