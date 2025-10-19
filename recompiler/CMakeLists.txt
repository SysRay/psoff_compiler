add_library(psOff_recompiler STATIC
  builder.cpp
  ir/debug_strings.cpp

  frontend/parser.cpp

  frontend/debug_strings.cpp
  frontend/parser/debug_strings.cpp

  frontend/parser/shader_input.cpp
  frontend/parser/shader_types.cpp
  frontend/parser/instruction_builder.cpp

  frontend/analysis/regions.cpp
  frontend/analysis/regions.cpp
  frontend/analysis/evaluate.cpp
  frontend/analysis/transform_cfg.cpp

  frontend/parser/translate/ds.cpp
  frontend/parser/translate/exp.cpp
  frontend/parser/translate/mimg.cpp
  frontend/parser/translate/mubuf.cpp
  frontend/parser/translate/mtbuf.cpp
  frontend/parser/translate/smrd.cpp
  frontend/parser/translate/sop1.cpp
  frontend/parser/translate/sop2.cpp
  frontend/parser/translate/sopc.cpp
  frontend/parser/translate/sopk.cpp
  frontend/parser/translate/sopp.cpp
  frontend/parser/translate/vintrp.cpp
  frontend/parser/translate/vop1.cpp
  frontend/parser/translate/vop2.cpp
  frontend/parser/translate/vop3.cpp
  frontend/parser/translate/vopc.cpp
)

target_include_directories(psOff_recompiler
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/alpaca/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fixed-containers/include
)
