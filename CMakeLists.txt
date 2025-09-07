cmake_minimum_required(VERSION 3.15)
project(psoff_compiler)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
set(CMAKE_WINDOWS_SYMBOL_VISIBILITY_IN_STATIC_LIBRARIES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INSTALL_MESSAGE LAZY)
unset(CMAKE_IMPORT_LIBRARY_SUFFIX)

add_executable(psoff_compiler
  test/main.cpp

  ir/instructions.cpp

  frontend/shader_types.cpp
  frontend/shader_input.cpp
  frontend/debug_strings.cpp

  frontend/parser/debug_strings.cpp
  frontend/parser/parser.cpp

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

target_include_directories(psoff_compiler PRIVATE ${CMAKE_SOURCE_DIR} include)
