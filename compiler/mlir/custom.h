#pragma once

#include <llvm/ADT/DenseMap.h>
#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/Pass/Pass.h>

// clang-format off
#define GET_OP_CLASSES
#define GET_ATTRDEF_CLASSES
#define GET_TYPEDEF_CLASSES
#include "psOff.td.types.h.inc"
#include "psOff.td.enum.h.inc"
#include "psOff.td.attr.h.inc"
#include "psOff.td.op.h.inc"
#include "psOff.td.h.inc"
//#include "psOff.td.rewrite.h.inc"

#undef GET_OP_CLASSES
#undef GET_ATTRDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

// namespace mlir{
// #define GEN_PASS_DECL_CONVERTPSOFFTOSPIRV
// #define GEN_PASS_DEF_CONVERTPSOFFTOSPIRV
// #include "psOff.td.pass.h.inc"
// }

// clang-format on