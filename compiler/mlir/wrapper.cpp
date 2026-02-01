#include "custom.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Interfaces/CastInterfaces.h>
#define GET_OP_CLASSES
#define GET_ATTRDEF_CLASSES
// clang-format off
#include "psOff.td.enum.cpp.inc"
#include "psOff.td.attr.cpp.inc"
#include "psOff.td.op.cpp.inc"
#include "psOff.td.cpp.inc"
// clang-format on

#undef GET_OP_CLASSES

namespace mlir::psoff {
void PSOFFDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "psOff.td.op.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "psOff.td.attr.cpp.inc"
      >();
}

::mlir::Operation* PSOFFDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value, mlir::Type type, mlir::Location loc) {
  // if (auto intAttr = llvm::cast<mlir::IntegerAttr>(value)) {
  //   return builder.create<mlir::arith::ConstantOp>(loc, type, intAttr);
  // }
  // if (auto intAttr = llvm::cast<mlir::FloatAttr>(value)) {
  //   return builder.create<mlir::arith::ConstantOp>(loc, type, intAttr);
  // }
  return nullptr; // Fallback if unsupported constant
}
} // namespace mlir::psoff
