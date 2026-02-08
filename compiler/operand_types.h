#pragma once

#include "util/common.h"

#include <mlir/IR/BuiltinTypes.h>

namespace compiler {
struct OperandTypeCache {
  CLASS_NO_COPY(OperandTypeCache);

  mlir::Type i1() const { return _types[0]; }

  mlir::Type i32() const { return _types[1]; }

  mlir::Type i64() const { return _types[2]; }

  OperandTypeCache(mlir::MLIRContext* ctx);

  private:
  std::array<mlir::Type, 4> _types;
};

} // namespace compiler