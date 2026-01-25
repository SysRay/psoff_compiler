#pragma once

#include "instructions.h"

#include <span>

namespace compiler::ir::dialect::core {

struct MoveOp {
  static IRResult create(IROperations& ir, OpDst const& dst, OpSrc const& rhs, OperandType type);
};

struct SelectOp {
  static IRResult create(IROperations& ir, OpDst const& dst, OpSrc predicate, OpSrc srcTrue, OpSrc srcFalse, OperandType type);
};

struct YieldOp {
  static IRResult create(IROperations& ir, std::span<SsaId_t> inputs);
};

struct ReturnOp {
  static IRResult create(IROperations& ir);
};

struct DiscardOp {
  static IRResult create(IROperations& ir, OpSrc const& predicate);
};

struct BarrierOp {
  static IRResult create(IROperations& ir);
};

struct JumpAbsOp {
  static IRResult create(IROperations& ir, OpSrc const& addr);
};

struct CjumpAbsOp {
  static IRResult create(IROperations& ir, OpSrc const& predicate, OpSrc const& addr);
};

struct ConstantOp {
  static IRResult create(IROperations& ir, OpDst const& dst, ir::ConstantValue value, ir::OperandType type);
};
} // namespace compiler::ir::dialect::core