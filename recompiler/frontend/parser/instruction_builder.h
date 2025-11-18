#pragma once

#include "../ir_types.h"
#include "ir/instructions.h"
#include "ir/instructions_userdata.h"

namespace compiler::ir {
class InstructionManager;
}

namespace compiler::frontend::translate {

struct OpSrc {
  eOperandKind    kind;
  OperandFlagsSrc flags {0};

  constexpr explicit OpSrc(): kind(eOperandKind::SGPR(0)) {}

  constexpr explicit OpSrc(eOperandKind kind): kind(kind) {}

  constexpr explicit OpSrc(eOperandKind kind, OperandFlagsSrc flags): kind(kind), flags(flags) {}

  constexpr explicit OpSrc(eOperandKind kind, bool negate, bool abs): kind(kind), flags(negate, abs) {}

  constexpr OpSrc& operator=(OpSrc const& other) = default;
};

struct OpDst {
  eOperandKind    kind;
  OperandFlagsDst flags {0};

  constexpr explicit OpDst(): kind(eOperandKind::SGPR(0)) {}

  constexpr explicit OpDst(eOperandKind kind): kind(kind) {}

  constexpr explicit OpDst(eOperandKind kind, uint8_t omod, bool clamp, bool negate): kind(kind), flags(omod, clamp, negate) {}

  constexpr OpDst& operator=(OpDst const& other) = default;
};

namespace create {
class IR {
  ir::InstructionManager& _ir;

  public:
  IR(ir::InstructionManager& manager): _ir(manager) {}

  InstructionId_t constantOp(OpDst dst, ir::ConstantValue, ir::OperandType type);
  InstructionId_t moveOp(OpDst dst, OpSrc src, ir::OperandType type);
  InstructionId_t selectOp(OpDst dst, OpSrc predicate, OpSrc srcTrue, OpSrc srcFalse, ir::OperandType type);
  InstructionId_t bitReverseOp(OpDst dst, OpSrc src, ir::OperandType type);
  InstructionId_t bitCountOp(OpDst dst, OpSrc src, ir::OperandType type);
  InstructionId_t bitFieldMaskOp(OpDst dst, OpSrc size, OpSrc offset, ir::OperandType type);
  InstructionId_t bitAndOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  InstructionId_t bitOrOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  InstructionId_t bitXorOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  InstructionId_t findILsbOp(OpDst dst, OpSrc src, ir::OperandType type);
  InstructionId_t findUMsbOp(OpDst dst, OpSrc src, ir::OperandType type);
  InstructionId_t findSMsbOp(OpDst dst, OpSrc src, ir::OperandType type);
  InstructionId_t signExtendI32Op(OpDst dst, OpSrc src, ir::OperandType type);
  InstructionId_t bitsetOp(OpDst dst, OpSrc src, OpSrc offset, OpSrc value, ir::OperandType type);
  InstructionId_t bitFieldInsertOp(OpDst dst, OpSrc value, OpSrc offset, OpSrc count, ir::OperandType type);
  InstructionId_t bitUIExtractOp(OpDst dst, OpSrc base, OpSrc offset, OpSrc count, ir::OperandType type);
  InstructionId_t bitSIExtractOp(OpDst dst, OpSrc base, OpSrc offset, OpSrc count, ir::OperandType type);
  InstructionId_t bitUIExtractOp(OpDst dst, OpSrc base, OpSrc compact, ir::OperandType type);
  InstructionId_t bitSIExtractOp(OpDst dst, OpSrc base, OpSrc compact, ir::OperandType type);
  InstructionId_t bitCmpOp(OpDst dst, OpSrc base, ir::OperandType type, OpSrc index);
  InstructionId_t cmpIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type, CmpIPredicate op);
  InstructionId_t cmpFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type, CmpFPredicate op);

  InstructionId_t shiftLUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  InstructionId_t shiftRUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  InstructionId_t shiftRSIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);

  // // arith
  InstructionId_t ldexpOp(OpDst dst, OpSrc vsrc, OpSrc vexp, ir::OperandType type);
  InstructionId_t addFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  InstructionId_t subFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  InstructionId_t mulFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  InstructionId_t fmaFOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  InstructionId_t fmaIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  InstructionId_t mulIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  InstructionId_t mulIExtendedOp(OpDst low, OpDst high, OpSrc src0, OpSrc src1, ir::OperandType type);
  InstructionId_t addIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  InstructionId_t addcIOp(OpDst dst, OpDst carryOut, OpSrc src0, OpSrc src1, OpSrc carryIn, ir::OperandType type);
  InstructionId_t subIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  InstructionId_t subbIOp(OpDst dst, OpDst carryOut, OpSrc src0, OpSrc src1, OpSrc carryIn, ir::OperandType type);

  InstructionId_t convFPToSIOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType);
  InstructionId_t convSIToFPOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType);
  InstructionId_t convFPToUIOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType);
  InstructionId_t convUIToFPOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType);
  InstructionId_t convSI4ToFloat(OpDst dst, OpSrc src0);

  InstructionId_t truncFOp(OpDst dst, OpSrc src0);
  InstructionId_t extFOp(OpDst dst, OpSrc src0);
  InstructionId_t packHalf2x16Op(OpDst dst, OpSrc src0, OpSrc src1);
  InstructionId_t unpackHalf2x16(OpDst low, OpDst high, OpSrc src);
  InstructionId_t packSnorm2x16Op(OpDst dst, OpSrc src0, OpSrc src1);
  InstructionId_t packUnorm2x16Op(OpDst dst, OpSrc src0, OpSrc src1);

  InstructionId_t truncOp(OpDst dst, OpSrc src0, ir::OperandType type);
  InstructionId_t ceilOp(OpDst dst, OpSrc src0, ir::OperandType type);
  InstructionId_t roundEvenOp(OpDst dst, OpSrc src0, ir::OperandType type);
  InstructionId_t fractOp(OpDst dst, OpSrc src0, ir::OperandType type);
  InstructionId_t floorOp(OpDst dst, OpSrc src0, ir::OperandType type);
  InstructionId_t rcpOp(OpDst dst, OpSrc src0, ir::OperandType type);
  InstructionId_t rsqrtOp(OpDst dst, OpSrc src0, ir::OperandType type);
  InstructionId_t sqrtOp(OpDst dst, OpSrc src0, ir::OperandType type);
  InstructionId_t exp2Op(OpDst dst, OpSrc src0);
  InstructionId_t log2Op(OpDst dst, OpSrc src0);
  InstructionId_t sinOp(OpDst dst, OpSrc src0);
  InstructionId_t cosOp(OpDst dst, OpSrc src0);
  InstructionId_t clampFMinMaxOp(OpDst dst, OpSrc src0, ir::OperandType type); ///< Clamp +-inf to +- flat max
  InstructionId_t clampFZeroOp(OpDst dst, OpSrc src0, ir::OperandType type);   ///< Clamp +-inf to Zero
  InstructionId_t frexpOp(OpDst exp, OpDst mant, OpSrc src0, ir::OperandType type);

  InstructionId_t medUIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  InstructionId_t medSIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  InstructionId_t medFOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  InstructionId_t max3UIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  InstructionId_t maxUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  InstructionId_t max3SIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  InstructionId_t maxSIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  InstructionId_t max3FOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  InstructionId_t maxFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  InstructionId_t maxNOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  InstructionId_t min3UIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  InstructionId_t minUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  InstructionId_t min3SIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  InstructionId_t minSIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  InstructionId_t min3FOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  InstructionId_t minFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  InstructionId_t minNOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);

  // // Flow control
  InstructionId_t returnOp();
  InstructionId_t discardOp(OpSrc predicate);
  InstructionId_t barrierOp();
  InstructionId_t jumpAbsOp(OpSrc addr);
  InstructionId_t cjumpAbsOp(OpSrc predicate, bool invert, OpSrc addr);
};
} // namespace create
} // namespace compiler::frontend::translate