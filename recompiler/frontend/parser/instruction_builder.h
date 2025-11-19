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
  SsaId_t         ssa = {};

  constexpr explicit OpSrc(): kind(eOperandKind::SGPR(0)) {}

  constexpr explicit OpSrc(eOperandKind kind): kind(kind) {}

  constexpr explicit OpSrc(eOperandKind kind, OperandFlagsSrc flags): kind(kind), flags(flags) {}

  constexpr explicit OpSrc(eOperandKind kind, bool negate, bool abs): kind(kind), flags(negate, abs) {}

  constexpr explicit OpSrc(SsaId_t op, bool negate = false, bool abs = false): kind(eOperandKind::Unset()), ssa(op), flags(negate, abs) {}

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
class IRResult {
  public:
  IRResult(ir::InstructionManager& ir, InstructionId_t id): _id(id), _ir(ir) {}

  operator SsaId_t() const { return _ir.getDef(_id, 0); }

  operator InstructionId_t() const { return _id; }

  private:
  ir::InstructionManager& _ir;
  InstructionId_t         _id;
};

class IRBuilder {
  ir::InstructionManager& _ir;
  bool                    _isVirtual;

  ir::OutputOperand& create(ir::OutputOperand& lhs, OpDst const& rhs, ir::OperandType type);
  ir::InputOperand&  create(ir::InputOperand& lhs, OpSrc const& rhs, ir::OperandType type);

  public:
  IRBuilder(ir::InstructionManager& manager, bool isVirtual = false): _ir(manager), _isVirtual(isVirtual) {}

  IRResult constantOp(OpDst dst, ir::ConstantValue, ir::OperandType type);

  inline IRResult literalOp(uint32_t value) {
    return constantOp(OpDst(eOperandKind::Literal()), ir::ConstantValue {.value_u64 = value}, ir::OperandType::i32());
  }

  inline IRResult constantIOp(uint32_t value) {
    return constantOp(OpDst(eOperandKind::Unset()), ir::ConstantValue {.value_u64 = value}, ir::OperandType::i32());
  }

  inline IRResult constantIOp(uint64_t value) {
    return constantOp(OpDst(eOperandKind::Unset()), ir::ConstantValue {.value_u64 = value}, ir::OperandType::i64());
  }

  inline IRResult constantSOp(int64_t value) {
    return constantOp(OpDst(eOperandKind::Unset()), ir::ConstantValue {.value_i64 = value}, ir::OperandType::i32());
  }

  inline IRResult constantFOp(float value) { return constantOp(OpDst(eOperandKind::Unset()), ir::ConstantValue {.value_f64 = value}, ir::OperandType::f32()); }

  IRResult moveOp(OpDst dst, OpSrc src, ir::OperandType type);
  IRResult selectOp(OpDst dst, OpSrc predicate, OpSrc srcTrue, OpSrc srcFalse, ir::OperandType type);
  IRResult bitReverseOp(OpDst dst, OpSrc src, ir::OperandType type);
  IRResult bitCountOp(OpDst dst, OpSrc src, ir::OperandType type);
  IRResult bitFieldMaskOp(OpDst dst, OpSrc size, OpSrc offset, ir::OperandType type);
  IRResult bitAndOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  IRResult bitOrOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  IRResult bitXorOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  IRResult findILsbOp(OpDst dst, OpSrc src, ir::OperandType type);
  IRResult findUMsbOp(OpDst dst, OpSrc src, ir::OperandType type);
  IRResult findSMsbOp(OpDst dst, OpSrc src, ir::OperandType type);
  IRResult signExtendI32Op(OpDst dst, OpSrc src, ir::OperandType type);
  IRResult bitsetOp(OpDst dst, OpSrc src, OpSrc offset, OpSrc value, ir::OperandType type);
  IRResult bitFieldInsertOp(OpDst dst, OpSrc value, OpSrc offset, OpSrc count, ir::OperandType type);
  IRResult bitUIExtractOp(OpDst dst, OpSrc base, OpSrc offset, OpSrc count, ir::OperandType type);
  IRResult bitSIExtractOp(OpDst dst, OpSrc base, OpSrc offset, OpSrc count, ir::OperandType type);
  IRResult bitUIExtractOp(OpDst dst, OpSrc base, OpSrc compact, ir::OperandType type);
  IRResult bitSIExtractOp(OpDst dst, OpSrc base, OpSrc compact, ir::OperandType type);
  IRResult bitCmpOp(OpDst dst, OpSrc base, ir::OperandType type, OpSrc index);
  IRResult cmpIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type, CmpIPredicate op);
  IRResult cmpFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type, CmpFPredicate op);

  IRResult shiftLUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  IRResult shiftRUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  IRResult shiftRSIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);

  // // arith
  IRResult ldexpOp(OpDst dst, OpSrc vsrc, OpSrc vexp, ir::OperandType type);
  IRResult addFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  IRResult subFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  IRResult mulFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  IRResult fmaFOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  IRResult fmaIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  IRResult mulIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  IRResult mulIExtendedOp(OpDst low, OpDst high, OpSrc src0, OpSrc src1, ir::OperandType type);
  IRResult addIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  IRResult addcIOp(OpDst dst, OpDst carryOut, OpSrc src0, OpSrc src1, OpSrc carryIn, ir::OperandType type);
  IRResult subIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  IRResult subbIOp(OpDst dst, OpDst carryOut, OpSrc src0, OpSrc src1, OpSrc carryIn, ir::OperandType type);

  IRResult convFPToSIOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType);
  IRResult convSIToFPOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType);
  IRResult convFPToUIOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType);
  IRResult convUIToFPOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType);
  IRResult convSI4ToFloat(OpDst dst, OpSrc src0);

  IRResult truncFOp(OpDst dst, OpSrc src0);
  IRResult extFOp(OpDst dst, OpSrc src0);
  IRResult packHalf2x16Op(OpDst dst, OpSrc src0, OpSrc src1);
  IRResult unpackHalf2x16(OpDst low, OpDst high, OpSrc src);
  IRResult packSnorm2x16Op(OpDst dst, OpSrc src0, OpSrc src1);
  IRResult packUnorm2x16Op(OpDst dst, OpSrc src0, OpSrc src1);

  IRResult truncOp(OpDst dst, OpSrc src0, ir::OperandType type);
  IRResult ceilOp(OpDst dst, OpSrc src0, ir::OperandType type);
  IRResult roundEvenOp(OpDst dst, OpSrc src0, ir::OperandType type);
  IRResult fractOp(OpDst dst, OpSrc src0, ir::OperandType type);
  IRResult floorOp(OpDst dst, OpSrc src0, ir::OperandType type);
  IRResult rcpOp(OpDst dst, OpSrc src0, ir::OperandType type);
  IRResult rsqrtOp(OpDst dst, OpSrc src0, ir::OperandType type);
  IRResult sqrtOp(OpDst dst, OpSrc src0, ir::OperandType type);
  IRResult exp2Op(OpDst dst, OpSrc src0);
  IRResult log2Op(OpDst dst, OpSrc src0);
  IRResult sinOp(OpDst dst, OpSrc src0);
  IRResult cosOp(OpDst dst, OpSrc src0);
  IRResult clampFMinMaxOp(OpDst dst, OpSrc src0, ir::OperandType type); ///< Clamp +-inf to +- flat max
  IRResult clampFZeroOp(OpDst dst, OpSrc src0, ir::OperandType type);   ///< Clamp +-inf to Zero
  IRResult frexpOp(OpDst exp, OpDst mant, OpSrc src0, ir::OperandType type);

  IRResult medUIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  IRResult medSIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  IRResult medFOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  IRResult max3UIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  IRResult maxUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  IRResult max3SIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  IRResult maxSIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  IRResult max3FOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  IRResult maxFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  IRResult maxNOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  IRResult min3UIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  IRResult minUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  IRResult min3SIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  IRResult minSIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  IRResult min3FOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
  IRResult minFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
  IRResult minNOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);

  // // Flow control
  InstructionId_t returnOp();
  InstructionId_t discardOp(OpSrc predicate);
  InstructionId_t barrierOp();
  IRResult        jumpAbsOp(OpSrc addr);
  IRResult        cjumpAbsOp(OpSrc predicate, bool invert, OpSrc addr);
};
} // namespace create
} // namespace compiler::frontend::translate