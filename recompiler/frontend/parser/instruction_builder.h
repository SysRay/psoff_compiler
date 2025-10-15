#pragma once

#include "../ir_types.h"
#include "ir/instructions.h"
#include "ir/instructions_userdata.h"

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
ir::InstCore literalOp(uint32_t);
ir::InstCore constantOp(OpDst dst, uint64_t, ir::OperandType type);
ir::InstCore constantOp(OpDst dst, int16_t, ir::OperandType type);
ir::InstCore moveOp(OpDst dst, OpSrc src, ir::OperandType type);
ir::InstCore selectOp(OpDst dst, OpSrc predicate, OpSrc srcTrue, OpSrc srcFalse, ir::OperandType type);
ir::InstCore bitReverseOp(OpDst dst, OpSrc src, ir::OperandType type);
ir::InstCore bitCountOp(OpDst dst, OpSrc src, ir::OperandType type);
ir::InstCore bitFieldMaskOp(OpDst dst, OpSrc size, OpSrc offset, ir::OperandType type);
ir::InstCore bitAndOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
ir::InstCore bitOrOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
ir::InstCore bitXorOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
ir::InstCore findILsbOp(OpDst dst, OpSrc src, ir::OperandType type);
ir::InstCore findUMsbOp(OpDst dst, OpSrc src, ir::OperandType type);
ir::InstCore findSMsbOp(OpDst dst, OpSrc src, ir::OperandType type);
ir::InstCore signExtendI32Op(OpDst dst, OpSrc src, ir::OperandType type);
ir::InstCore bitsetOp(OpDst dst, OpSrc src, OpSrc offset, OpSrc value, ir::OperandType type);
ir::InstCore bitFieldInsertOp(OpDst dst, OpSrc value, OpSrc offset, OpSrc count, ir::OperandType type);
ir::InstCore bitUIExtractOp(OpDst dst, OpSrc base, OpSrc offset, OpSrc count, ir::OperandType type);
ir::InstCore bitSIExtractOp(OpDst dst, OpSrc base, OpSrc offset, OpSrc count, ir::OperandType type);
ir::InstCore bitUIExtractOp(OpDst dst, OpSrc base, OpSrc compact, ir::OperandType type);
ir::InstCore bitSIExtractOp(OpDst dst, OpSrc base, OpSrc compact, ir::OperandType type);
ir::InstCore bitCmpOp(OpDst dst, OpSrc base, ir::OperandType type, OpSrc index);
ir::InstCore cmpIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type, CmpIPredicate op);
ir::InstCore cmpFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type, CmpFPredicate op);

ir::InstCore shiftLUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
ir::InstCore shiftRUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
ir::InstCore shiftRSIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);

// // arith
ir::InstCore ldexpOp(OpDst dst, OpSrc vsrc, OpSrc vexp, ir::OperandType type);
ir::InstCore addFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
ir::InstCore subFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
ir::InstCore mulFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
ir::InstCore fmaFOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
ir::InstCore fmaIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
ir::InstCore mulIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
ir::InstCore mulIExtendedOp(OpDst low, OpDst high, OpSrc src0, OpSrc src1, ir::OperandType type);
ir::InstCore addIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
ir::InstCore addcIOp(OpDst dst, OpDst carryOut, OpSrc src0, OpSrc src1, OpSrc carryIn, ir::OperandType type);
ir::InstCore subIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
ir::InstCore subbIOp(OpDst dst, OpDst carryOut, OpSrc src0, OpSrc src1, OpSrc carryIn, ir::OperandType type);

ir::InstCore convFPToSIOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType);
ir::InstCore convSIToFPOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType);
ir::InstCore convFPToUIOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType);
ir::InstCore convUIToFPOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType);
ir::InstCore convSI4ToFloat(OpDst dst, OpSrc src0);

ir::InstCore truncFOp(OpDst dst, OpSrc src0);
ir::InstCore extFOp(OpDst dst, OpSrc src0);
ir::InstCore packHalf2x16Op(OpDst dst, OpSrc src0, OpSrc src1);
ir::InstCore unpackHalf2x16(OpDst low, OpDst high, OpSrc src);
ir::InstCore packSnorm2x16Op(OpDst dst, OpSrc src0, OpSrc src1);
ir::InstCore packUnorm2x16Op(OpDst dst, OpSrc src0, OpSrc src1);

ir::InstCore truncOp(OpDst dst, OpSrc src0, ir::OperandType type);
ir::InstCore ceilOp(OpDst dst, OpSrc src0, ir::OperandType type);
ir::InstCore roundEvenOp(OpDst dst, OpSrc src0, ir::OperandType type);
ir::InstCore fractOp(OpDst dst, OpSrc src0, ir::OperandType type);
ir::InstCore floorOp(OpDst dst, OpSrc src0, ir::OperandType type);
ir::InstCore rcpOp(OpDst dst, OpSrc src0, ir::OperandType type);
ir::InstCore rsqrtOp(OpDst dst, OpSrc src0, ir::OperandType type);
ir::InstCore sqrtOp(OpDst dst, OpSrc src0, ir::OperandType type);
ir::InstCore exp2Op(OpDst dst, OpSrc src0);
ir::InstCore log2Op(OpDst dst, OpSrc src0);
ir::InstCore sinOp(OpDst dst, OpSrc src0);
ir::InstCore cosOp(OpDst dst, OpSrc src0);
ir::InstCore clampFMinMaxOp(OpDst dst, OpSrc src0, ir::OperandType type); ///< Clamp +-inf to +- flat max
ir::InstCore clampFZeroOp(OpDst dst, OpSrc src0, ir::OperandType type);   ///< Clamp +-inf to Zero
ir::InstCore frexpOp(OpDst exp, OpDst mant, OpSrc src0, ir::OperandType type);

ir::InstCore medUIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
ir::InstCore medSIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
ir::InstCore medFOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
ir::InstCore max3UIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
ir::InstCore maxUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
ir::InstCore max3SIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
ir::InstCore maxSIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
ir::InstCore max3FOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
ir::InstCore maxFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
ir::InstCore maxNOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
ir::InstCore min3UIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
ir::InstCore minUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
ir::InstCore min3SIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
ir::InstCore minSIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
ir::InstCore min3FOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
ir::InstCore minFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
ir::InstCore minNOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);

// // Flow control
ir::InstCore returnOp();
ir::InstCore discardOp(OpSrc predicate);
ir::InstCore barrierOp();
ir::InstCore jumpAbsOp(OpSrc addr);
ir::InstCore cjumpAbsOp(OpSrc predicate, bool invert, OpSrc addr);
} // namespace create
} // namespace compiler::frontend::translate