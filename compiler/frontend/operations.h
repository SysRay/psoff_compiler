#pragma once

#include "parser.h"

namespace compiler::frontend::op {

enum class eCmpIPredicate {
  AlwaysFalse,
  eq,
  ne,
  slt,
  sle,
  sgt,
  sge,
  ult,
  ule,
  ugt,
  uge,
  AlwaysTrue,
};

enum class eCmpFPredicate {
  AlwaysFalse,
  OEQ,
  OGT,
  OGE,
  OLT,
  OLE,
  ONE,
  ORD,
  UEQ,
  UGT,
  UGE,
  ULT,
  ULE,
  UNE,
  UNO,
  AlwaysTrue,
};

struct MoveOp {
  static void create(Parser* parser, OpDst dst, OpSrc src, OperandType_t type);
  static void create(Parser* parser, OpDst dst, uint64_t);
};

struct CMoveOp {
  static void create(Parser* parser, OpDst dst, OpSrc predicate, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct NotOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OperandType_t type);
};

struct BrevOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type);
};

struct BitCountOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OperandType_t type);
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct FindFirstLsbBitOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type);
};

struct FindFirstUMsbBitOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type);
};

struct FindFirstSMsbBitOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type);
};

struct SignExtOp {
  static void create(Parser* parser, OpDst dst, OperandType_t dstType, OpSrc src0, OperandType_t srcType);
};

struct AbsIOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type);
};

struct BitSetOp {
  static void create(Parser* parser, OpDst dst, OpSrc index, OperandType_t type);
};

struct BitClearOp {
  static void create(Parser* parser, OpDst dst, OpSrc index, OperandType_t type);
};

struct AddUIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
  static void create(Parser* parser, OpDst dst, OpDst carryOut, OpSrc src0, OpSrc src1, OpSrc carryIn, OperandType_t type);
};

struct AddSIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct AddFOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct SubUIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
  static void create(Parser* parser, OpDst dst, OpDst carryOut, OpSrc src0, OpSrc src1, OpSrc carryIn, OperandType_t type);
};

struct SubSIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct SubFOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct MulIOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OperandType_t type);
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OperandType_t type, bool isSigned, bool retHigh);
};

struct Mul24IOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OperandType_t type, bool isSigned, bool retHigh);
};

struct MulFOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct FmaOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OpSrc add, OperandType_t type);
};

struct FmaI24Op {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OpSrc add, bool isSigned);
};

struct CmpOp {
  static mlir::Value create(Parser* parser, eCmpIPredicate predOp, OpDst dst, OpSrc src0, OpSrc src1, OperandType_t type);
  static mlir::Value create(Parser* parser, eCmpFPredicate predOp, OpDst dst, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct IsBitSetOp {
  static void create(Parser* parser, OpDst dst, OpSrc src, OpSrc index, OperandType_t type);
};

struct IsBitClearOp {
  static void create(Parser* parser, OpDst dst, OpSrc src, OpSrc index, OperandType_t type);
};

struct MinUIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType_t type);
};

struct MaxUIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType_t type);
};

struct MinSIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType_t type);
};

struct MaxSIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType_t type);
};

struct MaxFOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OperandType_t type, bool legacy = false);
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType_t type);
};

struct MinFOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OperandType_t type, bool legacy = false);
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType_t type);
};

struct MedFOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType_t type);
};

struct MedSIOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType_t type);
};

struct MedUIOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType_t type);
};

struct AndIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct OrIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct XorIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct LSHLOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct LSHROp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct ASHROp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct BitfieldMaskOp {
  static void create(Parser* parser, OpDst dst, OpSrc width, OpSrc offset, OperandType_t type);
};

struct BitfieldExtractUIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src, OpSrc packed, OperandType_t type);
  static void create(Parser* parser, OpDst dst, OpSrc src, OpSrc offset, OpSrc width, OperandType_t type);
};

struct BitfieldInsertOp {
  static void create(Parser* parser, OpDst dst, OpSrc src, OpSrc value, OpSrc width, OpSrc offset, OperandType_t type);
  static void create(Parser* parser, OpDst dst, OpSrc mask, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct BitfieldExtractSIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src, OpSrc packed, OperandType_t type);
  static void create(Parser* parser, OpDst dst, OpSrc src, OpSrc offset, OpSrc width, OperandType_t type);
};

struct AbsDiffIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct BranchOp {
  static void create(Parser* parser, OpSrc src);
};

struct ConvertFtoSIOp {
  enum class eMode {
    Round,
    RPI,
    Floor,
  };
  static void create(Parser* parser, OpDst dst, OperandType_t dstType, OpSrc src, OperandType_t srcType, eMode roundMode = eMode::Round);
};

struct ConvertFtoUIOp {
  static void create(Parser* parser, OpDst dst, OperandType_t dstType, OpSrc src, OperandType_t srcType);
};

struct ConvertUItoFOp {
  static void create(Parser* parser, OpDst dst, OperandType_t dstType, OpSrc src, OperandType_t srcType);
};

struct ConvertSItoFOp {
  static void create(Parser* parser, OpDst dst, OperandType_t dstType, OpSrc src, OperandType_t srcType);
};

struct ConvertFtoFOp {
  static void create(Parser* parser, OpDst dst, OperandType_t dstType, OpSrc src, OperandType_t srcType);
};

struct ConvertSubPixelOffsetToFOp {
  static void create(Parser* parser, OpDst dst, OperandType_t dstType, OpSrc src, OperandType_t srcType);
};

struct ConvertByteToFOp {
  static void create(Parser* parser, OpDst dst, OperandType_t dstType, OpSrc src, OperandType_t srcType, uint8_t index);
};

struct TruncOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type);
};

struct CeilOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type);
};

struct FloorOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type);
};

struct RoundEvenOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type);
};

struct FractOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type);
};

struct Exp2Op {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type);
};

struct Log2Op {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type);
};

struct RcpOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type);
};

struct RsqOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type);
};

struct SqrtOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type);
};

struct SinOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type);
};

struct CosOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type);
};

struct GetExpOp {
  static void create(Parser* parser, OpDst dst, OperandType_t dstType, OpSrc src0, OperandType_t type);
};

struct GetMantOp {
  static void create(Parser* parser, OpDst dst, OperandType_t dstType, OpSrc src0, OperandType_t type);
};

struct LDExpOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct ConvertPackSnormOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1);
};

struct ConvertPackUnormOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1);
};

struct ConvertPackF32Op {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1);
};

struct ConvertPackSI32Op {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1);
};

struct ConvertPackUI32Op {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1);
};

struct ConvertPackUI8Op {
  static void create(Parser* parser, OpDst dst, OpSrc src, OpSrc pos, OpSrc packed);
};

struct SaveExecOp {
  enum class BitOp {
    eAND,
    eOR,
    eXOR,
  };
  static void create(Parser* parser, OpDst dst, BitOp bitop, OpSrc src0, OpSrc src1, OperandType_t type);
};
} // namespace compiler::frontend::op