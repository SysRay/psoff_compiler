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

struct SubUIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
  static void create(Parser* parser, OpDst dst, OpDst carryOut, OpSrc src0, OpSrc src1, OpSrc carryIn, OperandType_t type);
};

struct SubSIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct MulSIOp {
  static void create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct CmpIOp {
  static void create(Parser* parser, eCmpIPredicate predOp, OpDst dst, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct IsBitSetOp {
  static void create(Parser* parser, OpDst dst, OpSrc src, OpSrc index, OperandType_t type);
};

struct IsBitClearOp {
  static void create(Parser* parser, OpDst dst, OpSrc src, OpSrc index, OperandType_t type);
};

struct MinUIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct MaxUIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct MinSIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct MaxSIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
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
};

struct BitfieldInsertOp {
  static void create(Parser* parser, OpDst dst, OpSrc src, OpSrc value, OpSrc width, OpSrc offset, OperandType_t type);
};

struct BitfieldExtractSIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src, OpSrc packed, OperandType_t type);
};

struct AbsDiffIOp {
  static void create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type);
};

struct BranchOp {
  static void create(Parser* parser, OpSrc src);
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