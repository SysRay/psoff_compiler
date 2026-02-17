#pragma once

#include "parser.h"

namespace compiler::frontend::op {

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
  static void create(Parser* parser, OpDst dst, OpSrc offset, OperandType_t type);
};

struct BitClearOp {
  static void create(Parser* parser, OpDst dst, OpSrc offset, OperandType_t type);
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