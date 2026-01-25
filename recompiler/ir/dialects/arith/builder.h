#pragma once

#include "instructions.h"

#include <span>

namespace compiler::ir::dialect::arith {

enum class CmpIPredicate : InstructionUserData_t {
  AlwaysFalse = 0,
  eq          = 1,
  ne          = 2,
  slt         = 3,
  sle         = 4,
  sgt         = 5,
  sge         = 6,
  ult         = 7,
  ule         = 8,
  ugt         = 9,
  uge         = 10,
  AlwaysTrue  = 11,
};

enum class CmpFPredicate : InstructionUserData_t {
  AlwaysFalse = 0,
  OEQ         = 1,
  OGT         = 2,
  OGE         = 3,
  OLT         = 4,
  OLE         = 5,
  ONE         = 6,
  ORD         = 7,
  UEQ         = 8,
  UGT         = 9,
  UGE         = 10,
  ULT         = 11,
  ULE         = 12,
  UNE         = 13,
  UNO         = 14,
  AlwaysTrue  = 15,
};

// --- Operation structs ---------------------------------------------
struct BitReverseOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src, ir::OperandType type);
};

struct BitCountOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src, ir::OperandType type);
};

struct BitFieldMaskOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc size, OpSrc offset, ir::OperandType type);
};

struct BitAndOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

struct BitOrOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

struct BitXorOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

struct FindILsbOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src, ir::OperandType type);
};

struct FindUMsbOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src, ir::OperandType type);
};

struct FindSMsbOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src, ir::OperandType type);
};

struct SignExtendOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src, ir::OperandType type, OperandType dstType);
};

struct BitsetOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src, OpSrc offset, OpSrc value, ir::OperandType type);
};

struct BitFieldInsertOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc value, OpSrc offset, OpSrc count, ir::OperandType type);
};

struct BitUIExtractOp {

  static IRResult create(IROperations& ir, OpDst dst, OpSrc base, OpSrc offset, OpSrc count, ir::OperandType type);
  static IRResult create(IROperations& ir, OpDst dst, OpSrc base, OpSrc compact, ir::OperandType type);
};

struct BitSIExtractOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc base, OpSrc offset, OpSrc count, ir::OperandType type);
  static IRResult create(IROperations& ir, OpDst dst, OpSrc base, OpSrc compact, ir::OperandType type);
};

struct BitCmpOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc base, OpSrc index, ir::OperandType type);
};

struct CmpIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type, CmpIPredicate op);
};

struct CmpFOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type, CmpFPredicate op);
};

// --- Shift ops -------------------------------------------------------
struct ShiftLUIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

struct ShiftRUIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

struct ShiftRSIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

// --- Arithmetic / math ops -------------------------------------------
struct LdexpOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc vsrc, OpSrc vexp, ir::OperandType type);
};

struct AddFOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

struct SubFOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

struct MulFOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

struct FmaFOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
};

struct FmaIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
};

struct MulIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

struct MulIExtendedOp {
  static IRResult create(IROperations& ir, OpDst low, OpDst high, OpSrc src0, OpSrc src1, ir::OperandType type);
};

struct AddIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

struct AddCarryIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpDst carryOut, OpSrc src0, OpSrc src1, OpSrc carryIn, ir::OperandType type);
};

struct SubIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

struct SubBurrowIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpDst carryOut, OpSrc src0, OpSrc src1, OpSrc carryIn, ir::OperandType type);
};

// --- Conversions ----------------------------------------------------
struct ConvFPToSIOp {
  static IRResult create(IROperations& ir, OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType);
};

struct ConvSIToFPOp {
  static IRResult create(IROperations& ir, OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType);
};

struct ConvFPToUIOp {
  static IRResult create(IROperations& ir, OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType);
};

struct ConvUIToFPOp {
  static IRResult create(IROperations& ir, OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType);
};

struct ConvSI4ToFloat {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0);
};

// --- Floating‑point helpers -----------------------------------------
struct TruncFOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0);
};

struct ExtFOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0);
};

struct PackHalf2x16Op {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1);
};

struct UnpackHalf2x16 {
  static IRResult create(IROperations& ir, OpDst low, OpDst high, OpSrc src);
};

struct PackSnorm2x16Op {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1);
};

struct PackUnorm2x16Op {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1);
};

// --- Misc ops -------------------------------------------------------
struct TruncOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, ir::OperandType type);
};

struct CeilOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, ir::OperandType type);
};

struct RoundEvenOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, ir::OperandType type);
};

struct FractOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, ir::OperandType type);
};

struct FloorOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, ir::OperandType type);
};

struct RcpOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, ir::OperandType type);
};

struct RsqrtOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, ir::OperandType type);
};

struct SqrtOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, ir::OperandType type);
};

// --- Exponent / trig -------------------------------------------------
struct Exp2Op {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0);
};

struct Log2Op {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0);
};

struct SinOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0);
};

struct CosOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0);
};

// --- Clamp / frexp ---------------------------------------------------
struct ClampFMinMaxOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, ir::OperandType type);
};

struct ClampFZeroOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, ir::OperandType type);
};

struct FrexpOp {
  static IRResult create(IROperations& ir, OpDst exp, OpDst mant, OpSrc src0, ir::OperandType type);
};

// --- Median / max/min -----------------------------------------------
struct MedUIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
};

struct MedSIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
};

struct MedFOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
};

// Max
struct Max3UIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
};

struct MaxUIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

struct Max3SIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
};

struct MaxSIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

struct Max3FOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
};

struct MaxFOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

struct MaxNOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

// Min
struct Min3UIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
};

struct MinUIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

struct Min3SIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
};

struct MinSIOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

struct Min3FOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type);
};

struct MinFOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

struct MinNOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type);
};

struct NotOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, ir::OperandType type);
};

struct NegateFOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, ir::OperandType type);
};

struct AbsoluteOp {
  static IRResult create(IROperations& ir, OpDst dst, OpSrc src0, ir::OperandType type);
};
} // namespace compiler::ir::dialect::arith