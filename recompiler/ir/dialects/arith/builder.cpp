#include "builder.h"

namespace compiler::ir::dialect::arith {

static inline OutputOperand& createOp(IROperations& ir, OutputOperandId_t id, OpDst const& rhs, OperandType type) {
  auto& op = ir.getOperand(id);

  op.kind  = rhs.kind;
  op.flags = rhs.flags;
  op.type  = type;
  return op;
}

static inline InputOperand& createOp(IROperations& ir, InputOperandId_t id, OpSrc const& rhs, OperandType type) {
  auto& op = ir.getOperand(id);

  op.kind  = rhs.kind;
  op.flags = rhs.flags;
  op.type  = type;
  if (op.ssaId.isValid()) ir.connect(id, rhs.ssa);
  return op;
}

IRResult BitReverseOp::create(IROperations& ir, OpDst dst, OpSrc src, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::BitReverseOp));
  createOp(ir, ir.getSrc(id, 0), src, type);
  createOp(ir, ir.getDst(id, 0), dst, type);
  return IRResult(ir, id);
}

IRResult BitCountOp::create(IROperations& ir, OpDst dst, OpSrc src, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::BitCountOp));
  createOp(ir, ir.getSrc(id, 0), src, type);
  createOp(ir, ir.getDst(id, 0), dst, type);
  return IRResult(ir, id);
}

IRResult BitFieldMaskOp::create(IROperations& ir, OpDst dst, OpSrc size, OpSrc offset, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::BitFieldMaskOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), size, OperandType::i32());
  createOp(ir, ir.getSrc(id, 1), offset, OperandType::i32());
  return IRResult(ir, id);
}

IRResult BitAndOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::BitAndOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  return IRResult(ir, id);
}

IRResult BitOrOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::BitOrOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  return IRResult(ir, id);
}

IRResult BitXorOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::BitXorOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  return IRResult(ir, id);
}

IRResult FindILsbOp::create(IROperations& ir, OpDst dst, OpSrc src, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::FindILsbOp));
  createOp(ir, ir.getSrc(id, 0), src, type);
  createOp(ir, ir.getDst(id, 0), dst, OperandType::i32());
  return IRResult(ir, id);
}

IRResult FindUMsbOp::create(IROperations& ir, OpDst dst, OpSrc src, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::FindUMsbOp));
  createOp(ir, ir.getSrc(id, 0), src, type);
  createOp(ir, ir.getDst(id, 0), dst, OperandType::i32());
  return IRResult(ir, id);
}

IRResult FindSMsbOp::create(IROperations& ir, OpDst dst, OpSrc src, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::FindSMsbOp));
  createOp(ir, ir.getSrc(id, 0), src, type);
  createOp(ir, ir.getDst(id, 0), dst, OperandType::i32());
  return IRResult(ir, id);
}

IRResult SignExtendOp::create(IROperations& ir, OpDst dst, OpSrc src, OperandType type, OperandType dstType) {
  auto id = ir.createInstruction(getInfo(eInstKind::SignExtendOp));
  createOp(ir, ir.getSrc(id, 0), src, type);
  createOp(ir, ir.getDst(id, 0), dst, dstType);
  return IRResult(ir, id);
}

// ------------------------------------------------------------
// Bit field / extract
// ------------------------------------------------------------

IRResult BitsetOp::create(IROperations& ir, OpDst dst, OpSrc src, OpSrc offset, OpSrc value, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::BitsetOp));
  createOp(ir, ir.getSrc(id, 0), src, type);
  createOp(ir, ir.getSrc(id, 1), offset, OperandType::i32());
  createOp(ir, ir.getSrc(id, 2), value, type);
  createOp(ir, ir.getDst(id, 0), dst, OperandType::i32());
  return IRResult(ir, id);
}

IRResult BitFieldInsertOp::create(IROperations& ir, OpDst dst, OpSrc value, OpSrc offset, OpSrc count, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::BitFieldInsertOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), value, type);
  createOp(ir, ir.getSrc(id, 1), offset, type);
  createOp(ir, ir.getSrc(id, 2), count, type);
  return IRResult(ir, id);
}

IRResult BitUIExtractOp::create(IROperations& ir, OpDst dst, OpSrc base, OpSrc offset, OpSrc count, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::BitUIExtractOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), base, type);
  createOp(ir, ir.getSrc(id, 1), offset, type);
  createOp(ir, ir.getSrc(id, 2), count, type);
  return IRResult(ir, id);
}

IRResult BitSIExtractOp::create(IROperations& ir, OpDst dst, OpSrc base, OpSrc offset, OpSrc count, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::BitSIExtractOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), base, type);
  createOp(ir, ir.getSrc(id, 1), offset, type);
  createOp(ir, ir.getSrc(id, 2), count, type);
  return IRResult(ir, id);
}

IRResult BitUIExtractOp::create(IROperations& ir, OpDst dst, OpSrc base, OpSrc compact, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::BitUIExtractCompactOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), base, type);
  createOp(ir, ir.getSrc(id, 1), compact, type);
  return IRResult(ir, id);
}

IRResult BitSIExtractOp::create(IROperations& ir, OpDst dst, OpSrc base, OpSrc compact, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::BitSIExtractCompactOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), base, type);
  createOp(ir, ir.getSrc(id, 1), compact, type);
  return IRResult(ir, id);
}

// ------------------------------------------------------------
// Comparisons
// ------------------------------------------------------------

IRResult BitCmpOp::create(IROperations& ir, OpDst dst, OpSrc base, OpSrc index, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::BitCmpOp));
  createOp(ir, ir.getDst(id, 0), dst, OperandType::i1());
  createOp(ir, ir.getSrc(id, 0), base, type);
  createOp(ir, ir.getSrc(id, 1), index, OperandType::i32());
  return IRResult(ir, id);
}

IRResult CmpIOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type, CmpIPredicate pred) {
  auto id = ir.createInstruction(getInfo(eInstKind::CmpIOp));
  createOp(ir, ir.getDst(id, 0), dst, OperandType::i1());
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  ir.accessInstr(id).userData = (std::underlying_type<CmpIPredicate>::type)pred;
  return IRResult(ir, id);
}

IRResult CmpFOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type, CmpFPredicate pred) {
  auto id = ir.createInstruction(getInfo(eInstKind::CmpFOp));
  createOp(ir, ir.getDst(id, 0), dst, OperandType::i1());
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  ir.accessInstr(id).userData = (std::underlying_type<CmpFPredicate>::type)pred;
  return IRResult(ir, id);
}

// ------------------------------------------------------------
// Arithmetic (int / float / misc)
// ------------------------------------------------------------

IRResult AddIOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::AddIOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  return IRResult(ir, id);
}

IRResult SubIOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::SubIOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  return IRResult(ir, id);
}

IRResult MulIOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::MulIOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  return IRResult(ir, id);
}

IRResult MulIExtendedOp::create(IROperations& ir, OpDst low, OpDst high, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::MulIExtendedOp));
  createOp(ir, ir.getDst(id, 0), low, type);
  createOp(ir, ir.getDst(id, 1), high, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  return IRResult(ir, id);
}

IRResult Max3UIOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::Max3UIOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  createOp(ir, ir.getSrc(id, 2), src2, type);
  return IRResult(ir, id);
}

IRResult MaxUIOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::MaxUIOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  return IRResult(ir, id);
}

IRResult Max3SIOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::Max3SIOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  createOp(ir, ir.getSrc(id, 2), src2, type);
  return IRResult(ir, id);
}

IRResult MaxSIOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::MaxSIOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  return IRResult(ir, id);
}

IRResult Max3FOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::Max3FOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  createOp(ir, ir.getSrc(id, 2), src2, type);
  return IRResult(ir, id);
}

IRResult MaxFOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::MaxFOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  return IRResult(ir, id);
}

IRResult MaxNOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::MaxNOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  return IRResult(ir, id);
}

IRResult Min3UIOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::Min3UIOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  createOp(ir, ir.getSrc(id, 2), src2, type);
  return IRResult(ir, id);
}

IRResult MinUIOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::MinUIOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  return IRResult(ir, id);
}

IRResult Min3SIOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::Min3SIOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  createOp(ir, ir.getSrc(id, 2), src2, type);
  return IRResult(ir, id);
}

IRResult MinSIOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::MinSIOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  return IRResult(ir, id);
}

IRResult Min3FOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::Min3FOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  createOp(ir, ir.getSrc(id, 2), src2, type);
  return IRResult(ir, id);
}

IRResult MinFOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::MinFOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  return IRResult(ir, id);
}

IRResult MinNOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::MinNOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  return IRResult(ir, id);
}

IRResult MedUIOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::MedUIOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  createOp(ir, ir.getSrc(id, 2), src2, type);
  return IRResult(ir, id);
}

IRResult MedSIOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::MedSIOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  createOp(ir, ir.getSrc(id, 2), src2, type);
  return IRResult(ir, id);
}

IRResult MedFOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::MedFOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  createOp(ir, ir.getSrc(id, 2), src2, type);
  return IRResult(ir, id);
}

// ------------------------------------------------------------
// Integer / Float arithmetic
// ------------------------------------------------------------

IRResult AddFOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::AddFOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  return IRResult(ir, id);
}

IRResult SubFOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::SubFOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  return IRResult(ir, id);
}

IRResult MulFOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::MulFOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  return IRResult(ir, id);
}

IRResult FmaFOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::FmaFOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  createOp(ir, ir.getSrc(id, 2), src2, type);
  return IRResult(ir, id);
}

IRResult FmaIOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::FmaIOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  createOp(ir, ir.getSrc(id, 2), src2, type);
  return IRResult(ir, id);
}

IRResult AddCarryIOp::create(IROperations& ir, OpDst dst, OpDst carryOut, OpSrc src0, OpSrc src1, OpSrc carryIn, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::AddCarryIOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getDst(id, 1), carryOut, OperandType::i1());
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  createOp(ir, ir.getSrc(id, 2), carryIn, OperandType::i1());
  return IRResult(ir, id);
}

IRResult SubBurrowIOp::create(IROperations& ir, OpDst dst, OpDst carryOut, OpSrc src0, OpSrc src1, OpSrc carryIn, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::SubBurrowIOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getDst(id, 1), carryOut, OperandType::i1());
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, type);
  createOp(ir, ir.getSrc(id, 2), carryIn, OperandType::i1());
  return IRResult(ir, id);
}

IRResult TruncFOp::create(IROperations& ir, OpDst dst, OpSrc src0) {
  auto id = ir.createInstruction(getInfo(eInstKind::TruncFOp));
  createOp(ir, ir.getDst(id, 0), dst, ir::OperandType::f32());
  createOp(ir, ir.getSrc(id, 0), src0, ir::OperandType::f32());
  return IRResult(ir, id);
}

IRResult ExtFOp::create(IROperations& ir, OpDst dst, OpSrc src0) {
  auto id = ir.createInstruction(getInfo(eInstKind::ExtFOp));
  createOp(ir, ir.getDst(id, 0), dst, ir::OperandType::f64());
  createOp(ir, ir.getSrc(id, 0), src0, ir::OperandType::f32());
  return IRResult(ir, id);
}

IRResult LdexpOp::create(IROperations& ir, OpDst dst, OpSrc vsrc, OpSrc vexp, ir::OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::LdexpOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), vsrc, type);
  createOp(ir, ir.getSrc(id, 1), vexp, ir::OperandType::i32());
  return IRResult(ir, id);
}

IRResult ConvFPToSIOp::create(IROperations& ir, OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType) {
  auto id = ir.createInstruction(getInfo(eInstKind::FPToSIOp));
  createOp(ir, ir.getDst(id, 0), dst, dstType);
  createOp(ir, ir.getSrc(id, 0), src0, srcType);
  return IRResult(ir, id);
}

IRResult ConvSIToFPOp::create(IROperations& ir, OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType) {
  auto id = ir.createInstruction(getInfo(eInstKind::SIToFPOp));
  createOp(ir, ir.getDst(id, 0), dst, dstType);
  createOp(ir, ir.getSrc(id, 0), src0, srcType);
  return IRResult(ir, id);
}

IRResult ConvFPToUIOp::create(IROperations& ir, OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType) {
  auto id = ir.createInstruction(getInfo(eInstKind::FPToUIOp));
  createOp(ir, ir.getDst(id, 0), dst, dstType);
  createOp(ir, ir.getSrc(id, 0), src0, srcType);
  return IRResult(ir, id);
}

IRResult ConvUIToFPOp::create(IROperations& ir, OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType) {
  auto id = ir.createInstruction(getInfo(eInstKind::UIToFPOp));
  createOp(ir, ir.getDst(id, 0), dst, dstType);
  createOp(ir, ir.getSrc(id, 0), src0, srcType);
  return IRResult(ir, id);
}

IRResult ConvSI4ToFloat::create(IROperations& ir, OpDst dst, OpSrc src0) {
  auto id = ir.createInstruction(getInfo(eInstKind::SI4ToFloat));
  createOp(ir, ir.getDst(id, 0), dst, ir::OperandType::f32());
  createOp(ir, ir.getSrc(id, 0), src0, ir::OperandType::i32());
  return IRResult(ir, id);
}

IRResult PackHalf2x16Op::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1) {
  auto id = ir.createInstruction(getInfo(eInstKind::PackHalf2x16Op));
  createOp(ir, ir.getDst(id, 0), dst, ir::OperandType::i32());
  createOp(ir, ir.getSrc(id, 0), src0, ir::OperandType::f32());
  createOp(ir, ir.getSrc(id, 1), src1, ir::OperandType::f32());
  return IRResult(ir, id);
}

IRResult UnpackHalf2x16::create(IROperations& ir, OpDst low, OpDst high, OpSrc src) {
  auto id = ir.createInstruction(getInfo(eInstKind::UnpackHalf2x16));
  createOp(ir, ir.getDst(id, 0), low, ir::OperandType::f32());
  createOp(ir, ir.getDst(id, 1), high, ir::OperandType::f32());
  createOp(ir, ir.getSrc(id, 0), src, ir::OperandType::i32());
  return IRResult(ir, id);
}

IRResult PackSnorm2x16Op::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1) {
  auto id = ir.createInstruction(getInfo(eInstKind::PackSnorm2x16Op));
  createOp(ir, ir.getDst(id, 0), dst, ir::OperandType::i32());
  createOp(ir, ir.getSrc(id, 0), src0, ir::OperandType::f32());
  createOp(ir, ir.getSrc(id, 1), src1, ir::OperandType::f32());
  return IRResult(ir, id);
}

IRResult PackUnorm2x16Op::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1) {
  auto id = ir.createInstruction(getInfo(eInstKind::PackUnorm2x16Op));
  createOp(ir, ir.getDst(id, 0), dst, ir::OperandType::i32());
  createOp(ir, ir.getSrc(id, 0), src0, ir::OperandType::f32());
  createOp(ir, ir.getSrc(id, 1), src1, ir::OperandType::f32());
  return IRResult(ir, id);
}

// ------------------------------------------------------------
// Shifts
// ------------------------------------------------------------

IRResult ShiftLUIOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::ShiftLUIOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, OperandType::i32());
  return IRResult(ir, id);
}

IRResult ShiftRUIOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::ShiftRUIOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, OperandType::i32());
  return IRResult(ir, id);
}

IRResult ShiftRSIOp::create(IROperations& ir, OpDst dst, OpSrc src0, OpSrc src1, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::ShiftRSIOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  createOp(ir, ir.getSrc(id, 1), src1, OperandType::i32());
  return IRResult(ir, id);
}

// ------------------------------------------------------------
// Floating point unary math
// ------------------------------------------------------------

IRResult TruncOp::create(IROperations& ir, OpDst dst, OpSrc src0, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::TruncOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  return IRResult(ir, id);
}

IRResult CeilOp::create(IROperations& ir, OpDst dst, OpSrc src0, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::CeilOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  return IRResult(ir, id);
}

IRResult RoundEvenOp::create(IROperations& ir, OpDst dst, OpSrc src0, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::RoundEvenOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  return IRResult(ir, id);
}

IRResult FractOp::create(IROperations& ir, OpDst dst, OpSrc src0, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::FractOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  return IRResult(ir, id);
}

IRResult FloorOp::create(IROperations& ir, OpDst dst, OpSrc src0, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::FloorOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  return IRResult(ir, id);
}

IRResult RcpOp::create(IROperations& ir, OpDst dst, OpSrc src0, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::RcpOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  return IRResult(ir, id);
}

IRResult RsqrtOp::create(IROperations& ir, OpDst dst, OpSrc src0, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::RsqrtOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  return IRResult(ir, id);
}

IRResult SqrtOp::create(IROperations& ir, OpDst dst, OpSrc src0, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::SqrtOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  return IRResult(ir, id);
}

// ------------------------------------------------------------
// Trig / exp / log
// ------------------------------------------------------------

IRResult Exp2Op::create(IROperations& ir, OpDst dst, OpSrc src0) {
  auto id = ir.createInstruction(getInfo(eInstKind::Exp2Op));
  createOp(ir, ir.getDst(id, 0), dst, OperandType::f32());
  createOp(ir, ir.getSrc(id, 0), src0, OperandType::f32());
  return IRResult(ir, id);
}

IRResult Log2Op::create(IROperations& ir, OpDst dst, OpSrc src0) {
  auto id = ir.createInstruction(getInfo(eInstKind::Log2Op));
  createOp(ir, ir.getDst(id, 0), dst, OperandType::f32());
  createOp(ir, ir.getSrc(id, 0), src0, OperandType::f32());
  return IRResult(ir, id);
}

IRResult SinOp::create(IROperations& ir, OpDst dst, OpSrc src0) {
  auto id = ir.createInstruction(getInfo(eInstKind::SinOp));
  createOp(ir, ir.getDst(id, 0), dst, OperandType::f32());
  createOp(ir, ir.getSrc(id, 0), src0, OperandType::f32());
  return IRResult(ir, id);
}

IRResult CosOp::create(IROperations& ir, OpDst dst, OpSrc src0) {
  auto id = ir.createInstruction(getInfo(eInstKind::CosOp));
  createOp(ir, ir.getDst(id, 0), dst, OperandType::f32());
  createOp(ir, ir.getSrc(id, 0), src0, OperandType::f32());
  return IRResult(ir, id);
}

// ------------------------------------------------------------
// Clamp / frexp
// ------------------------------------------------------------

IRResult ClampFMinMaxOp::create(IROperations& ir, OpDst dst, OpSrc src0, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::ClampFMinMaxOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  return IRResult(ir, id);
}

IRResult ClampFZeroOp::create(IROperations& ir, OpDst dst, OpSrc src0, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::ClampFZeroOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  return IRResult(ir, id);
}

IRResult FrexpOp::create(IROperations& ir, OpDst exp, OpDst mant, OpSrc src0, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::FrexpOp));
  createOp(ir, ir.getDst(id, 0), exp, OperandType::i32());
  createOp(ir, ir.getDst(id, 1), mant, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  return IRResult(ir, id);
}

IRResult NotOp::create(IROperations& ir, OpDst dst, OpSrc src0, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::NotOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  return IRResult(ir, id);
}

IRResult NegateFOp::create(IROperations& ir, OpDst dst, OpSrc src0, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::NegateFOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  return IRResult(ir, id);
}

IRResult AbsoluteOp::create(IROperations& ir, OpDst dst, OpSrc src0, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::AbsoluteOp));
  createOp(ir, ir.getDst(id, 0), dst, type);
  createOp(ir, ir.getSrc(id, 0), src0, type);
  return IRResult(ir, id);
}
} // namespace compiler::ir::dialect::arith