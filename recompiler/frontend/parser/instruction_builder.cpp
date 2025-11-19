#include "instruction_builder.h"

#include "ir/ir.h"

namespace compiler::frontend::translate::create {

static ir::Operand& create(ir::Operand& lhs, OpDst const& rhs, ir::OperandType type) {
  lhs.kind  = getOperandKind(rhs.kind);
  lhs.flags = rhs.flags;
  lhs.type  = type;
  return lhs;
}

static ir::Operand& create(ir::Operand& lhs, OpSrc const& rhs, ir::OperandType type) {
  lhs.kind       = getOperandKind(rhs.kind);
  lhs.flags      = rhs.flags;
  lhs.type       = type;
  lhs.constantId = rhs.constantId;
  return lhs;
}

InstructionId_t IRBuilder::constantOp(OpDst dst, ir::ConstantValue value, ir::OperandType type) {
  auto  id    = _ir.createInstruction(ir::getInfo(ir::eInstKind::ConstantOp));
  auto& dstOp = _ir.getDst(id, 0);

  create(_ir.getDst(id, 0), dst, type).constantId = _ir.createConstant(value);
  return id;
}

InstructionId_t IRBuilder::moveOp(OpDst dst, OpSrc src, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::MoveOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src, type);
  return id;
}

InstructionId_t IRBuilder::selectOp(OpDst dst, OpSrc predicate, OpSrc srcTrue, OpSrc srcFalse, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::SelectOp));
  create(_ir.getDst(id, 0), dst, type);

  create(_ir.getSrc(id, 0), predicate, ir::OperandType::i1());
  create(_ir.getSrc(id, 1), srcTrue, type);
  create(_ir.getSrc(id, 2), srcFalse, type);
  return id;
}

InstructionId_t IRBuilder::bitReverseOp(OpDst dst, OpSrc src, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::BitReverseOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src, type);
  return id;
}

InstructionId_t IRBuilder::bitCountOp(OpDst dst, OpSrc src, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::BitCountOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src, type);
  return id;
}

InstructionId_t IRBuilder::findILsbOp(OpDst dst, OpSrc src, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::FindILsbOp));
  create(_ir.getDst(id, 0), dst, ir::OperandType::i32());
  create(_ir.getSrc(id, 0), src, type);
  return id;
}

InstructionId_t IRBuilder::findUMsbOp(OpDst dst, OpSrc src, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::FindUMsbOp));
  create(_ir.getDst(id, 0), dst, ir::OperandType::i32());
  create(_ir.getSrc(id, 0), src, type);
  return id;
}

InstructionId_t IRBuilder::findSMsbOp(OpDst dst, OpSrc src, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::FindSMsbOp));
  create(_ir.getDst(id, 0), dst, ir::OperandType::i32());
  create(_ir.getSrc(id, 0), src, type);
  return id;
}

InstructionId_t IRBuilder::signExtendI32Op(OpDst dst, OpSrc src, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::SignExtendOp));
  create(_ir.getDst(id, 0), dst, ir::OperandType::i32());
  create(_ir.getSrc(id, 0), src, type);
  return id;
}

InstructionId_t IRBuilder::bitsetOp(OpDst dst, OpSrc src, OpSrc offset, OpSrc value, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::BitsetOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src, type);
  create(_ir.getSrc(id, 1), offset, ir::OperandType::i32());
  create(_ir.getSrc(id, 2), value, type);
  return id;
}

InstructionId_t IRBuilder::bitFieldInsertOp(OpDst dst, OpSrc value, OpSrc offset, OpSrc count, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::BitFieldInsertOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), value, type);
  create(_ir.getSrc(id, 1), offset, type);
  create(_ir.getSrc(id, 2), count, type);
  return id;
}

InstructionId_t IRBuilder::bitUIExtractOp(OpDst dst, OpSrc base, OpSrc offset, OpSrc count, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::BitUIExtractOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), base, type);
  create(_ir.getSrc(id, 1), offset, type);
  create(_ir.getSrc(id, 2), count, type);
  return id;
}

InstructionId_t IRBuilder::bitSIExtractOp(OpDst dst, OpSrc base, OpSrc offset, OpSrc count, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::BitSIExtractOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), base, type);
  create(_ir.getSrc(id, 1), offset, type);
  create(_ir.getSrc(id, 2), count, type);
  return id;
}

InstructionId_t IRBuilder::bitUIExtractOp(OpDst dst, OpSrc base, OpSrc compact, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::BitUIExtractCompactOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), base, type);
  create(_ir.getSrc(id, 1), compact, type);
  return id;
}

InstructionId_t IRBuilder::bitSIExtractOp(OpDst dst, OpSrc base, OpSrc compact, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::BitSIExtractCompactOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), base, type);
  create(_ir.getSrc(id, 1), compact, type);
  return id;
}

InstructionId_t IRBuilder::bitCmpOp(OpDst dst, OpSrc base, ir::OperandType type, OpSrc index) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::BitCmpOp));
  create(_ir.getDst(id, 0), dst, ir::OperandType::i1());
  create(_ir.getSrc(id, 0), base, type);
  create(_ir.getSrc(id, 1), index, ir::OperandType::i32());
  return id;
}

InstructionId_t IRBuilder::returnOp() {
  return _ir.createInstruction(ir::getInfo(ir::eInstKind::ReturnOp));
}

InstructionId_t IRBuilder::discardOp(OpSrc predicate) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::DiscardOp));
  create(_ir.getSrc(id, 0), predicate, ir::OperandType::i1());
  return id;
}

InstructionId_t IRBuilder::barrierOp() {
  return _ir.createInstruction(ir::getInfo(ir::eInstKind::BarrierOp));
}

InstructionId_t IRBuilder::jumpAbsOp(OpSrc addr) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::JumpAbsOp));
  create(_ir.getSrc(id, 0), addr, ir::OperandType::i64());
  return id;
}

InstructionId_t IRBuilder::cjumpAbsOp(OpSrc predicate, bool invert, OpSrc addr) {
  auto  id   = _ir.createInstruction(ir::getInfo(ir::eInstKind::CondJumpAbsOp));
  auto& pred = create(_ir.getSrc(id, 0), predicate, ir::OperandType::i1());
  create(_ir.getSrc(id, 1), addr, ir::OperandType::i64());
  if (invert) pred.flags |= OperandFlagsSrc(true, false);
  return id;
}

InstructionId_t IRBuilder::bitFieldMaskOp(OpDst dst, OpSrc size, OpSrc offset, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::BitFieldMaskOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), size, ir::OperandType::i32());
  create(_ir.getSrc(id, 1), offset, ir::OperandType::i32());
  return id;
}

InstructionId_t IRBuilder::bitAndOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::BitAndOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  return id;
}

InstructionId_t IRBuilder::bitOrOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::BitOrOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  return id;
}

InstructionId_t IRBuilder::bitXorOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::BitXorOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  return id;
}

InstructionId_t IRBuilder::cmpIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type, CmpIPredicate op) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::CmpIOp));
  create(_ir.getDst(id, 0), dst, ir::OperandType::i1());
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  _ir.accessInstr(id).userData = (std::underlying_type<CmpIPredicate>::type)op;
  return id;
}

InstructionId_t IRBuilder::cmpFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type, CmpFPredicate op) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::CmpFOp));
  create(_ir.getDst(id, 0), dst, ir::OperandType::i1());
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  _ir.accessInstr(id).userData = (std::underlying_type<CmpFPredicate>::type)op;
  return id;
}

InstructionId_t IRBuilder::mulIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::MulIOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  return id;
}

InstructionId_t IRBuilder::mulIExtendedOp(OpDst low, OpDst high, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::MulIExtendedOp));
  create(_ir.getDst(id, 0), low, type);
  create(_ir.getDst(id, 1), high, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  return id;
}

InstructionId_t IRBuilder::max3UIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::Max3UIOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  create(_ir.getSrc(id, 2), src2, type);
  return id;
}

InstructionId_t IRBuilder::maxUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::MaxUIOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  return id;
}

InstructionId_t IRBuilder::max3SIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::Max3SIOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  create(_ir.getSrc(id, 2), src2, type);
  return id;
}

InstructionId_t IRBuilder::maxSIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::MaxSIOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  return id;
}

InstructionId_t IRBuilder::max3FOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::Max3FOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  create(_ir.getSrc(id, 2), src2, type);
  return id;
}

InstructionId_t IRBuilder::maxFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::MaxFOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  return id;
}

InstructionId_t IRBuilder::maxNOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::MaxNOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  return id;
}

InstructionId_t IRBuilder::min3UIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::Min3UIOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  create(_ir.getSrc(id, 2), src2, type);
  return id;
}

InstructionId_t IRBuilder::minUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::MinUIOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  return id;
}

InstructionId_t IRBuilder::min3SIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::Min3SIOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  create(_ir.getSrc(id, 2), src2, type);
  return id;
}

InstructionId_t IRBuilder::minSIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::MinSIOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  return id;
}

InstructionId_t IRBuilder::min3FOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::Min3FOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  create(_ir.getSrc(id, 2), src2, type);
  return id;
}

InstructionId_t IRBuilder::minFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::MinFOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  return id;
}

InstructionId_t IRBuilder::minNOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::MinNOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  return id;
}

InstructionId_t IRBuilder::medUIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::MedUIOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  create(_ir.getSrc(id, 2), src2, type);
  return id;
}

InstructionId_t IRBuilder::medSIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::MedSIOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  create(_ir.getSrc(id, 2), src2, type);
  return id;
}

InstructionId_t IRBuilder::medFOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::MedFOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  create(_ir.getSrc(id, 2), src2, type);
  return id;
}

InstructionId_t IRBuilder::addIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::AddIOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  return id;
}

InstructionId_t IRBuilder::ldexpOp(OpDst dst, OpSrc vsrc, OpSrc vexp, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::LdexpOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), vsrc, type);
  create(_ir.getSrc(id, 1), vexp, ir::OperandType::i32());
  return id;
}

InstructionId_t IRBuilder::addFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::AddFOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  return id;
}

InstructionId_t IRBuilder::subFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::SubFOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  return id;
}

InstructionId_t IRBuilder::mulFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::MulFOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  return id;
}

InstructionId_t IRBuilder::fmaFOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::FmaFOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  create(_ir.getSrc(id, 2), src2, type);
  return id;
}

InstructionId_t IRBuilder::fmaIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::FmaIOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  create(_ir.getSrc(id, 2), src2, type);
  return id;
}

InstructionId_t IRBuilder::addcIOp(OpDst dst, OpDst carryOut, OpSrc src0, OpSrc src1, OpSrc carryIn, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::AddCarryIOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getDst(id, 1), carryOut, ir::OperandType::i1());
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  create(_ir.getSrc(id, 2), carryIn, ir::OperandType::i1());
  return id;
}

InstructionId_t IRBuilder::subIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::SubIOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  return id;
}

InstructionId_t IRBuilder::subbIOp(OpDst dst, OpDst carryOut, OpSrc src0, OpSrc src1, OpSrc carryIn, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::SubBurrowIOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getDst(id, 1), carryOut, ir::OperandType::i1());
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, type);
  create(_ir.getSrc(id, 2), carryIn, ir::OperandType::i1());
  return id;
}

InstructionId_t IRBuilder::shiftLUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::ShiftLUIOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, ir::OperandType::i32());
  return id;
}

InstructionId_t IRBuilder::shiftRUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::ShiftRUIOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, ir::OperandType::i32());
  return id;
}

InstructionId_t IRBuilder::shiftRSIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::ShiftRSIOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  create(_ir.getSrc(id, 1), src1, ir::OperandType::i32());
  return id;
}

InstructionId_t IRBuilder::truncFOp(OpDst dst, OpSrc src0) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::TruncFOp));
  create(_ir.getDst(id, 0), dst, ir::OperandType::f32());
  create(_ir.getSrc(id, 0), src0, ir::OperandType::f32());
  return id;
}

InstructionId_t IRBuilder::extFOp(OpDst dst, OpSrc src0) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::ExtFOp));
  create(_ir.getDst(id, 0), dst, ir::OperandType::f64());
  create(_ir.getSrc(id, 0), src0, ir::OperandType::f32());
  return id;
}

InstructionId_t IRBuilder::convFPToSIOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::FPToSIOp));
  create(_ir.getDst(id, 0), dst, dstType);
  create(_ir.getSrc(id, 0), src0, srcType);
  return id;
}

InstructionId_t IRBuilder::convSIToFPOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::SIToFPOp));
  create(_ir.getDst(id, 0), dst, dstType);
  create(_ir.getSrc(id, 0), src0, srcType);
  return id;
}

InstructionId_t IRBuilder::convFPToUIOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::FPToUIOp));
  create(_ir.getDst(id, 0), dst, dstType);
  create(_ir.getSrc(id, 0), src0, srcType);
  return id;
}

InstructionId_t IRBuilder::convUIToFPOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::UIToFPOp));
  create(_ir.getDst(id, 0), dst, dstType);
  create(_ir.getSrc(id, 0), src0, srcType);
  return id;
}

InstructionId_t IRBuilder::convSI4ToFloat(OpDst dst, OpSrc src0) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::SI4ToFloat));
  create(_ir.getDst(id, 0), dst, ir::OperandType::f32());
  create(_ir.getSrc(id, 0), src0, ir::OperandType::i32());
  return id;
}

InstructionId_t IRBuilder::packHalf2x16Op(OpDst dst, OpSrc src0, OpSrc src1) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::PackHalf2x16Op));
  create(_ir.getDst(id, 0), dst, ir::OperandType::i32());
  create(_ir.getSrc(id, 0), src0, ir::OperandType::f32());
  create(_ir.getSrc(id, 1), src1, ir::OperandType::f32());
  return id;
}

InstructionId_t IRBuilder::unpackHalf2x16(OpDst low, OpDst high, OpSrc src) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::UnpackHalf2x16));
  create(_ir.getDst(id, 0), low, ir::OperandType::f32());
  create(_ir.getDst(id, 1), high, ir::OperandType::f32());
  create(_ir.getSrc(id, 0), src, ir::OperandType::i32());
  return id;
}

InstructionId_t IRBuilder::packSnorm2x16Op(OpDst dst, OpSrc src0, OpSrc src1) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::PackSnorm2x16Op));
  create(_ir.getDst(id, 0), dst, ir::OperandType::i32());
  create(_ir.getSrc(id, 0), src0, ir::OperandType::f32());
  create(_ir.getSrc(id, 1), src1, ir::OperandType::f32());
  return id;
}

InstructionId_t IRBuilder::packUnorm2x16Op(OpDst dst, OpSrc src0, OpSrc src1) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::PackUnorm2x16Op));
  create(_ir.getDst(id, 0), dst, ir::OperandType::i32());
  create(_ir.getSrc(id, 0), src0, ir::OperandType::f32());
  create(_ir.getSrc(id, 1), src1, ir::OperandType::f32());
  return id;
}

InstructionId_t IRBuilder::truncOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::TruncOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  return id;
}

InstructionId_t IRBuilder::ceilOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::CeilOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  return id;
}

InstructionId_t IRBuilder::roundEvenOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::RoundEvenOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  return id;
}

InstructionId_t IRBuilder::fractOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::FractOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  return id;
}

InstructionId_t IRBuilder::floorOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::FloorOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  return id;
}

InstructionId_t IRBuilder::rcpOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::RcpOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  return id;
}

InstructionId_t IRBuilder::rsqrtOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::RsqrtOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  return id;
}

InstructionId_t IRBuilder::sqrtOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::SqrtOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  return id;
}

InstructionId_t IRBuilder::exp2Op(OpDst dst, OpSrc src0) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::Exp2Op));
  create(_ir.getDst(id, 0), dst, ir::OperandType::f32());
  create(_ir.getSrc(id, 0), src0, ir::OperandType::f32());
  return id;
}

InstructionId_t IRBuilder::log2Op(OpDst dst, OpSrc src0) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::Log2Op));
  create(_ir.getDst(id, 0), dst, ir::OperandType::f32());
  create(_ir.getSrc(id, 0), src0, ir::OperandType::f32());
  return id;
}

InstructionId_t IRBuilder::sinOp(OpDst dst, OpSrc src0) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::SinOp));
  create(_ir.getDst(id, 0), dst, ir::OperandType::f32());
  create(_ir.getSrc(id, 0), src0, ir::OperandType::f32());
  return id;
}

InstructionId_t IRBuilder::cosOp(OpDst dst, OpSrc src0) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::CosOp));
  create(_ir.getDst(id, 0), dst, ir::OperandType::f32());
  create(_ir.getSrc(id, 0), src0, ir::OperandType::f32());
  return id;
}

InstructionId_t IRBuilder::clampFMinMaxOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::ClampFMinMaxOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  return id;
}

InstructionId_t IRBuilder::clampFZeroOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::ClampFZeroOp));
  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src0, type);
  return id;
}

InstructionId_t IRBuilder::frexpOp(OpDst exp, OpDst mant, OpSrc src0, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::FrexpOp));
  create(_ir.getDst(id, 0), exp, ir::OperandType::i32());
  create(_ir.getDst(id, 1), mant, type);
  create(_ir.getSrc(id, 0), src0, type);
  return id;
}
} // namespace compiler::frontend::translate::create