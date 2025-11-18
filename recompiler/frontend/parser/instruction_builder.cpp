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
  lhs.kind  = getOperandKind(rhs.kind);
  lhs.flags = rhs.flags;
  lhs.type  = type;
  return lhs;
}

InstructionId_t IR::constantOp(OpDst dst, ir::ConstantValue value, ir::OperandType type) {
  auto  id    = _ir.createInstruction(ir::getInfo(ir::eInstKind::ConstantOp));
  auto& dstOp = _ir.getDst(id, 0);

  create(_ir.getDst(id, 0), dst, type).constantId = _ir.createConstant(value);
  return id;
}

InstructionId_t IR::moveOp(OpDst dst, OpSrc src, ir::OperandType type) {
  auto id = _ir.createInstruction(ir::getInfo(ir::eInstKind::MoveOp));

  create(_ir.getDst(id, 0), dst, type);
  create(_ir.getSrc(id, 0), src, type);
  return id;
}

InstructionId_t IR::selectOp(OpDst dst, OpSrc predicate, OpSrc srcTrue, OpSrc srcFalse, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::SelectOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(predicate.kind);
  inst.srcOperands[1].kind  = getOperandKind(srcTrue.kind);
  inst.srcOperands[2].kind  = getOperandKind(srcFalse.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = predicate.flags;
  inst.srcOperands[1].flags = srcTrue.flags;
  inst.srcOperands[2].flags = srcFalse.flags;
  inst.dstOperands[1].type = inst.srcOperands[1].type = inst.srcOperands[2].type = type;
  return inst;
}

InstructionId_t IR::bitReverseOp(OpDst dst, OpSrc src, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::BitReverseOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

InstructionId_t IR::bitCountOp(OpDst dst, OpSrc src, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::BitCountOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src.flags;
  inst.srcOperands[0].type  = type;
  return inst;
}

InstructionId_t IR::findILsbOp(OpDst dst, OpSrc src, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::FindILsbOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src.flags;
  inst.dstOperands[0].type  = ir::OperandType::i32();
  inst.srcOperands[0].type  = type;
  return inst;
}

InstructionId_t IR::findUMsbOp(OpDst dst, OpSrc src, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::FindUMsbOp);
  inst.dstOperands[0].kind = getOperandKind(dst.kind);
  inst.srcOperands[0].kind = getOperandKind(src.kind);
  inst.dstOperands[0].type = ir::OperandType::i32();
  inst.srcOperands[0].type = type;
  return inst;
}

InstructionId_t IR::findSMsbOp(OpDst dst, OpSrc src, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::FindUMsbOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src.flags;
  inst.dstOperands[0].type  = ir::OperandType::i32();
  inst.srcOperands[0].type  = type;
  return inst;
}

InstructionId_t IR::signExtendI32Op(OpDst dst, OpSrc src, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::SignExtendOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src.flags;
  inst.dstOperands[0].type  = ir::OperandType::i32();
  inst.srcOperands[0].type  = type;
  return inst;
}

InstructionId_t IR::bitsetOp(OpDst dst, OpSrc src, OpSrc offset, OpSrc value, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::BitsetOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src.kind);
  inst.srcOperands[1].kind  = getOperandKind(offset.kind);
  inst.srcOperands[2].kind  = getOperandKind(value.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src.flags;
  inst.srcOperands[1].flags = offset.flags;
  inst.srcOperands[2].flags = value.flags;

  inst.dstOperands[0].type = type;
  inst.srcOperands[0].type = type;
  inst.srcOperands[1].type = ir::OperandType::i32();
  inst.srcOperands[2].type = type;
  return inst;
}

InstructionId_t IR::bitFieldInsertOp(OpDst dst, OpSrc value, OpSrc offset, OpSrc count, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::BitFieldInsertOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(value.kind);
  inst.srcOperands[1].kind  = getOperandKind(offset.kind);
  inst.srcOperands[2].kind  = getOperandKind(count.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = value.flags;
  inst.srcOperands[1].flags = offset.flags;
  inst.srcOperands[2].flags = count.flags;

  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

InstructionId_t IR::bitUIExtractOp(OpDst dst, OpSrc base, OpSrc offset, OpSrc count, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::BitUIExtractOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(base.kind);
  inst.srcOperands[1].kind  = getOperandKind(offset.kind);
  inst.srcOperands[2].kind  = getOperandKind(count.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = base.flags;
  inst.srcOperands[1].flags = offset.flags;
  inst.srcOperands[2].flags = count.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

InstructionId_t IR::bitSIExtractOp(OpDst dst, OpSrc base, OpSrc offset, OpSrc count, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::BitSIExtractOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(base.kind);
  inst.srcOperands[1].kind  = getOperandKind(offset.kind);
  inst.srcOperands[2].kind  = getOperandKind(count.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = base.flags;
  inst.srcOperands[1].flags = offset.flags;
  inst.srcOperands[2].flags = count.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

InstructionId_t IR::bitUIExtractOp(OpDst dst, OpSrc base, OpSrc compact, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::BitUIExtractCompactOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(base.kind);
  inst.srcOperands[1].kind  = getOperandKind(compact.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = base.flags;
  inst.srcOperands[1].flags = compact.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

InstructionId_t IR::bitSIExtractOp(OpDst dst, OpSrc base, OpSrc compact, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::BitSIExtractCompactOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(base.kind);
  inst.srcOperands[1].kind  = getOperandKind(compact.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = base.flags;
  inst.srcOperands[1].flags = compact.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

InstructionId_t IR::bitCmpOp(OpDst dst, OpSrc base, ir::OperandType type, OpSrc index) {
  auto inst                 = ir::getInfo(ir::eInstKind::BitCmpOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(base.kind);
  inst.srcOperands[1].kind  = getOperandKind(index.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = base.flags;
  inst.srcOperands[1].flags = index.flags;

  inst.srcOperands[0].type = type;
  inst.srcOperands[1].type = ir::OperandType::i32();
  return inst;
}

InstructionId_t IR::returnOp() {
  return ir::getInfo(ir::eInstKind::ReturnOp);
}

InstructionId_t IR::discardOp(OpSrc predicate) {
  auto inst                 = ir::getInfo(ir::eInstKind::DiscardOp);
  inst.srcOperands[0].kind  = getOperandKind(predicate.kind);
  inst.srcOperands[0].flags = predicate.flags;
  return inst;
}

InstructionId_t IR::barrierOp() {
  return ir::getInfo(ir::eInstKind::BarrierOp);
}

InstructionId_t IR::jumpAbsOp(OpSrc addr) {
  auto inst                = ir::getInfo(ir::eInstKind::JumpAbsOp);
  inst.srcOperands[0].kind = getOperandKind(addr.kind);
  inst.srcOperands[1].type = ir::OperandType::i64();
  return inst;
}

InstructionId_t IR::cjumpAbsOp(OpSrc predicate, bool invert, OpSrc addr) {
  auto inst                 = ir::getInfo(ir::eInstKind::CondJumpAbsOp);
  inst.srcOperands[0].kind  = getOperandKind(predicate.kind);
  inst.srcOperands[1].kind  = getOperandKind(addr.kind);
  inst.srcOperands[0].flags = predicate.flags;
  inst.srcOperands[1].flags = addr.flags;
  if (invert) inst.srcOperands[0].flags |= OperandFlagsSrc(true, false);
  return inst;
}

InstructionId_t IR::bitFieldMaskOp(OpDst dst, OpSrc size, OpSrc offset, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::BitFieldMaskOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(size.kind);
  inst.srcOperands[1].kind  = getOperandKind(offset.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = size.flags;
  inst.srcOperands[1].flags = offset.flags;
  inst.dstOperands[0].type  = type;
  return inst;
}

InstructionId_t IR::bitAndOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::BitAndOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::bitOrOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::BitOrOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::bitXorOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::BitXorOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::cmpIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type, CmpIPredicate op) {
  auto inst                 = ir::getInfo(ir::eInstKind::CmpIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.srcOperands[0].type = inst.srcOperands[1].type = type;

  inst.userData = (std::underlying_type<CmpIPredicate>::type)op;
  return inst;
}

InstructionId_t IR::cmpFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type, CmpFPredicate op) {
  auto inst                 = ir::getInfo(ir::eInstKind::CmpFOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.srcOperands[0].type = inst.srcOperands[1].type = type;

  inst.userData = (std::underlying_type<CmpFPredicate>::type)op;
  return inst;
}

InstructionId_t IR::mulIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::MulIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::mulIExtendedOp(OpDst low, OpDst high, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::MulIExtendedOp);
  inst.dstOperands[0].kind  = getOperandKind(low.kind);
  inst.dstOperands[1].kind  = getOperandKind(high.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = low.flags;
  inst.dstOperands[1].flags = high.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::max3UIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::Max3UIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.srcOperands[2].kind  = getOperandKind(src2.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.srcOperands[2].flags = src2.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = inst.srcOperands[2].type = type;
  return inst;
}

InstructionId_t IR::maxUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::MaxUIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::max3SIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::Max3SIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.srcOperands[2].kind  = getOperandKind(src2.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.srcOperands[2].flags = src2.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = inst.srcOperands[2].type = type;
  return inst;
}

InstructionId_t IR::maxSIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::MaxSIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::max3FOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::Max3FOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.srcOperands[2].kind  = getOperandKind(src2.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.srcOperands[2].flags = src2.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = inst.srcOperands[2].type = type;
  return inst;
}

InstructionId_t IR::maxFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::MaxFOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::maxNOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::MaxNOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::min3UIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::Min3UIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.srcOperands[2].kind  = getOperandKind(src2.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.srcOperands[2].flags = src2.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = inst.srcOperands[2].type = type;
  return inst;
}

InstructionId_t IR::minUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::MinUIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::min3SIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::Min3SIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.srcOperands[2].kind  = getOperandKind(src2.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.srcOperands[2].flags = src2.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = inst.srcOperands[2].type = type;
  return inst;
}

InstructionId_t IR::minSIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::MinSIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::min3FOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::Min3FOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.srcOperands[2].kind  = getOperandKind(src2.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.srcOperands[2].flags = src2.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = inst.srcOperands[2].type = type;
  return inst;
}

InstructionId_t IR::minFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::MinFOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::minNOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::MinNOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::medUIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::MedUIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.srcOperands[2].kind  = getOperandKind(src2.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.srcOperands[2].flags = src2.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = inst.srcOperands[2].type = type;
  return inst;
}

InstructionId_t IR::medSIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::MedSIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.srcOperands[2].kind  = getOperandKind(src2.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.srcOperands[2].flags = src2.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = inst.srcOperands[2].type = type;
  return inst;
}

InstructionId_t IR::medFOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::MedFOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.srcOperands[2].kind  = getOperandKind(src2.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.srcOperands[2].flags = src2.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = inst.srcOperands[2].type = type;
  return inst;
}

InstructionId_t IR::addIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::AddIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::ldexpOp(OpDst dst, OpSrc vsrc, OpSrc vexp, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::LdexpOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(vsrc.kind);
  inst.srcOperands[1].kind  = getOperandKind(vexp.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = vsrc.flags;
  inst.srcOperands[1].flags = vexp.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

InstructionId_t IR::addFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::AddFOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::subFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::SubFOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::mulFOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::MulFOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::fmaFOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::FmaFOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.srcOperands[2].kind  = getOperandKind(src2.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.srcOperands[2].flags = src2.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = inst.srcOperands[2].type = type;
  return inst;
}

InstructionId_t IR::fmaIOp(OpDst dst, OpSrc src0, OpSrc src1, OpSrc src2, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::FmaIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.srcOperands[2].kind  = getOperandKind(src2.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.srcOperands[2].flags = src2.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = inst.srcOperands[2].type = type;
  return inst;
}

InstructionId_t IR::addcIOp(OpDst dst, OpDst carryOut, OpSrc src0, OpSrc src1, OpSrc carryIn, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::AddCarryIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.dstOperands[1].kind  = getOperandKind(carryOut.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.srcOperands[2].kind  = getOperandKind(carryIn.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.dstOperands[1].flags = carryOut.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.srcOperands[2].flags = carryIn.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::subIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::SubIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::subbIOp(OpDst dst, OpDst carryOut, OpSrc src0, OpSrc src1, OpSrc carryIn, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::SubBurrowIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.dstOperands[1].kind  = getOperandKind(carryOut.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.srcOperands[2].kind  = getOperandKind(carryIn.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.dstOperands[1].flags = carryOut.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.srcOperands[2].flags = carryIn.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

InstructionId_t IR::shiftLUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::ShiftLUIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

InstructionId_t IR::shiftRUIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::ShiftRUIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

InstructionId_t IR::shiftRSIOp(OpDst dst, OpSrc src0, OpSrc src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::ShiftRSIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

InstructionId_t IR::truncFOp(OpDst dst, OpSrc src0) {
  auto inst                 = ir::getInfo(ir::eInstKind::TruncFOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  return inst;
}

InstructionId_t IR::extFOp(OpDst dst, OpSrc src0) {
  auto inst                 = ir::getInfo(ir::eInstKind::ExtFOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  return inst;
}

InstructionId_t IR::convFPToSIOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType) {
  auto inst                 = ir::getInfo(ir::eInstKind::FPToSIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.dstOperands[0].type  = dstType;
  inst.srcOperands[0].type  = srcType;
  return inst;
}

InstructionId_t IR::convSIToFPOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType) {
  auto inst                 = ir::getInfo(ir::eInstKind::SIToFPOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.dstOperands[0].type  = dstType;
  inst.srcOperands[0].type  = srcType;
  return inst;
}

InstructionId_t IR::convFPToUIOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType) {
  auto inst                 = ir::getInfo(ir::eInstKind::FPToUIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.dstOperands[0].type  = dstType;
  inst.srcOperands[0].type  = srcType;
  return inst;
}

InstructionId_t IR::convUIToFPOp(OpDst dst, ir::OperandType dstType, OpSrc src0, ir::OperandType srcType) {
  auto inst                 = ir::getInfo(ir::eInstKind::UIToFPOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.dstOperands[0].type  = dstType;
  inst.srcOperands[0].type  = srcType;
  return inst;
}

InstructionId_t IR::convSI4ToFloat(OpDst dst, OpSrc src0) {
  auto inst                 = ir::getInfo(ir::eInstKind::SI4ToFloat);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  return inst;
}

InstructionId_t IR::packHalf2x16Op(OpDst dst, OpSrc src0, OpSrc src1) {
  auto inst                 = ir::getInfo(ir::eInstKind::PackHalf2x16Op);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  return inst;
}

InstructionId_t IR::unpackHalf2x16(OpDst low, OpDst high, OpSrc src) {
  auto inst                 = ir::getInfo(ir::eInstKind::UnpackHalf2x16);
  inst.dstOperands[0].kind  = getOperandKind(low.kind);
  inst.dstOperands[1].kind  = getOperandKind(high.kind);
  inst.srcOperands[0].kind  = getOperandKind(src.kind);
  inst.dstOperands[0].flags = low.flags;
  inst.dstOperands[1].flags = high.flags;
  inst.srcOperands[0].flags = src.flags;
  return inst;
}

InstructionId_t IR::packSnorm2x16Op(OpDst dst, OpSrc src0, OpSrc src1) {
  auto inst                 = ir::getInfo(ir::eInstKind::PackSnorm2x16Op);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  return inst;
}

InstructionId_t IR::packUnorm2x16Op(OpDst dst, OpSrc src0, OpSrc src1) {
  auto inst                 = ir::getInfo(ir::eInstKind::PackUnorm2x16Op);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.srcOperands[1].kind  = getOperandKind(src1.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.srcOperands[1].flags = src1.flags;
  return inst;
}

InstructionId_t IR::truncOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::TruncOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

InstructionId_t IR::ceilOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::CeilOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

InstructionId_t IR::roundEvenOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::RoundEvenOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

InstructionId_t IR::fractOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::FractOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

InstructionId_t IR::floorOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::FloorOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

InstructionId_t IR::rcpOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::RcpOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

InstructionId_t IR::rsqrtOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::RsqrtOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

InstructionId_t IR::sqrtOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::SqrtOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

InstructionId_t IR::exp2Op(OpDst dst, OpSrc src0) {
  auto inst                 = ir::getInfo(ir::eInstKind::Exp2Op);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  return inst;
}

InstructionId_t IR::log2Op(OpDst dst, OpSrc src0) {
  auto inst                 = ir::getInfo(ir::eInstKind::Log2Op);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  return inst;
}

InstructionId_t IR::sinOp(OpDst dst, OpSrc src0) {
  auto inst                 = ir::getInfo(ir::eInstKind::SinOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  return inst;
}

InstructionId_t IR::cosOp(OpDst dst, OpSrc src0) {
  auto inst                 = ir::getInfo(ir::eInstKind::CosOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  return inst;
}

InstructionId_t IR::clampFMinMaxOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::ClampFMinMaxOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.dstOperands[0].type  = inst.srcOperands[0].type;
  return inst;
}

InstructionId_t IR::clampFZeroOp(OpDst dst, OpSrc src0, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::ClampFZeroOp);
  inst.dstOperands[0].kind  = getOperandKind(dst.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = dst.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.dstOperands[0].type  = inst.srcOperands[0].type;
  return inst;
}

InstructionId_t IR::frexpOp(OpDst exp, OpDst mant, OpSrc src0, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::ClampFZeroOp);
  inst.dstOperands[0].kind  = getOperandKind(exp.kind);
  inst.dstOperands[1].kind  = getOperandKind(mant.kind);
  inst.srcOperands[0].kind  = getOperandKind(src0.kind);
  inst.dstOperands[0].flags = exp.flags;
  inst.dstOperands[1].flags = mant.flags;
  inst.srcOperands[0].flags = src0.flags;
  inst.dstOperands[0].type  = inst.srcOperands[0].type;
  return inst;
}
} // namespace compiler::frontend::translate::create