#include "builder.h"

namespace compiler::ir::dialect::core {

static inline OutputOperand& createOp(OutputOperand& lhs, OpDst const& rhs, OperandType type) {
  lhs.kind  = rhs.kind;
  lhs.flags = rhs.flags;
  lhs.type  = type;
  return lhs;
}

static inline InputOperand& createOp(InstructionManager& ir, InputOperand& lhs, OpSrc const& rhs, OperandType type) {
  lhs.kind  = rhs.kind;
  lhs.flags = rhs.flags;
  lhs.type  = type;
  lhs.ssaId = rhs.ssa;
  if (lhs.ssaId.isValid()) ir.connect(lhs.id, lhs.ssaId);
  return lhs;
}

IRResult MoveOp::create(InstructionManager& ir, OpDst const& dst, OpSrc const& src, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::MoveOp));
  createOp(ir, ir.getSrc(id, 0), src, type);
  createOp(ir.getDst(id, 0), dst, type);
  return IRResult(ir, id);
}

IRResult SelectOp::create(InstructionManager& ir, OpDst const& dst, OpSrc predicate, OpSrc srcTrue, OpSrc srcFalse, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::SelectOp));
  createOp(ir, ir.getSrc(id, 0), predicate, OperandType::i1());
  createOp(ir, ir.getSrc(id, 1), srcTrue, type);
  createOp(ir, ir.getSrc(id, 2), srcFalse, type);
  createOp(ir.getDst(id, 0), dst, type);
  return IRResult(ir, id);
}

IRResult YieldOp::create(InstructionManager& ir, std::span<InputOperand> inputs) {
  auto info   = getInfo(eInstKind::YieldOp);
  info.numSrc = inputs.size();
  auto id     = ir.createInstruction(info);

  for (size_t n = 0; n < inputs.size(); ++n)
    ir.getSrc(id, n) = inputs[n];

  return IRResult(ir, id);
}

IRResult ReturnOp::create(InstructionManager& ir) {
  return IRResult(ir, ir.createInstruction(getInfo(eInstKind::ReturnOp)));
}

IRResult DiscardOp::create(InstructionManager& ir, OpSrc const& predicate) {
  auto id = ir.createInstruction(getInfo(eInstKind::DiscardOp));
  createOp(ir, ir.getSrc(id, 0), predicate, OperandType::i1());
  return IRResult(ir, id);
}

IRResult BarrierOp::create(InstructionManager& ir) {
  return IRResult(ir, ir.createInstruction(getInfo(eInstKind::BarrierOp)));
}

IRResult JumpAbsOp::create(InstructionManager& ir, OpSrc const& addr) {
  auto id = ir.createInstruction(getInfo(eInstKind::JumpAbsOp));
  createOp(ir, ir.getSrc(id, 0), addr, OperandType::i64());
  return IRResult(ir, id);
}

IRResult CjumpAbsOp::create(InstructionManager& ir, OpSrc const& predicate, OpSrc const& addr) {
  auto id = ir.createInstruction(getInfo(eInstKind::CondJumpAbsOp));
  createOp(ir, ir.getSrc(id, 0), predicate, OperandType::i1());
  createOp(ir, ir.getSrc(id, 1), addr, OperandType::i64());
  return IRResult(ir, id);
}

IRResult ConstantOp::create(InstructionManager& ir, OpDst const& dst, ir::ConstantValue value, ir::OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::ConstantOp));
  createOp(ir.getDst(id, 0), dst, type);

  ir.accessInstr(id).constantId = ir.createConstant(value);
  return IRResult(ir, id);
}

} // namespace compiler::ir::dialect::core