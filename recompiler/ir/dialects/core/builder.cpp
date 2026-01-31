#include "builder.h"

namespace compiler::ir::dialect::core {

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
  if (rhs.ssa.isValid()) ir.connect(id, rhs.ssa);
  return op;
}

IRResult MoveOp::create(IROperations& ir, OpDst const& dst, OpSrc const& src, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::MoveOp));
  createOp(ir, ir.getSrc(id, 0), src, type);
  createOp(ir, ir.getDst(id, 0), dst, type);
  return IRResult(ir, id);
}

IRResult SelectOp::create(IROperations& ir, OpDst const& dst, OpSrc predicate, OpSrc srcTrue, OpSrc srcFalse, OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::SelectOp));
  createOp(ir, ir.getSrc(id, 0), predicate, OperandType::i1());
  createOp(ir, ir.getSrc(id, 1), srcTrue, type);
  createOp(ir, ir.getSrc(id, 2), srcFalse, type);
  createOp(ir, ir.getDst(id, 0), dst, type);
  return IRResult(ir, id);
}

IRResult YieldOp::create(IROperations& ir, std::span<SsaId_t> inputs) {
  auto info   = getInfo(eInstKind::YieldOp);
  info.numSrc = inputs.size();
  auto id     = ir.createInstruction(info);

  for (size_t n = 0; n < inputs.size(); ++n) {
    auto  srcId = ir.getSrc(id, n);
    auto& op    = ir.getOperand(srcId);

    op.ssaId = inputs[n];
    ir.connect(srcId, inputs[n]);
  }

  return IRResult(ir, id);
}

IRResult ReturnOp::create(IROperations& ir) {
  return IRResult(ir, ir.createInstruction(getInfo(eInstKind::ReturnOp)));
}

IRResult DiscardOp::create(IROperations& ir, OpSrc const& predicate) {
  auto id = ir.createInstruction(getInfo(eInstKind::DiscardOp));
  createOp(ir, ir.getSrc(id, 0), predicate, OperandType::i1());
  return IRResult(ir, id);
}

IRResult BarrierOp::create(IROperations& ir) {
  return IRResult(ir, ir.createInstruction(getInfo(eInstKind::BarrierOp)));
}

IRResult JumpAbsOp::create(IROperations& ir, OpSrc const& addr) {
  auto id = ir.createInstruction(getInfo(eInstKind::JumpAbsOp));
  createOp(ir, ir.getSrc(id, 0), addr, OperandType::i64());
  return IRResult(ir, id);
}

IRResult CjumpAbsOp::create(IROperations& ir, OpSrc const& predicate, OpSrc const& addr) {
  auto id = ir.createInstruction(getInfo(eInstKind::CondJumpAbsOp));
  createOp(ir, ir.getSrc(id, 0), predicate, OperandType::i1());
  createOp(ir, ir.getSrc(id, 1), addr, OperandType::i64());
  return IRResult(ir, id);
}

IRResult ConstantOp::create(IROperations& ir, OpDst const& dst, ir::ConstantValue value, ir::OperandType type) {
  auto id = ir.createInstruction(getInfo(eInstKind::ConstantOp));
  createOp(ir, ir.getDst(id, 0), dst, type);

  ir.accessInstr(id).constantId = ir.createConstant(value);
  return IRResult(ir, id);
}

} // namespace compiler::ir::dialect::core