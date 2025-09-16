#include "instruction_builder.h"

// #include "ir/types.h"

namespace compiler::frontend::translate::create {
ir::InstCore literalOp(uint32_t value) {
  auto inst                = ir::getInfo(ir::eInstKind::ConstantOp);
  inst.srcConstant.value   = value;
  inst.dstOperands[0].kind = getOperandKind(eOperandKind::Literal);
  inst.dstOperands[0].type = inst.srcConstant.type = ir::OperandType::i32();
  return inst;
}

ir::InstCore constantOp(eOperandKind dst, uint64_t value, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::ConstantOp);
  inst.srcConstant.value   = value;
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.dstOperands[0].type = inst.srcConstant.type = type;
  return inst;
}

ir::InstCore moveOp(eOperandKind dst, eOperandKind src, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::MoveOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src);
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

ir::InstCore notOp(eOperandKind dst, eOperandKind src, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::MoveOp);
  inst.dstOperands[0].kind  = getOperandKind(dst);
  inst.srcOperands[0].kind  = getOperandKind(src);
  inst.srcOperands[0].flags = OperandFlagsSrc(eRegClass::SGPR, true, false);
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

ir::InstCore absOp(eOperandKind dst, eOperandKind src, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::MoveOp);
  inst.dstOperands[0].kind  = getOperandKind(dst);
  inst.srcOperands[0].kind  = getOperandKind(src);
  inst.srcOperands[0].flags = OperandFlagsSrc(eRegClass::SGPR, false, true);
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

ir::InstCore condMoveOp(eOperandKind dst, eOperandKind src, eOperandKind predicate, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::CMoveOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(predicate);
  inst.srcOperands[1].kind = getOperandKind(src);
  inst.dstOperands[1].type = inst.srcOperands[1].type = type;
  return inst;
}

ir::InstCore bitReverseOp(eOperandKind dst, eOperandKind src, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::BitReverseOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src);
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

ir::InstCore bitCountOp(eOperandKind dst, eOperandKind src, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::BitCountOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src);
  inst.dstOperands[0].type = ir::OperandType::i32();
  inst.srcOperands[0].type = type;
  return inst;
}

ir::InstCore findILsbOp(eOperandKind dst, eOperandKind src, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::FindILsbOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src);
  inst.dstOperands[0].type = ir::OperandType::i32();
  inst.srcOperands[0].type = type;
  return inst;
}

ir::InstCore findUMsbOp(eOperandKind dst, eOperandKind src, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::FindUMsbOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src);
  inst.dstOperands[0].type = ir::OperandType::i32();
  inst.srcOperands[0].type = type;
  return inst;
}

ir::InstCore findSMsbOp(eOperandKind dst, eOperandKind src, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::FindUMsbOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src);
  inst.dstOperands[0].type = ir::OperandType::i32();
  inst.srcOperands[0].type = type;
  return inst;
}

ir::InstCore signExtendI32Op(eOperandKind dst, eOperandKind src, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::SignExtendOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src);
  inst.dstOperands[0].type = ir::OperandType::i32();
  inst.srcOperands[0].type = type;
  return inst;
}

ir::InstCore bitsetOp(eOperandKind dst, eOperandKind src, eOperandKind offset, eOperandKind value, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::BitsetOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src);
  inst.srcOperands[0].kind = getOperandKind(offset);
  inst.srcOperands[0].kind = getOperandKind(value);
  inst.dstOperands[0].type = type;
  inst.srcOperands[0].type = type;
  inst.srcOperands[1].type = ir::OperandType::i32();
  inst.srcOperands[2].type = type;
  return inst;
}

ir::InstCore jumpAbsOp(eOperandKind addr) {
  auto inst                = ir::getInfo(ir::eInstKind::JumpAbsOp);
  inst.srcOperands[0].kind = getOperandKind(addr);
  inst.srcOperands[1].type = ir::OperandType::i64();
  return inst;
}

ir::InstCore bitAndOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::BitAndOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src0);
  inst.srcOperands[1].kind = getOperandKind(src1);
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

ir::InstCore bitOrOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::BitOrOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src0);
  inst.srcOperands[1].kind = getOperandKind(src1);
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

ir::InstCore bitXorOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::BitXorOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src0);
  inst.srcOperands[1].kind = getOperandKind(src1);
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

ir::InstCore bitAndNOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::BitAndOp);
  inst.dstOperands[0].kind  = getOperandKind(dst);
  inst.srcOperands[0].kind  = getOperandKind(src0);
  inst.srcOperands[1].kind  = getOperandKind(src1);
  inst.srcOperands[0].flags = OperandFlagsSrc(eRegClass::SGPR, true, false);
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

ir::InstCore bitOrNOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::BitOrOp);
  inst.dstOperands[0].kind  = getOperandKind(dst);
  inst.srcOperands[0].kind  = getOperandKind(src0);
  inst.srcOperands[1].kind  = getOperandKind(src1);
  inst.srcOperands[0].flags = OperandFlagsSrc(eRegClass::SGPR, true, false);
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

ir::InstCore bitNandOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::BitAndOp);
  inst.dstOperands[0].kind  = getOperandKind(dst);
  inst.srcOperands[0].kind  = getOperandKind(src0);
  inst.srcOperands[1].kind  = getOperandKind(src1);
  inst.dstOperands[0].flags = OperandFlagsDst(eRegClass::SGPR, 0, false, true);
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

ir::InstCore bitNorOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::BitOrOp);
  inst.dstOperands[0].kind  = getOperandKind(dst);
  inst.srcOperands[0].kind  = getOperandKind(src0);
  inst.srcOperands[1].kind  = getOperandKind(src1);
  inst.dstOperands[0].flags = OperandFlagsDst(eRegClass::SGPR, 0, false, true);
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

ir::InstCore bitXnorOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::BitXorOp);
  inst.dstOperands[0].kind  = getOperandKind(dst);
  inst.srcOperands[0].kind  = getOperandKind(src0);
  inst.srcOperands[1].kind  = getOperandKind(src1);
  inst.dstOperands[0].flags = OperandFlagsDst(eRegClass::SGPR, 0, false, true);
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

ir::InstCore cmpIOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type, CmpIPredicate op) {
  auto inst                = ir::getInfo(ir::eInstKind::CmpIOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src0);
  inst.srcOperands[1].kind = getOperandKind(src1);
  inst.srcOperands[0].type = inst.srcOperands[1].type = type;

  inst.userData = (std::underlying_type<CmpIPredicate>::type)op;
  return inst;
}
} // namespace compiler::frontend::translate::create