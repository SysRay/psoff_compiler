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

ir::InstCore constantOp(eOperandKind dst, int16_t value, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::ConstantOp);
  inst.srcConstant.value   = (uint64_t)((int64_t)value);
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

ir::InstCore absDiff(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type) {
  auto inst                 = ir::getInfo(ir::eInstKind::SubIOp);
  inst.dstOperands[0].kind  = getOperandKind(dst);
  inst.srcOperands[0].kind  = getOperandKind(src0);
  inst.srcOperands[1].kind  = getOperandKind(src1);
  inst.srcOperands[0].flags = OperandFlagsSrc(eRegClass::SGPR, false, true);
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

ir::InstCore selectOp(eOperandKind dst, eOperandKind predicate, eOperandKind srcTrue, eOperandKind srcFalse, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::SelectOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(predicate);
  inst.srcOperands[1].kind = getOperandKind(srcTrue);
  inst.srcOperands[2].kind = getOperandKind(srcFalse);
  inst.dstOperands[1].type = inst.srcOperands[1].type = inst.srcOperands[2].type = type;
  return inst;
}

ir::InstCore bitReverseOp(eOperandKind dst, eOperandKind src, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::BitReverseOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src);
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

ir::InstCore bitCountOp(eOperandKind dst, eOperandKind src, ir::OperandType type, bool value) {
  auto inst                = ir::getInfo(ir::eInstKind::BitCountOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src);
  inst.dstOperands[0].type = ir::OperandType::i32();
  inst.srcOperands[0].type = type;
  if (!value) inst.srcOperands[0].flags = OperandFlagsSrc(eRegClass::SGPR, true, false); // invert it
  return inst;
}

ir::InstCore findILsbOp(eOperandKind dst, eOperandKind src, ir::OperandType type, bool value) {
  auto inst                = ir::getInfo(ir::eInstKind::FindILsbOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src);
  inst.dstOperands[0].type = ir::OperandType::i32();
  inst.srcOperands[0].type = type;
  if (!value) inst.srcOperands[0].flags = OperandFlagsSrc(eRegClass::SGPR, true, false); // invert it
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
  inst.srcOperands[1].kind = getOperandKind(offset);
  inst.srcOperands[2].kind = getOperandKind(value);
  inst.dstOperands[0].type = type;
  inst.srcOperands[0].type = type;
  inst.srcOperands[1].type = ir::OperandType::i32();
  inst.srcOperands[2].type = type;
  return inst;
}

ir::InstCore bitFieldInsertOp(eOperandKind dst, eOperandKind value, eOperandKind offset, eOperandKind count, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::BitFieldInsertOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(value);
  inst.srcOperands[1].kind = getOperandKind(offset);
  inst.srcOperands[2].kind = getOperandKind(count);
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

ir::InstCore bitUIExtractOp(eOperandKind dst, eOperandKind base, eOperandKind offset, eOperandKind count, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::BitUIExtractOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(base);
  inst.srcOperands[1].kind = getOperandKind(offset);
  inst.srcOperands[2].kind = getOperandKind(count);
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

ir::InstCore bitSIExtractOp(eOperandKind dst, eOperandKind base, eOperandKind offset, eOperandKind count, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::BitSIExtractOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(base);
  inst.srcOperands[1].kind = getOperandKind(offset);
  inst.srcOperands[2].kind = getOperandKind(count);
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

ir::InstCore bitUIExtractOp(eOperandKind dst, eOperandKind base, eOperandKind compact, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::BitUIExtractCompactOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(base);
  inst.srcOperands[1].kind = getOperandKind(compact);
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;

  return inst;
}

ir::InstCore bitSIExtractOp(eOperandKind dst, eOperandKind base, eOperandKind compact, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::BitSIExtractCompactOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(base);
  inst.srcOperands[1].kind = getOperandKind(compact);
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

ir::InstCore bitCmpOp(eOperandKind dst, eOperandKind base, ir::OperandType type, eOperandKind index, bool value) {
  auto inst                = ir::getInfo(ir::eInstKind::BitCmpOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(base);
  inst.srcOperands[1].kind = getOperandKind(index);

  inst.srcOperands[0].type = type;
  inst.srcOperands[1].type = ir::OperandType::i32();
  if (!value) inst.dstOperands[0].flags = OperandFlagsDst(eRegClass::SGPR, 0, false, true); // invert it
  return inst;
}

ir::InstCore jumpAbsOp(eOperandKind addr) {
  auto inst                = ir::getInfo(ir::eInstKind::JumpAbsOp);
  inst.srcOperands[0].kind = getOperandKind(addr);
  inst.srcOperands[1].type = ir::OperandType::i64();
  return inst;
}

ir::InstCore jumpAbsOp(uint64_t addr) {
  auto inst              = ir::getInfo(ir::eInstKind::JumpAbsOp);
  inst.srcConstant.value = addr;
  inst.srcConstant.type  = ir::OperandType::i64();
  inst.flags |= ir::Flags<ir::eInstructionFlags>(ir::eInstructionFlags::kConstant);
  return inst;
}

ir::InstCore bitFieldMaskOp(eOperandKind dst, eOperandKind size, eOperandKind offset, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::BitFieldMaskOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(size);
  inst.srcOperands[1].kind = getOperandKind(offset);
  inst.dstOperands[0].type = type;
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

ir::InstCore mulIOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::MulIOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src0);
  inst.srcOperands[1].kind = getOperandKind(src1);
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

ir::InstCore addIOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::AddIOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src0);
  inst.srcOperands[1].kind = getOperandKind(src1);
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

ir::InstCore addcIOp(eOperandKind dst, eOperandKind carryOut, eOperandKind src0, eOperandKind src1, eOperandKind carryIn, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::AddCarryIOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.dstOperands[1].kind = getOperandKind(carryOut);
  inst.srcOperands[0].kind = getOperandKind(src0);
  inst.srcOperands[1].kind = getOperandKind(src1);
  inst.srcOperands[2].kind = getOperandKind(carryIn);
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

ir::InstCore subIOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::SubIOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src0);
  inst.srcOperands[1].kind = getOperandKind(src1);
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

ir::InstCore subbIOp(eOperandKind dst, eOperandKind carryOut, eOperandKind src0, eOperandKind src1, eOperandKind carryIn, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::SubBurrowIOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.dstOperands[1].kind = getOperandKind(carryOut);
  inst.srcOperands[0].kind = getOperandKind(src0);
  inst.srcOperands[1].kind = getOperandKind(src1);
  inst.srcOperands[2].kind = getOperandKind(carryIn);
  inst.dstOperands[0].type = inst.srcOperands[0].type = inst.srcOperands[1].type = type;
  return inst;
}

ir::InstCore shiftLUIOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::ShiftLUIOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src0);
  inst.srcOperands[1].kind = getOperandKind(src1);
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

ir::InstCore shiftRUIOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::ShiftRUIOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src0);
  inst.srcOperands[1].kind = getOperandKind(src1);
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}

ir::InstCore shiftRSIOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type) {
  auto inst                = ir::getInfo(ir::eInstKind::ShiftRSIOp);
  inst.dstOperands[0].kind = getOperandKind(dst);
  inst.srcOperands[0].kind = getOperandKind(src0);
  inst.srcOperands[1].kind = getOperandKind(src1);
  inst.dstOperands[0].type = inst.srcOperands[0].type = type;
  return inst;
}
} // namespace compiler::frontend::translate::create