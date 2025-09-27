#include "../../frontend.h"
#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "instruction_builder.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
InstructionKind_t handleSopk(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = SOPK(**pCode);
  auto const op   = (eOpcode)(OPcodeStart_SOPK + inst.template get<SOPK::Field::OP>());

  auto const sdst  = OpDst(eOperandKind((eOperandKind_t)inst.template get<SOPK::Field::SDST>()));
  auto const imm16 = (int16_t)inst.template get<SOPK::Field::SIMM16>();

  *pCode += 1;

  switch (op) {
    case eOpcode::S_MOVK_I32: {
      builder.createInstruction(create::constantOp(sdst, imm16, ir::OperandType::i32()));
    } break;
    case eOpcode::S_MOVK_HI_I32: {
      builder.createInstruction(create::literalOp((uint32_t)imm16 << 16u));
      builder.createInstruction(create::bitFieldInsertOp(sdst, OpSrc(eOperandKind::Temp0()), OpSrc(OpSrc(eOperandKind::createImm(16))),
                                                         OpSrc(OpSrc(eOperandKind::createImm(16))), ir::OperandType::i32()));
    } break;
    case eOpcode::S_CMOVK_I32: {
      builder.createInstruction(create::literalOp((uint32_t)(int32_t)imm16));
      builder.createInstruction(create::selectOp(sdst, OpSrc(eOperandKind::SCC()), OpSrc(eOperandKind::Literal()), OpSrc(sdst.kind), ir::OperandType::i32()));
    } break;
    case eOpcode::S_CMPK_EQ_I32: {
      builder.createInstruction(create::literalOp((uint32_t)(int32_t)imm16));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), OpSrc(eOperandKind::Literal()), ir::OperandType::i32(), CmpIPredicate::eq));
    } break;
    case eOpcode::S_CMPK_LG_I32: {
      builder.createInstruction(create::literalOp((uint32_t)(int32_t)imm16));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), OpSrc(eOperandKind::Literal()), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_CMPK_GT_I32: {
      builder.createInstruction(create::literalOp((uint32_t)(int32_t)imm16));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), OpSrc(eOperandKind::Literal()), ir::OperandType::i32(), CmpIPredicate::sgt));
    } break;
    case eOpcode::S_CMPK_GE_I32: {
      builder.createInstruction(create::literalOp((uint32_t)(int32_t)imm16));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), OpSrc(eOperandKind::Literal()), ir::OperandType::i32(), CmpIPredicate::sge));
    } break;
    case eOpcode::S_CMPK_LT_I32: {
      builder.createInstruction(create::literalOp((uint32_t)(int32_t)imm16));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), OpSrc(eOperandKind::Literal()), ir::OperandType::i32(), CmpIPredicate::slt));
    } break;
    case eOpcode::S_CMPK_LE_I32: {
      builder.createInstruction(create::literalOp((uint32_t)(int32_t)imm16));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), OpSrc(eOperandKind::Literal()), ir::OperandType::i32(), CmpIPredicate::sle));
    } break;
    case eOpcode::S_CMPK_EQ_U32: {
      builder.createInstruction(create::literalOp((uint32_t)(uint16_t)imm16));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), OpSrc(eOperandKind::Literal()), ir::OperandType::i32(), CmpIPredicate::eq));
    } break;
    case eOpcode::S_CMPK_LG_U32: {
      builder.createInstruction(create::literalOp((uint32_t)(uint16_t)imm16));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), OpSrc(eOperandKind::Literal()), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_CMPK_GT_U32: {
      builder.createInstruction(create::literalOp((uint32_t)(uint16_t)imm16));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), OpSrc(eOperandKind::Literal()), ir::OperandType::i32(), CmpIPredicate::ugt));
    } break;
    case eOpcode::S_CMPK_GE_U32: {
      builder.createInstruction(create::literalOp((uint32_t)(uint16_t)imm16));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), OpSrc(eOperandKind::Literal()), ir::OperandType::i32(), CmpIPredicate::uge));
    } break;
    case eOpcode::S_CMPK_LT_U32: {
      builder.createInstruction(create::literalOp((uint32_t)(uint16_t)imm16));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), OpSrc(eOperandKind::Literal()), ir::OperandType::i32(), CmpIPredicate::ult));
    } break;
    case eOpcode::S_CMPK_LE_U32: {
      builder.createInstruction(create::literalOp((uint32_t)(uint16_t)imm16));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), OpSrc(eOperandKind::Literal()), ir::OperandType::i32(), CmpIPredicate::ule));
    } break;
    case eOpcode::S_ADDK_I32: {
      builder.createInstruction(create::literalOp((uint32_t)(int32_t)imm16));
      builder.createInstruction(create::addIOp(sdst, OpSrc(eOperandKind::Literal()), OpSrc(sdst.kind), ir::OperandType::i32()));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), OpSrc(eOperandKind::Literal()), ir::OperandType::i32(), CmpIPredicate::ult));
    } break;
    case eOpcode::S_MULK_I32: {
      builder.createInstruction(create::literalOp((uint32_t)(int32_t)imm16));
      builder.createInstruction(create::mulIOp(sdst, OpSrc(eOperandKind::Literal()), OpSrc(sdst.kind), ir::OperandType::i32()));
    } break;
      // case eOpcode::S_CBRANCH_I_FORK: {} break; // todo
    // case eOpcode::S_GETREG_B32: {} break; // todo
    // case eOpcode::S_SETREG_B32: {} break; // todo
    // case eOpcode::S_GETREG_REGRD_B32: {} break;// todo
    // case eOpcode::S_SETREG_IMM32_B32: { // todo
    //   builder.createInstruction(create::literalOp(**pCode));
    //   *pCode += 1;
    // } break;
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }
  return conv(op);
}
} // namespace compiler::frontend::translate