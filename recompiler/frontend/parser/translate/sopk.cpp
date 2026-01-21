#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "frontend/ir_types.h"
#include "ir/dialects/arith/builder.h"
#include "ir/dialects/core/builder.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
InstructionKind_t handleSopk(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = SOPK(**pCode);
  auto const op   = (eOpcode)(OPcodeStart_SOPK + inst.template get<SOPK::Field::OP>());

  auto const sdst  = createDst(eOperandKind((eOperandKind_t)inst.template get<SOPK::Field::SDST>()));
  auto const imm16 = (int16_t)inst.template get<SOPK::Field::SIMM16>();

  *pCode += 1;

  using namespace ir::dialect;
  switch (op) {
    case eOpcode::S_MOVK_I32: {
      ctx.create<core::ConstantOp>(sdst, ir::ConstantValue {.value_i64 = imm16}, ir::OperandType::i32());
    } break;
    case eOpcode::S_MOVK_HI_I32: {
      auto const K = ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = (uint32_t)imm16}, ir::OperandType::i32());
      ctx.create<arith::BitFieldInsertOp>(sdst, K, createSrc(eOperandKind::createImm(16)), createSrc(eOperandKind::createImm(16)), ir::OperandType::i32());
    } break;
    case eOpcode::S_CMOVK_I32: {
      auto const K = ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = imm16}, ir::OperandType::i32());
      ctx.create<core::SelectOp>(sdst, createSrc(eOperandKind::SCC()), K, createSrc(eOperandKind(sdst.kind)), ir::OperandType::i32());
    } break;
    case eOpcode::S_CMPK_EQ_I32: {
      auto const K = ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = imm16}, ir::OperandType::i32());
      ctx.create<arith::CmpIOp>(createDst(eOperandKind::SCC()), createSrc(eOperandKind(sdst.kind)), K, ir::OperandType::i32(), arith::CmpIPredicate::eq);
    } break;
    case eOpcode::S_CMPK_LG_I32: {
      auto const K = ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = imm16}, ir::OperandType::i32());
      ctx.create<arith::CmpIOp>(createDst(eOperandKind::SCC()), createSrc(eOperandKind(sdst.kind)), K, ir::OperandType::i32(), arith::CmpIPredicate::ne);
    } break;
    case eOpcode::S_CMPK_GT_I32: {
      auto const K = ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = imm16}, ir::OperandType::i32());
      ctx.create<arith::CmpIOp>(createDst(eOperandKind::SCC()), createSrc(eOperandKind(sdst.kind)), K, ir::OperandType::i32(), arith::CmpIPredicate::sgt);
    } break;
    case eOpcode::S_CMPK_GE_I32: {
      auto const K = ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = imm16}, ir::OperandType::i32());
      ctx.create<arith::CmpIOp>(createDst(eOperandKind::SCC()), createSrc(eOperandKind(sdst.kind)), K, ir::OperandType::i32(), arith::CmpIPredicate::sge);
    } break;
    case eOpcode::S_CMPK_LT_I32: {
      auto const K = ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = imm16}, ir::OperandType::i32());
      ctx.create<arith::CmpIOp>(createDst(eOperandKind::SCC()), createSrc(eOperandKind(sdst.kind)), K, ir::OperandType::i32(), arith::CmpIPredicate::slt);
    } break;
    case eOpcode::S_CMPK_LE_I32: {
      auto const K = ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = imm16}, ir::OperandType::i32());
      ctx.create<arith::CmpIOp>(createDst(eOperandKind::SCC()), createSrc(eOperandKind(sdst.kind)), K, ir::OperandType::i32(), arith::CmpIPredicate::sle);
    } break;
    case eOpcode::S_CMPK_EQ_U32: {
      auto const K = ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = (uint16_t)imm16}, ir::OperandType::i32());
      ctx.create<arith::CmpIOp>(createDst(eOperandKind::SCC()), createSrc(eOperandKind(sdst.kind)), K, ir::OperandType::i32(), arith::CmpIPredicate::eq);
    } break;
    case eOpcode::S_CMPK_LG_U32: {
      auto const K = ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = (uint16_t)imm16}, ir::OperandType::i32());
      ctx.create<arith::CmpIOp>(createDst(eOperandKind::SCC()), createSrc(eOperandKind(sdst.kind)), K, ir::OperandType::i32(), arith::CmpIPredicate::ne);
    } break;
    case eOpcode::S_CMPK_GT_U32: {
      auto const K = ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = (uint16_t)imm16}, ir::OperandType::i32());
      ctx.create<arith::CmpIOp>(createDst(eOperandKind::SCC()), createSrc(eOperandKind(sdst.kind)), K, ir::OperandType::i32(), arith::CmpIPredicate::ugt);
    } break;
    case eOpcode::S_CMPK_GE_U32: {
      auto const K = ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = (uint16_t)imm16}, ir::OperandType::i32());
      ctx.create<arith::CmpIOp>(createDst(eOperandKind::SCC()), createSrc(eOperandKind(sdst.kind)), K, ir::OperandType::i32(), arith::CmpIPredicate::uge);
    } break;
    case eOpcode::S_CMPK_LT_U32: {
      auto const K = ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = (uint16_t)imm16}, ir::OperandType::i32());
      ctx.create<arith::CmpIOp>(createDst(eOperandKind::SCC()), createSrc(eOperandKind(sdst.kind)), K, ir::OperandType::i32(), arith::CmpIPredicate::ult);
    } break;
    case eOpcode::S_CMPK_LE_U32: {
      auto const K = ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = (uint16_t)imm16}, ir::OperandType::i32());
      ctx.create<arith::CmpIOp>(createDst(eOperandKind::SCC()), createSrc(eOperandKind(sdst.kind)), K, ir::OperandType::i32(), arith::CmpIPredicate::ule);
    } break;
    case eOpcode::S_ADDK_I32: {
      auto const K = ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = imm16}, ir::OperandType::i32());
      ctx.create<arith::AddIOp>(sdst, K, createSrc(eOperandKind(sdst.kind)), ir::OperandType::i32());
      ctx.create<arith::CmpIOp>(createDst(eOperandKind::SCC()), createSrc(eOperandKind(sdst.kind)), K, ir::OperandType::i32(), arith::CmpIPredicate::ult);
    } break;
    case eOpcode::S_MULK_I32: {
      auto const K = ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = imm16}, ir::OperandType::i32());
      ctx.create<arith::MulIOp>(sdst, K, createSrc(eOperandKind(sdst.kind)), ir::OperandType::i32());
    } break;
      // case eOpcode::S_CBRANCH_I_FORK: {} break; // todo
    // case eOpcode::S_GETREG_B32: {} break; // todo
    // case eOpcode::S_SETREG_B32: {} break; // todo
    // case eOpcode::S_GETREG_REGRD_B32: {} break;// todo
    // case eOpcode::S_SETREG_IMM32_B32: { // todo
    //   ir.literalOp(**pCode);;
    //   *pCode += 1;
    // } break;
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }
  return conv(op);
}
} // namespace compiler::frontend::translate