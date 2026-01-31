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
InstructionKind_t handleSopc(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = SOPC(**pCode);
  auto const op   = (parser::eOpcode)(OPcodeStart_SOPC + inst.template get<SOPC::Field::OP>());

  auto src0 = createSrc(eOperandKind((eOperandKind_t)inst.template get<SOPC::Field::SSRC0>()));
  auto src1 = createSrc(eOperandKind((eOperandKind_t)inst.template get<SOPC::Field::SSRC1>()));

  if (eOperandKind(src0.kind).isLiteral()) {
    *pCode += 1;
    src0 = createSrc(ctx.create<ir::dialect::core::ConstantOp>(createDst(), ir::ConstantValue {.value_u64 = **pCode}, ir::OperandType::i32()));
  } else if (eOperandKind(src1.kind).isLiteral()) {
    *pCode += 1;
    src1 = createSrc(ctx.create<ir::dialect::core::ConstantOp>(createDst(), ir::ConstantValue {.value_u64 = **pCode}, ir::OperandType::i32()));
  }

  *pCode += 1;

  using namespace ir::dialect::arith;
  switch (op) {
    case eOpcode::S_CMP_EQ_I32: {
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::eq);
    } break;
    case eOpcode::S_CMP_LG_I32: {
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_CMP_GT_I32: {
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::sgt);
    } break;
    case eOpcode::S_CMP_GE_I32: {
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::sge);
    } break;
    case eOpcode::S_CMP_LT_I32: {
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::slt);
    } break;
    case eOpcode::S_CMP_LE_I32: {
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::sle);
    } break;
    case eOpcode::S_CMP_EQ_U32: {
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::eq);
    } break;
    case eOpcode::S_CMP_LG_U32: {
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_CMP_GT_U32: {
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::ugt);
    } break;
    case eOpcode::S_CMP_GE_U32: {
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::uge);
    } break;
    case eOpcode::S_CMP_LT_U32: {
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::ult);
    } break;
    case eOpcode::S_CMP_LE_U32: {
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::ule);
    } break;
    case eOpcode::S_BITCMP0_B32: {
      auto in0 = ctx.create<ir::dialect::arith::NotOp>(createDst(), src0, ir::OperandType::i32());
      ctx.create<BitCmpOp>(createDst(eOperandKind::SCC()), in0, src1, ir::OperandType::i32());
    } break;
    case eOpcode::S_BITCMP1_B32: {
      auto in0 = ctx.create<ir::dialect::arith::NotOp>(createDst(), src0, ir::OperandType::i32());
      ctx.create<BitCmpOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32());
    } break;
    case eOpcode::S_BITCMP0_B64: {
      ctx.create<BitCmpOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i64());
    } break;
    case eOpcode::S_BITCMP1_B64: {
      ctx.create<BitCmpOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i64());
    } break;
    case eOpcode::S_SETVSKIP: {
      ctx.create<BitCmpOp>(createDst(eOperandKind::VSKIP()), src0, src1, ir::OperandType::i32());
    } break;
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }
  return conv(op);
}
} // namespace compiler::frontend::translate