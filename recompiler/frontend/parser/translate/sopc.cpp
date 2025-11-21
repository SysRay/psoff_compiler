#include "../debug_strings.h"
#include "../instruction_builder.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "frontend/ir_types.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
InstructionKind_t handleSopc(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = SOPC(**pCode);
  auto const op   = (parser::eOpcode)(OPcodeStart_SOPC + inst.template get<SOPC::Field::OP>());

  auto src0 = OpSrc(eOperandKind((eOperandKind_t)inst.template get<SOPC::Field::SSRC0>()));
  auto src1 = OpSrc(eOperandKind((eOperandKind_t)inst.template get<SOPC::Field::SSRC1>()));

  create::IRBuilder ir(ctx.instructions);
  if (src0.kind.isLiteral()) {
    *pCode += 1;
    src0 = OpSrc(ir.literalOp(**pCode));
  } else if (src1.kind.isLiteral()) {
    *pCode += 1;
    src1 = OpSrc(ir.literalOp(**pCode));
  }

  *pCode += 1;

  switch (op) {
    case eOpcode::S_CMP_EQ_I32: {
      ir.cmpIOp(OpDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::eq);
    } break;
    case eOpcode::S_CMP_LG_I32: {
      ir.cmpIOp(OpDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_CMP_GT_I32: {
      ir.cmpIOp(OpDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::sgt);
    } break;
    case eOpcode::S_CMP_GE_I32: {
      ir.cmpIOp(OpDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::sge);
    } break;
    case eOpcode::S_CMP_LT_I32: {
      ir.cmpIOp(OpDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::slt);
    } break;
    case eOpcode::S_CMP_LE_I32: {
      ir.cmpIOp(OpDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::sle);
    } break;
    case eOpcode::S_CMP_EQ_U32: {
      ir.cmpIOp(OpDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::eq);
    } break;
    case eOpcode::S_CMP_LG_U32: {
      ir.cmpIOp(OpDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_CMP_GT_U32: {
      ir.cmpIOp(OpDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::ugt);
    } break;
    case eOpcode::S_CMP_GE_U32: {
      ir.cmpIOp(OpDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::uge);
    } break;
    case eOpcode::S_CMP_LT_U32: {
      ir.cmpIOp(OpDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::ult);
    } break;
    case eOpcode::S_CMP_LE_U32: {
      ir.cmpIOp(OpDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::ule);
    } break;
    case eOpcode::S_BITCMP0_B32: {
      ir.bitCmpOp(OpDst(eOperandKind::SCC()), OpSrc(src0.kind, true, false), ir::OperandType::i32(), src1);
    } break;
    case eOpcode::S_BITCMP1_B32: {
      ir.bitCmpOp(OpDst(eOperandKind::SCC()), src0, ir::OperandType::i32(), src1);
    } break;
    case eOpcode::S_BITCMP0_B64: {
      ir.bitCmpOp(OpDst(eOperandKind::SCC()), OpSrc(src0.kind, true, false), ir::OperandType::i64(), src1);
    } break;
    case eOpcode::S_BITCMP1_B64: {
      ir.bitCmpOp(OpDst(eOperandKind::SCC()), src0, ir::OperandType::i64(), src1);
    } break;
    case eOpcode::S_SETVSKIP: {
      ir.bitCmpOp(OpDst(eOperandKind::VSKIP()), src0, ir::OperandType::i32(), src1);
    } break;
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }
  return conv(op);
}
} // namespace compiler::frontend::translate