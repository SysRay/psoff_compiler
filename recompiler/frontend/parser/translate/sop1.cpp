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

bool handleSop1(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = SOP1(**pCode);
  auto const op   = (parser::eOpcode)(OPcodeStart_SOP1 +inst.template get<SOP1::Field::OP>());

  auto const sdst = (eOperandKind)inst.template get<SOP1::Field::SDST>();
  auto const src0 = (eOperandKind)inst.template get<SOP1::Field::SSRC0>();

  if (src0 == eOperandKind::Literal) {
    *pCode += 1;
    builder.createInstruction(create::literalOp(**pCode));
  }
  *pCode += 1;

  switch (op) {
    case eOpcode::S_MOV_B32: {
      builder.createInstruction(create::moveOp(sdst, src0, ir::OperandType::i32()));
    } break;
    case eOpcode::S_MOV_B64: {
      builder.createInstruction(create::moveOp(sdst, src0, ir::OperandType::i64()));
    } break;
    case eOpcode::S_CMOV_B32: {
      builder.createInstruction(create::selectOp(sdst, eOperandKind::Scc, src0, sdst, ir::OperandType::i32()));
    } break;
    case eOpcode::S_CMOV_B64: {
      builder.createInstruction(create::selectOp(sdst, eOperandKind::Scc, src0, sdst, ir::OperandType::i64()));
    } break;
    case eOpcode::S_NOT_B32: {
      builder.createInstruction(create::notOp(sdst, src0, ir::OperandType::i32()));
      builder.createInstruction(create::cmpIOp(eOperandKind::Scc, sdst, eOperandKind::ConstZero, ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_NOT_B64: {
      builder.createInstruction(create::notOp(sdst, src0, ir::OperandType::i64()));
      builder.createInstruction(create::cmpIOp(eOperandKind::Scc, sdst, eOperandKind::ConstZero, ir::OperandType::i64(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_WQM_B32:
    case eOpcode::S_WQM_B64: {
      if (sdst != eOperandKind::ExecLo || src0 != eOperandKind::ExecLo) {
        throw std::runtime_error(std::format("missing wqm {} {}", (uint16_t)sdst, (uint16_t)src0));
      }
    } break;
    case eOpcode::S_BREV_B32: {
      builder.createInstruction(create::bitReverseOp(sdst, src0, ir::OperandType::i32()));
    } break;
    case eOpcode::S_BREV_B64: {
      builder.createInstruction(create::bitReverseOp(sdst, src0, ir::OperandType::i64()));
    } break;
    case eOpcode::S_BCNT0_I32_B32: {
      builder.createInstruction(create::bitCountOp(sdst, src0, ir::OperandType::i32(), false));
      builder.createInstruction(create::cmpIOp(eOperandKind::Scc, sdst, eOperandKind::ConstZero, ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_BCNT0_I32_B64: {
      builder.createInstruction(create::bitCountOp(sdst, src0, ir::OperandType::i64(), false));
      builder.createInstruction(create::cmpIOp(eOperandKind::Scc, sdst, eOperandKind::ConstZero, ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_BCNT1_I32_B32: {
      builder.createInstruction(create::bitCountOp(sdst, src0, ir::OperandType::i32(), true));
      builder.createInstruction(create::cmpIOp(eOperandKind::Scc, sdst, eOperandKind::ConstZero, ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_BCNT1_I32_B64: {
      builder.createInstruction(create::bitCountOp(sdst, src0, ir::OperandType::i64(), true));
      builder.createInstruction(create::cmpIOp(eOperandKind::Scc, sdst, eOperandKind::ConstZero, ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_FF0_I32_B32: {
      builder.createInstruction(create::findILsbOp(sdst, src0, ir::OperandType::i32(), false));
    } break;
    case eOpcode::S_FF0_I32_B64: {
      builder.createInstruction(create::findILsbOp(sdst, src0, ir::OperandType::i64(), false));
    } break;
    case eOpcode::S_FF1_I32_B32: {
      builder.createInstruction(create::findILsbOp(sdst, src0, ir::OperandType::i32(), true));
    } break;
    case eOpcode::S_FF1_I32_B64: {
      builder.createInstruction(create::findILsbOp(sdst, src0, ir::OperandType::i64(), true));
    } break;
    case eOpcode::S_FLBIT_I32_B32: {
      builder.createInstruction(create::findUMsbOp(sdst, src0, ir::OperandType::i32()));
    } break;
    case eOpcode::S_FLBIT_I32_B64: {
      builder.createInstruction(create::findUMsbOp(sdst, src0, ir::OperandType::i64()));
    } break;
    case eOpcode::S_FLBIT_I32: {
      builder.createInstruction(create::findSMsbOp(sdst, src0, ir::OperandType::i32()));
    } break;
    case eOpcode::S_FLBIT_I32_I64: {
      builder.createInstruction(create::findSMsbOp(sdst, src0, ir::OperandType::i64()));
    } break;
    case eOpcode::S_SEXT_I32_I8: {
      builder.createInstruction(create::signExtendI32Op(sdst, src0, ir::OperandType::i8()));
    } break;
    case eOpcode::S_SEXT_I32_I16: {
      builder.createInstruction(create::signExtendI32Op(sdst, src0, ir::OperandType::i16()));
    } break;
    case eOpcode::S_BITSET0_B32: {
      builder.createInstruction(create::bitsetOp(sdst, sdst, src0, eOperandKind::ConstZero, ir::OperandType::i32()));
    } break;
    case eOpcode::S_BITSET0_B64: {
      builder.createInstruction(create::bitsetOp(sdst, sdst, src0, eOperandKind::ConstZero, ir::OperandType::i64()));
    } break;
    case eOpcode::S_BITSET1_B32: {
      builder.createInstruction(create::bitsetOp(sdst, sdst, src0, getUImm(1), ir::OperandType::i32()));
    } break;
    case eOpcode::S_BITSET1_B64: {
      builder.createInstruction(create::bitsetOp(sdst, sdst, src0, getUImm(1), ir::OperandType::i64()));
    } break;
    case eOpcode::S_GETPC_B64: {
      builder.createInstruction(create::constantOp(sdst, 4 + pc, ir::OperandType::i64()));
    } break;
    case eOpcode::S_SETPC_B64: {
      builder.createInstruction(create::jumpAbsOp(src0));
    } break;
    case eOpcode::S_SWAPPC_B64: {
      builder.createInstruction(create::constantOp(sdst, 4 + pc, ir::OperandType::i64()));
      builder.createInstruction(create::jumpAbsOp(src0));
    } break;
    // case eOpcode::S_RFE_B64: break; // Does not exist
    case eOpcode::S_AND_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(sdst, eOperandKind::ExecLo, ir::OperandType::i1()));
      builder.createInstruction(create::bitAndOp(eOperandKind::ExecLo, src0, sdst, ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(eOperandKind::Scc, eOperandKind::ExecLo, ir::OperandType::i1()));
    } break;
    case eOpcode::S_OR_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(sdst, eOperandKind::ExecLo, ir::OperandType::i1()));
      builder.createInstruction(create::bitOrOp(eOperandKind::ExecLo, src0, sdst, ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(eOperandKind::Scc, eOperandKind::ExecLo, ir::OperandType::i1()));
    } break;
    case eOpcode::S_XOR_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(sdst, eOperandKind::ExecLo, ir::OperandType::i1()));
      builder.createInstruction(create::bitXorOp(eOperandKind::ExecLo, src0, sdst, ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(eOperandKind::Scc, eOperandKind::ExecLo, ir::OperandType::i1()));
    } break;
    case eOpcode::S_ANDN2_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(sdst, eOperandKind::ExecLo, ir::OperandType::i1()));
      builder.createInstruction(create::bitAndNOp(eOperandKind::ExecLo, src0, sdst, ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(eOperandKind::Scc, eOperandKind::ExecLo, ir::OperandType::i1()));
    } break;
    case eOpcode::S_ORN2_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(sdst, eOperandKind::ExecLo, ir::OperandType::i1()));
      builder.createInstruction(create::bitOrNOp(eOperandKind::ExecLo, src0, sdst, ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(eOperandKind::Scc, eOperandKind::ExecLo, ir::OperandType::i1()));
    } break;
    case eOpcode::S_NAND_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(sdst, eOperandKind::ExecLo, ir::OperandType::i1()));
      builder.createInstruction(create::bitNandOp(eOperandKind::ExecLo, src0, sdst, ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(eOperandKind::Scc, eOperandKind::ExecLo, ir::OperandType::i1()));
    } break;
    case eOpcode::S_NOR_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(sdst, eOperandKind::ExecLo, ir::OperandType::i1()));
      builder.createInstruction(create::bitNorOp(eOperandKind::ExecLo, src0, sdst, ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(eOperandKind::Scc, eOperandKind::ExecLo, ir::OperandType::i1()));
    } break;
    case eOpcode::S_XNOR_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(sdst, eOperandKind::ExecLo, ir::OperandType::i1()));
      builder.createInstruction(create::bitXnorOp(eOperandKind::ExecLo, src0, sdst, ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(eOperandKind::Scc, eOperandKind::ExecLo, ir::OperandType::i1()));
    } break;
      // case eOpcode::S_QUADMASK_B
      // 32: break; // todo,  might be same as wqm
      // case eOpcode::S_QUADMASK_B64: break; // todo, might be same as wqm
    // case eOpcode::S_MOVRELS_B32: {} break; // todo
    // case eOpcode::S_MOVRELS_B64: {} break; // todo
    // case eOpcode::S_MOVRELD_B32: {} break; // todo
    // case eOpcode::S_MOVRELD_B64: {} break; // todo
    // case eOpcode::S_CBRANCH_JOIN: {} break; // todo, make a block falltrough or handle data?
    //  case eOpcode::S_MOV_REGRD_B32: break; // Does not exist
    case eOpcode::S_ABS_I32: {
      builder.createInstruction(create::absOp(sdst, src0, ir::OperandType::i32()));
    } break;
    // case eOpcode::S_MOV_FED_B32: break; // Does not exist
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }

  return true;
}
} // namespace compiler::frontend::translate