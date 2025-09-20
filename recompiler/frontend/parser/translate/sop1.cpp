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
  auto const op   = (parser::eOpcode)(OPcodeStart_SOP1 + inst.template get<SOP1::Field::OP>());

  auto const sdst = (eOperandKind)inst.template get<SOP1::Field::SDST>();
  auto const src0 = (eOperandKind)inst.template get<SOP1::Field::SSRC0>();

  if (src0 == eOperandKind::Literal) {
    *pCode += 1;
    builder.createInstruction(create::literalOp(**pCode));
  }
  *pCode += 1;

  switch (op) {
    case eOpcode::S_MOV_B32: {
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(src0), ir::OperandType::i32()));
    } break;
    case eOpcode::S_MOV_B64: {
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(src0), ir::OperandType::i64()));
    } break;
    case eOpcode::S_CMOV_B32: {
      builder.createInstruction(create::selectOp(OpDst(sdst), OpSrc(eOperandKind::Scc), OpSrc(src0), OpSrc(eOperandKind::Scc), ir::OperandType::i32()));
    } break;
    case eOpcode::S_CMOV_B64: {
      builder.createInstruction(create::selectOp(OpDst(sdst), OpSrc(eOperandKind::Scc), OpSrc(src0), OpSrc(eOperandKind::Scc), ir::OperandType::i64()));
    } break;
    case eOpcode::S_NOT_B32: {
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(src0, true, false), ir::OperandType::i32()));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(eOperandKind::ConstZero), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_NOT_B64: {
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(src0, true, false), ir::OperandType::i64()));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(eOperandKind::ConstZero), ir::OperandType::i64(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_WQM_B32:
    case eOpcode::S_WQM_B64: {
      if (sdst != eOperandKind::ExecLo || src0 != eOperandKind::ExecLo) {
        throw std::runtime_error(std::format("missing wqm {} {}", (uint16_t)eOperandKind::Scc, (uint16_t)src0));
      }
    } break;
    case eOpcode::S_BREV_B32: {
      builder.createInstruction(create::bitReverseOp(OpDst(sdst), OpSrc(src0), ir::OperandType::i32()));
    } break;
    case eOpcode::S_BREV_B64: {
      builder.createInstruction(create::bitReverseOp(OpDst(sdst), OpSrc(src0), ir::OperandType::i64()));
    } break;
    case eOpcode::S_BCNT0_I32_B32: {
      builder.createInstruction(create::bitCountOp(OpDst(sdst), OpSrc(src0, true, false), ir::OperandType::i32()));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(eOperandKind::ConstZero), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_BCNT0_I32_B64: {
      builder.createInstruction(create::bitCountOp(OpDst(sdst), OpSrc(src0, true, false), ir::OperandType::i64()));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(eOperandKind::ConstZero), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_BCNT1_I32_B32: {
      builder.createInstruction(create::bitCountOp(OpDst(sdst), OpSrc(src0), ir::OperandType::i32()));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(eOperandKind::ConstZero), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_BCNT1_I32_B64: {
      builder.createInstruction(create::bitCountOp(OpDst(sdst), OpSrc(src0), ir::OperandType::i64()));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(eOperandKind::ConstZero), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_FF0_I32_B32: {
      builder.createInstruction(create::findILsbOp(OpDst(sdst), OpSrc(src0, true, false), ir::OperandType::i32()));
    } break;
    case eOpcode::S_FF0_I32_B64: {
      builder.createInstruction(create::findILsbOp(OpDst(sdst), OpSrc(src0, true, false), ir::OperandType::i64()));
    } break;
    case eOpcode::S_FF1_I32_B32: {
      builder.createInstruction(create::findILsbOp(OpDst(sdst), OpSrc(src0), ir::OperandType::i32()));
    } break;
    case eOpcode::S_FF1_I32_B64: {
      builder.createInstruction(create::findILsbOp(OpDst(sdst), OpSrc(src0), ir::OperandType::i64()));
    } break;
    case eOpcode::S_FLBIT_I32_B32: {
      builder.createInstruction(create::findUMsbOp(OpDst(sdst), OpSrc(src0), ir::OperandType::i32()));
    } break;
    case eOpcode::S_FLBIT_I32_B64: {
      builder.createInstruction(create::findUMsbOp(OpDst(sdst), OpSrc(src0), ir::OperandType::i64()));
    } break;
    case eOpcode::S_FLBIT_I32: {
      builder.createInstruction(create::findSMsbOp(OpDst(sdst), OpSrc(src0), ir::OperandType::i32()));
    } break;
    case eOpcode::S_FLBIT_I32_I64: {
      builder.createInstruction(create::findSMsbOp(OpDst(sdst), OpSrc(src0), ir::OperandType::i64()));
    } break;
    case eOpcode::S_SEXT_I32_I8: {
      builder.createInstruction(create::signExtendI32Op(OpDst(sdst), OpSrc(src0), ir::OperandType::i8()));
    } break;
    case eOpcode::S_SEXT_I32_I16: {
      builder.createInstruction(create::signExtendI32Op(OpDst(sdst), OpSrc(src0), ir::OperandType::i16()));
    } break;
    case eOpcode::S_BITSET0_B32: {
      builder.createInstruction(create::bitsetOp(OpDst(sdst), OpSrc(sdst), OpSrc(src0), OpSrc(eOperandKind::ConstZero), ir::OperandType::i32()));
    } break;
    case eOpcode::S_BITSET0_B64: {
      builder.createInstruction(create::bitsetOp(OpDst(sdst), OpSrc(sdst), OpSrc(src0), OpSrc(eOperandKind::ConstZero), ir::OperandType::i64()));
    } break;
    case eOpcode::S_BITSET1_B32: {
      builder.createInstruction(create::bitsetOp(OpDst(sdst), OpSrc(sdst), OpSrc(src0), OpSrc(getUImm(1)), ir::OperandType::i32()));
    } break;
    case eOpcode::S_BITSET1_B64: {
      builder.createInstruction(create::bitsetOp(OpDst(sdst), OpSrc(sdst), OpSrc(src0), OpSrc(getUImm(1)), ir::OperandType::i64()));
    } break;
    case eOpcode::S_GETPC_B64: {
      builder.createInstruction(create::constantOp(OpDst(sdst), 4 + pc, ir::OperandType::i64()));
    } break;
    case eOpcode::S_SETPC_B64: {
      builder.createInstruction(create::jumpAbsOp(OpSrc(src0)));
    } break;
    case eOpcode::S_SWAPPC_B64: {
      builder.createInstruction(create::constantOp(OpDst(sdst), 4 + pc, ir::OperandType::i64()));
      builder.createInstruction(create::jumpAbsOp(OpSrc(src0)));
    } break;
    // case eOpcode::S_RFE_B64: break; // Does not exist
    case eOpcode::S_AND_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(eOperandKind::ExecLo), ir::OperandType::i1()));
      builder.createInstruction(create::bitAndOp(OpDst(eOperandKind::ExecLo), OpSrc(src0), OpSrc(sdst), ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(eOperandKind::ExecLo), ir::OperandType::i1()));
    } break;
    case eOpcode::S_OR_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(eOperandKind::ExecLo), ir::OperandType::i1()));
      builder.createInstruction(create::bitOrOp(OpDst(eOperandKind::ExecLo), OpSrc(src0), OpSrc(sdst), ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(eOperandKind::ExecLo), ir::OperandType::i1()));
    } break;
    case eOpcode::S_XOR_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(eOperandKind::ExecLo), ir::OperandType::i1()));
      builder.createInstruction(create::bitXorOp(OpDst(eOperandKind::ExecLo), OpSrc(src0), OpSrc(sdst), ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(eOperandKind::ExecLo), ir::OperandType::i1()));
    } break;
    case eOpcode::S_ANDN2_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(eOperandKind::ExecLo), ir::OperandType::i1()));
      builder.createInstruction(create::bitAndOp(OpDst(eOperandKind::ExecLo), OpSrc(src0), OpSrc(sdst, true, false), ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(eOperandKind::ExecLo), ir::OperandType::i1()));
    } break;
    case eOpcode::S_ORN2_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(eOperandKind::ExecLo), ir::OperandType::i1()));
      builder.createInstruction(create::bitOrOp(OpDst(eOperandKind::ExecLo), OpSrc(src0), OpSrc(sdst, true, false), ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(eOperandKind::ExecLo), ir::OperandType::i1()));
    } break;
    case eOpcode::S_NAND_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(eOperandKind::ExecLo), ir::OperandType::i1()));
      builder.createInstruction(create::bitAndOp(OpDst(eOperandKind::ExecLo, 0, false, true), OpSrc(src0), OpSrc(sdst), ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(eOperandKind::ExecLo), ir::OperandType::i1()));
    } break;
    case eOpcode::S_NOR_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(eOperandKind::ExecLo), ir::OperandType::i1()));
      builder.createInstruction(create::bitOrOp(OpDst(eOperandKind::ExecLo, 0, false, true), OpSrc(src0), OpSrc(sdst), ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(eOperandKind::ExecLo), ir::OperandType::i1()));
    } break;
    case eOpcode::S_XNOR_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(eOperandKind::ExecLo), ir::OperandType::i1()));
      builder.createInstruction(create::bitXorOp(OpDst(eOperandKind::ExecLo, 0, false, true), OpSrc(src0), OpSrc(sdst), ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(eOperandKind::ExecLo), ir::OperandType::i1()));
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
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(src0, false, true), ir::OperandType::i32()));
    } break;
    // case eOpcode::S_MOV_FED_B32: break; // Does not exist
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }

  return true;
}
} // namespace compiler::frontend::translate