#include "frontend/ir_types.h"
#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "../instruction_builder.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {

InstructionKind_t handleSop1(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = SOP1(**pCode);
  auto const op   = (parser::eOpcode)(OPcodeStart_SOP1 + inst.template get<SOP1::Field::OP>());

  auto const sdst = OpDst(eOperandKind((eOperandKind_t)inst.template get<SOP1::Field::SDST>()));
  auto const src0 = OpSrc(eOperandKind((eOperandKind_t)inst.template get<SOP1::Field::SSRC0>()));

  if (src0.kind.isLiteral()) {
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
      builder.createInstruction(create::selectOp(sdst, OpSrc(eOperandKind::SCC()), src0, OpSrc(eOperandKind::SCC()), ir::OperandType::i32()));
    } break;
    case eOpcode::S_CMOV_B64: {
      builder.createInstruction(create::selectOp(sdst, OpSrc(eOperandKind::SCC()), src0, OpSrc(eOperandKind::SCC()), ir::OperandType::i64()));
    } break;
    case eOpcode::S_NOT_B32: {
      builder.createInstruction(create::moveOp(sdst, OpSrc(src0.kind, true, false), ir::OperandType::i32()));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), OpSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_NOT_B64: {
      builder.createInstruction(create::moveOp(sdst, OpSrc(src0.kind, true, false), ir::OperandType::i64()));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), OpSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_WQM_B32:
    case eOpcode::S_WQM_B64: {
      if (sdst.kind.base() != eOperandKind::eBase::ExecLo || src0.kind.base() != eOperandKind::eBase::ExecLo) {
        throw std::runtime_error(std::format("missing wqm {}", (uint16_t)src0.kind.base()));
      }
    } break;
    case eOpcode::S_BREV_B32: {
      builder.createInstruction(create::bitReverseOp(sdst, src0, ir::OperandType::i32()));
    } break;
    case eOpcode::S_BREV_B64: {
      builder.createInstruction(create::bitReverseOp(sdst, src0, ir::OperandType::i64()));
    } break;
    case eOpcode::S_BCNT0_I32_B32: {
      builder.createInstruction(create::bitCountOp(sdst, OpSrc(src0.kind, true, false), ir::OperandType::i32()));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), OpSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_BCNT0_I32_B64: {
      builder.createInstruction(create::bitCountOp(sdst, OpSrc(src0.kind, true, false), ir::OperandType::i64()));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), OpSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_BCNT1_I32_B32: {
      builder.createInstruction(create::bitCountOp(sdst, src0, ir::OperandType::i32()));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), OpSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_BCNT1_I32_B64: {
      builder.createInstruction(create::bitCountOp(sdst, src0, ir::OperandType::i64()));
      builder.createInstruction(
          create::cmpIOp(OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), OpSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_FF0_I32_B32: {
      builder.createInstruction(create::findILsbOp(sdst, OpSrc(src0.kind, true, false), ir::OperandType::i32()));
    } break;
    case eOpcode::S_FF0_I32_B64: {
      builder.createInstruction(create::findILsbOp(sdst, OpSrc(src0.kind, true, false), ir::OperandType::i64()));
    } break;
    case eOpcode::S_FF1_I32_B32: {
      builder.createInstruction(create::findILsbOp(sdst, src0, ir::OperandType::i32()));
    } break;
    case eOpcode::S_FF1_I32_B64: {
      builder.createInstruction(create::findILsbOp(sdst, src0, ir::OperandType::i64()));
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
      builder.createInstruction(create::bitsetOp(sdst, OpSrc(sdst.kind), src0, OpSrc(eOperandKind::createImm(0)), ir::OperandType::i32()));
    } break;
    case eOpcode::S_BITSET0_B64: {
      builder.createInstruction(create::bitsetOp(sdst, OpSrc(sdst.kind), src0, OpSrc(eOperandKind::createImm(0)), ir::OperandType::i64()));
    } break;
    case eOpcode::S_BITSET1_B32: {
      builder.createInstruction(create::bitsetOp(sdst, OpSrc(sdst.kind), src0, OpSrc(OpSrc(eOperandKind::createImm(1))), ir::OperandType::i32()));
    } break;
    case eOpcode::S_BITSET1_B64: {
      builder.createInstruction(create::bitsetOp(sdst, OpSrc(sdst.kind), src0, OpSrc(OpSrc(eOperandKind::createImm(1))), ir::OperandType::i64()));
    } break;
    case eOpcode::S_GETPC_B64: {
      builder.createInstruction(create::constantOp(sdst, 4 + pc, ir::OperandType::i64()));
    } break;
    case eOpcode::S_SETPC_B64: {
      builder.createInstruction(create::jumpAbsOp(src0));
    } break;
    case eOpcode::S_SWAPPC_B64: {
      builder.createInstruction(create::constantOp(sdst, 4 + pc, ir::OperandType::i64())); // Save current pc

      if (builder.getShaderInput().getLogicalStage() == ShaderLogicalStage::Vertex) {
        assert(src0.kind.isSGPR());

        auto const srcId = src0.kind.getSGPR();
        if (srcId >= builder.getShaderInput().userSGPRSize) {
          throw std::runtime_error("fetch shader: not in user data");
        }

        auto const fetchAddr    = (((pc_t)builder.getShaderInput().userSGPR[1 + srcId] << 32) | (pc_t)builder.getShaderInput().userSGPR[srcId]);
        auto       fetchMapping = builder.getHostMapping(fetchAddr);
        if (fetchMapping == nullptr) throw std::runtime_error("fetch shader: no addr");

        auto pCode   = (uint32_t const*)fetchMapping->host;
        auto curCode = pCode;

        while (true) {
          auto const pc      = fetchAddr + (frontend::parser::pc_t)curCode - (frontend::parser::pc_t)pCode;
          auto const fetchOp = (eOpcode)frontend::parser::parseInstruction(builder, pc, &curCode);
          if (fetchOp == eOpcode::S_SETPC_B64) break;

          if (curCode >= (pCode + SPEC_FETCHSHADER_MAX_SIZE_DW)) throw std::runtime_error("fetch shader: reached max size");
        }

        fetchMapping->size_dw = curCode - pCode;

        //  remove setpc
        auto& instructions = builder.getInstructions();
        while (true) {
          auto const& item    = instructions.back();
          bool const  isSetpc = item.kind == conv(ir::eInstKind::JumpAbsOp);
          instructions.pop_back();
          if (isSetpc) break;
          if (instructions.empty()) throw std::runtime_error("fetch shader: couldn't find setpc");
        }
      } else {
        builder.createInstruction(create::jumpAbsOp(src0));
      }
    } break;
    // case eOpcode::S_RFE_B64: break; // Does not exist
    case eOpcode::S_AND_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(sdst, OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
      builder.createInstruction(create::bitAndOp(OpDst(eOperandKind::EXEC()), src0, OpSrc(sdst.kind), ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(sdst, OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
    } break;
    case eOpcode::S_OR_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(sdst, OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
      builder.createInstruction(create::bitOrOp(OpDst(eOperandKind::EXEC()), src0, OpSrc(sdst.kind), ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(sdst, OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
    } break;
    case eOpcode::S_XOR_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(sdst, OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
      builder.createInstruction(create::bitXorOp(OpDst(eOperandKind::EXEC()), src0, OpSrc(sdst.kind), ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(sdst, OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
    } break;
    case eOpcode::S_ANDN2_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(sdst, OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
      builder.createInstruction(create::bitAndOp(OpDst(eOperandKind::EXEC()), src0, OpSrc(sdst.kind, true, false), ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(sdst, OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
    } break;
    case eOpcode::S_ORN2_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(sdst, OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
      builder.createInstruction(create::bitOrOp(OpDst(eOperandKind::EXEC()), src0, OpSrc(sdst.kind, true, false), ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(sdst, OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
    } break;
    case eOpcode::S_NAND_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(sdst, OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
      builder.createInstruction(create::bitAndOp(OpDst(eOperandKind::EXEC(), 0, false, true), src0, OpSrc(sdst.kind), ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(sdst, OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
    } break;
    case eOpcode::S_NOR_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(sdst, OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
      builder.createInstruction(create::bitOrOp(OpDst(eOperandKind::EXEC(), 0, false, true), src0, OpSrc(sdst.kind), ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(sdst, OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
    } break;
    case eOpcode::S_XNOR_SAVEEXEC_B64: {
      builder.createInstruction(create::moveOp(sdst, OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
      builder.createInstruction(create::bitXorOp(OpDst(eOperandKind::EXEC(), 0, false, true), src0, OpSrc(sdst.kind), ir::OperandType::i1()));
      builder.createInstruction(create::moveOp(sdst, OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
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
      builder.createInstruction(create::moveOp(sdst, OpSrc(src0.kind, false, true), ir::OperandType::i32()));
    } break;
    // case eOpcode::S_MOV_FED_B32: break; // Does not exist
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }

  return conv(op);
}
} // namespace compiler::frontend::translate