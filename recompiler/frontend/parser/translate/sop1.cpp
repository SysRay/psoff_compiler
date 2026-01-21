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

InstructionKind_t handleSop1(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;
  using namespace ir::dialect;

  auto       inst = SOP1(**pCode);
  auto const op   = (parser::eOpcode)(OPcodeStart_SOP1 + inst.template get<SOP1::Field::OP>());

  auto const sdst = createDst(eOperandKind((eOperandKind_t)inst.template get<SOP1::Field::SDST>()));
  auto       src0 = createSrc(eOperandKind((eOperandKind_t)inst.template get<SOP1::Field::SSRC0>()));

  if (eOperandKind(src0.kind).isLiteral()) {
    *pCode += 1;
    src0 = createSrc(ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_u64 = **pCode}, ir::OperandType::i32()));
  }

  *pCode += 1;

  switch (op) {
    case eOpcode::S_MOV_B32: {
      ctx.create<core::MoveOp>(sdst, src0, ir::OperandType::i32());
    } break;
    case eOpcode::S_MOV_B64: {
      ctx.create<core::MoveOp>(sdst, src0, ir::OperandType::i64());
    } break;
    case eOpcode::S_CMOV_B32: {
      ctx.create<core::SelectOp>(sdst, createSrc(eOperandKind::SCC()), src0, createSrc(eOperandKind::SCC()), ir::OperandType::i32());
    } break;
    case eOpcode::S_CMOV_B64: {
      ctx.create<core::SelectOp>(sdst, createSrc(eOperandKind::SCC()), src0, createSrc(eOperandKind::SCC()), ir::OperandType::i64());
    } break;
    case eOpcode::S_NOT_B32: {
      auto in0 = ctx.create<arith::NotOp>(createDst(), src0, ir::OperandType::i32());
      auto res = ctx.create<core::MoveOp>(sdst, in0, ir::OperandType::i32());

      ctx.create<arith::CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), arith::CmpIPredicate::ne);
    } break;
    case eOpcode::S_NOT_B64: {
      auto in0 = ctx.create<arith::NotOp>(createDst(), src0, ir::OperandType::i64());
      auto res = ctx.create<core::MoveOp>(sdst, in0, ir::OperandType::i64());

      ctx.create<arith::CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), arith::CmpIPredicate::ne);
    } break;
    case eOpcode::S_WQM_B32:
    case eOpcode::S_WQM_B64: {
      if (eOperandKind(sdst.kind).base() != eOperandKind::eBase::ExecLo || eOperandKind(src0.kind).base() != eOperandKind::eBase::ExecLo) {
        throw std::runtime_error(std::format("missing wqm {}", (uint16_t)eOperandKind(src0.kind).base()));
      }
    } break;
    case eOpcode::S_BREV_B32: {
      ctx.create<arith::BitReverseOp>(sdst, src0, ir::OperandType::i32());
    } break;
    case eOpcode::S_BREV_B64: {
      ctx.create<arith::BitReverseOp>(sdst, src0, ir::OperandType::i64());
    } break;
    case eOpcode::S_BCNT0_I32_B32: {
      auto in0 = ctx.create<arith::NotOp>(createDst(), src0, ir::OperandType::i32());
      auto res = ctx.create<arith::BitCountOp>(sdst, in0, ir::OperandType::i32());
      ctx.create<arith::CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), arith::CmpIPredicate::ne);
    } break;
    case eOpcode::S_BCNT0_I32_B64: {
      auto in0 = ctx.create<arith::NotOp>(createDst(), src0, ir::OperandType::i64());
      auto res = ctx.create<arith::BitCountOp>(sdst, in0, ir::OperandType::i64());
      ctx.create<arith::CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), arith::CmpIPredicate::ne);
    } break;
    case eOpcode::S_BCNT1_I32_B32: {
      auto res = ctx.create<arith::BitCountOp>(sdst, src0, ir::OperandType::i32());
      ctx.create<arith::CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), arith::CmpIPredicate::ne);
    } break;
    case eOpcode::S_BCNT1_I32_B64: {
      auto res = ctx.create<arith::BitCountOp>(sdst, src0, ir::OperandType::i64());
      ctx.create<arith::CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), arith::CmpIPredicate::ne);
    } break;
    case eOpcode::S_FF0_I32_B32: {
      auto in0 = ctx.create<arith::NotOp>(createDst(), src0, ir::OperandType::i32());
      ctx.create<arith::FindILsbOp>(sdst, in0, ir::OperandType::i32());
    } break;
    case eOpcode::S_FF0_I32_B64: {
      auto in0 = ctx.create<arith::NotOp>(createDst(), src0, ir::OperandType::i32());
      ctx.create<arith::FindILsbOp>(sdst, in0, ir::OperandType::i64());
    } break;
    case eOpcode::S_FF1_I32_B32: {
      ctx.create<arith::FindILsbOp>(sdst, src0, ir::OperandType::i32());
    } break;
    case eOpcode::S_FF1_I32_B64: {
      ctx.create<arith::FindILsbOp>(sdst, src0, ir::OperandType::i64());
    } break;
    case eOpcode::S_FLBIT_I32_B32: {
      ctx.create<arith::FindUMsbOp>(sdst, src0, ir::OperandType::i32());
    } break;
    case eOpcode::S_FLBIT_I32_B64: {
      ctx.create<arith::FindUMsbOp>(sdst, src0, ir::OperandType::i64());
    } break;
    case eOpcode::S_FLBIT_I32: {
      ctx.create<arith::FindSMsbOp>(sdst, src0, ir::OperandType::i32());
    } break;
    case eOpcode::S_FLBIT_I32_I64: {
      ctx.create<arith::FindSMsbOp>(sdst, src0, ir::OperandType::i64());
    } break;
    case eOpcode::S_SEXT_I32_I8: {
      ctx.create<arith::SignExtendOp>(sdst, src0, ir::OperandType::i8(), ir::OperandType::i32());
    } break;
    case eOpcode::S_SEXT_I32_I16: {
      ctx.create<arith::SignExtendOp>(sdst, src0, ir::OperandType::i16(), ir::OperandType::i32());
    } break;
    case eOpcode::S_BITSET0_B32: {
      ctx.create<arith::BitsetOp>(sdst, createSrc(eOperandKind(sdst.kind)), src0, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32());
    } break;
    case eOpcode::S_BITSET0_B64: {
      ctx.create<arith::BitsetOp>(sdst, createSrc(eOperandKind(sdst.kind)), src0, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64());
    } break;
    case eOpcode::S_BITSET1_B32: {
      ctx.create<arith::BitsetOp>(sdst, createSrc(eOperandKind(sdst.kind)), src0, createSrc(eOperandKind::createImm(1)), ir::OperandType::i32());
    } break;
    case eOpcode::S_BITSET1_B64: {
      ctx.create<arith::BitsetOp>(sdst, createSrc(eOperandKind(sdst.kind)), src0, createSrc(eOperandKind::createImm(1)), ir::OperandType::i64());
    } break;
    case eOpcode::S_GETPC_B64: {
      ctx.create<core::ConstantOp>(sdst, ir::ConstantValue {.value_u64 = pc}, ir::OperandType::i64());
    } break;
    case eOpcode::S_SETPC_B64: {
      ctx.create<core::JumpAbsOp>(src0);
    } break;
    case eOpcode::S_SWAPPC_B64: {
      ctx.create<core::ConstantOp>(sdst, ir::ConstantValue {.value_u64 = 4 + pc}, ir::OperandType::i64());

      auto const& shaderInput = ctx.builder.getShaderInput();
      if (shaderInput.getLogicalStage() == ShaderLogicalStage::Vertex) {
        assert(src0.kind.isSGPR());

        auto const srcId = eOperandKind(src0.kind).getSGPR();
        if (srcId >= shaderInput.userSGPRSize) {
          throw std::runtime_error("fetch shader: not in user data");
        }

        auto const fetchAddr    = (((pc_t)shaderInput.userSGPR[1 + srcId] << 32) | (pc_t)shaderInput.userSGPR[srcId]);
        auto       fetchMapping = ctx.builder.getHostMapping(fetchAddr);
        if (fetchMapping == nullptr) throw std::runtime_error("fetch shader: no addr");

        auto pCode   = (uint32_t const*)fetchMapping->host;
        auto curCode = pCode;

        while (true) {
          auto const pc      = fetchAddr + (frontend::parser::pc_t)curCode - (frontend::parser::pc_t)pCode;
          auto const fetchOp = (eOpcode)frontend::parser::parseInstruction(ctx, pc, &curCode);
          if (fetchOp == eOpcode::S_SETPC_B64) break;

          if (curCode >= (pCode + SPEC_FETCHSHADER_MAX_SIZE_DW)) throw std::runtime_error("fetch shader: reached max size");
        }

        fetchMapping->size_dw = curCode - pCode;

        //  remove setpc
        auto& instructions = ctx.instructions.access();
        while (true) {
          auto const& item    = instructions.back();
          bool const  isSetpc = item.dialect == ir::eDialect::kCore && item.kind == conv(core::eInstKind::JumpAbsOp);
          instructions.pop_back();
          if (isSetpc) break;
          if (instructions.empty()) throw std::runtime_error("fetch shader: couldn't find setpc");
        }
      } else {
        ctx.create<core::JumpAbsOp>(src0);
      }
    } break;
    // case eOpcode::S_RFE_B64: break; // Does not exist
    case eOpcode::S_AND_SAVEEXEC_B64: {
      auto saved = ctx.create<core::MoveOp>(sdst, createSrc(eOperandKind::EXEC()), ir::OperandType::i1());
      ctx.create<arith::BitAndOp>(createDst(eOperandKind::EXEC()), src0, saved, ir::OperandType::i1());
    } break;
    case eOpcode::S_OR_SAVEEXEC_B64: {
      auto saved = ctx.create<core::MoveOp>(sdst, createSrc(eOperandKind::EXEC()), ir::OperandType::i1());
      auto res   = ctx.create<arith::BitOrOp>(createDst(eOperandKind::EXEC()), src0, saved, ir::OperandType::i1());
      ctx.create<core::MoveOp>(sdst, res, ir::OperandType::i1());
    } break;
    case eOpcode::S_XOR_SAVEEXEC_B64: {
      auto saved = ctx.create<core::MoveOp>(sdst, createSrc(eOperandKind::EXEC()), ir::OperandType::i1());
      ctx.create<arith::BitXorOp>(createDst(eOperandKind::EXEC()), src0, saved, ir::OperandType::i1());
    } break;
    case eOpcode::S_ANDN2_SAVEEXEC_B64: {
      auto saved = ctx.create<core::MoveOp>(sdst, createSrc(eOperandKind::EXEC()), ir::OperandType::i1());
      auto in0   = ctx.create<arith::NotOp>(createDst(), saved, ir::OperandType::i1());
      ctx.create<arith::BitAndOp>(createDst(eOperandKind::EXEC()), src0, in0, ir::OperandType::i1());
    } break;
    case eOpcode::S_ORN2_SAVEEXEC_B64: {
      auto saved = ctx.create<core::MoveOp>(sdst, createSrc(eOperandKind::EXEC()), ir::OperandType::i1());
      auto in0   = ctx.create<arith::NotOp>(createDst(), saved, ir::OperandType::i1());
      ctx.create<arith::BitOrOp>(createDst(eOperandKind::EXEC()), src0, in0, ir::OperandType::i1());
    } break;
    case eOpcode::S_NAND_SAVEEXEC_B64: {
      auto saved = ctx.create<core::MoveOp>(sdst, createSrc(eOperandKind::EXEC()), ir::OperandType::i1());
      auto res   = ctx.create<arith::BitAndOp>(createDst(), src0, saved, ir::OperandType::i1());
      ctx.create<arith::NotOp>(createDst(eOperandKind::EXEC()), res, ir::OperandType::i1());
    } break;
    case eOpcode::S_NOR_SAVEEXEC_B64: {
      auto saved = ctx.create<core::MoveOp>(sdst, createSrc(eOperandKind::EXEC()), ir::OperandType::i1());
      auto res   = ctx.create<arith::BitOrOp>(createDst(), src0, saved, ir::OperandType::i1());
      ctx.create<arith::NotOp>(createDst(eOperandKind::EXEC()), res, ir::OperandType::i1());
    } break;
    case eOpcode::S_XNOR_SAVEEXEC_B64: {
      auto saved = ctx.create<core::MoveOp>(sdst, createSrc(eOperandKind::EXEC()), ir::OperandType::i1());
      auto res   = ctx.create<arith::BitXorOp>(createDst(), src0, saved, ir::OperandType::i1());
      ctx.create<arith::NotOp>(createDst(eOperandKind::EXEC()), res, ir::OperandType::i1());
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
      ctx.create<arith::AbsoluteOp>(sdst, src0, ir::OperandType::i32());
    } break;
    // case eOpcode::S_MOV_FED_B32: break; // Does not exist
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }

  return conv(op);
}
} // namespace compiler::frontend::translate