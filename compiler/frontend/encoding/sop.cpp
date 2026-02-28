#include "../debug_strings.h"
#include "../gfx/encoding_types.h"
#include "../operations.h"
#include "../parser.h"
#include "compiler_ctx.h"
#include "opcodes_table.h"

#include <format>
#include <stdexcept>

// mlir
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace compiler::frontend {

uint8_t Parser::handleSop1(CodeBlock& cb, pc_t pc, uint32_t const* pCode) {
  auto       inst = SOP1(*pCode);
  auto const op   = (eOpcode)(OPcodeStart_SOP1 + inst.template get<SOP1::Field::OP>());

  auto const sdst = OpDst(eOperandKind((eOperandKind_t)inst.template get<SOP1::Field::SDST>()));
  auto       src0 = OpSrc(eOperandKind((eOperandKind_t)inst.template get<SOP1::Field::SSRC0>()));

  switch (op) {
    case eOpcode::S_MOV_B32: {
      parse<op::MoveOp>(sdst, src0, types().i32());
    } break;
    case eOpcode::S_MOV_B64: {
      parse<op::MoveOp>(sdst, src0, types().i64());
    } break;
    case eOpcode::S_CMOV_B32: {
      parse<op::CMoveOp>(sdst, OpSrc(eOperandKind::SCC()), src0, OpSrc(sdst.kind), types().i32());
    } break;
    case eOpcode::S_CMOV_B64: {
      parse<op::CMoveOp>(sdst, OpSrc(eOperandKind::SCC()), src0, OpSrc(sdst.kind), types().i64());
    } break;
    case eOpcode::S_NOT_B32: {
      parse<op::NotOp>(sdst, OpDst(eOperandKind::SCC()), src0, types().i32());
    } break;
    case eOpcode::S_NOT_B64: {
      parse<op::NotOp>(sdst, OpDst(eOperandKind::SCC()), src0, types().i64());
    } break;
    case eOpcode::S_WQM_B32:
    case eOpcode::S_WQM_B64: {
      if (eOperandKind(sdst.kind).value() != eOperandKind::eBase::ExecLo) {
        throw std::runtime_error(std::format("missing wqm {}", (uint16_t)eOperandKind(src0.kind).value()));
      }
    } break;
    case eOpcode::S_BREV_B32: {
      parse<op::BrevOp>(sdst, src0, types().i32());
    } break;
    case eOpcode::S_BREV_B64: {
      parse<op::BrevOp>(sdst, src0, types().i64());
    } break;
    case eOpcode::S_BCNT0_I32_B32: {
      parse<op::BitCountOp>(sdst, OpDst(eOperandKind::SCC()), OpSrc(src0.kind, {OpFlags::eNot}), types().i32());
    } break;
    case eOpcode::S_BCNT0_I32_B64: {
      parse<op::BitCountOp>(sdst, OpDst(eOperandKind::SCC()), OpSrc(src0.kind, {OpFlags::eNot}), types().i64());
    } break;
    case eOpcode::S_BCNT1_I32_B32: {
      parse<op::BitCountOp>(sdst, OpDst(eOperandKind::SCC()), src0, types().i32());
    } break;
    case eOpcode::S_BCNT1_I32_B64: {
      parse<op::BitCountOp>(sdst, OpDst(eOperandKind::SCC()), src0, types().i64());
    } break;
    case eOpcode::S_FF0_I32_B32: {
      parse<op::FindFirstLsbBitOp>(sdst, OpSrc(src0.kind, {OpFlags::eNot}), types().i32());
    } break;
    case eOpcode::S_FF0_I32_B64: {
      parse<op::FindFirstLsbBitOp>(sdst, OpSrc(src0.kind, {OpFlags::eNot}), types().i64());
    } break;
    case eOpcode::S_FF1_I32_B32: {
      parse<op::FindFirstLsbBitOp>(sdst, src0, types().i32());
    } break;
    case eOpcode::S_FF1_I32_B64: {
      parse<op::FindFirstLsbBitOp>(sdst, src0, types().i64());
    } break;
    case eOpcode::S_FLBIT_I32_B32: {
      parse<op::FindFirstUMsbBitOp>(sdst, src0, types().i32());
    } break;
    case eOpcode::S_FLBIT_I32_B64: {
      parse<op::FindFirstUMsbBitOp>(sdst, src0, types().i64());
    } break;
    case eOpcode::S_FLBIT_I32: {
      parse<op::FindFirstSMsbBitOp>(sdst, src0, types().i32());
    } break;
    case eOpcode::S_FLBIT_I32_I64: {
      parse<op::FindFirstSMsbBitOp>(sdst, src0, types().i64());
    } break;
    case eOpcode::S_SEXT_I32_I8: {
      parse<op::SignExtOp>(sdst, types().i32(), src0, types().i8());
    } break;
    case eOpcode::S_SEXT_I32_I16: {
      parse<op::SignExtOp>(sdst, types().i32(), src0, types().i16());
    } break;
    case eOpcode::S_BITSET0_B32: {
      parse<op::BitClearOp>(sdst, src0, types().i32());
    } break;
    case eOpcode::S_BITSET0_B64: {
      parse<op::BitClearOp>(sdst, src0, types().i64());
    } break;
    case eOpcode::S_BITSET1_B32: {
      parse<op::BitSetOp>(sdst, src0, types().i32());
    } break;
    case eOpcode::S_BITSET1_B64: {
      parse<op::BitSetOp>(sdst, src0, types().i64());
    } break;
    case eOpcode::S_GETPC_B64: {
      parse<op::MoveOp>(sdst, (uint64_t)(4 + pc));
    } break;
    case eOpcode::S_SETPC_B64: {
      cb.pc_end = pc;
      parse<op::BranchOp>(src0);
    } break;
    case eOpcode::S_SWAPPC_B64: { // todo move out
      parse<op::MoveOp>(sdst, (uint64_t)(4 + pc));

      auto const& shaderInput = _compilerCtx.getShaderInput();
      if (shaderInput.getLogicalStage() == ShaderLogicalStage::Vertex && src0.kind.isSGPR() && src0.kind.getSGPR() < shaderInput.userSGPRSize) {
        // todo fetch shader
      } else {
        cb.pc_end = pc;
        parse<op::BranchOp>(src0);
        auto target0 = getOrCreateBlock(sizeof(uint32_t) + pc, cb.mlirBlock->getParent());
      }
    } break;
    // case eOpcode::S_RFE_B64: break; // Does not exist
    case eOpcode::S_AND_SAVEEXEC_B64: {
      parse<op::SaveExecOp>(sdst, op::SaveExecOp::BitOp::eAND, src0, OpSrc(eOperandKind::EXEC()), types().i64());
    } break;
    case eOpcode::S_OR_SAVEEXEC_B64: {
      parse<op::SaveExecOp>(sdst, op::SaveExecOp::BitOp::eOR, src0, OpSrc(eOperandKind::EXEC()), types().i64());
    } break;
    case eOpcode::S_XOR_SAVEEXEC_B64: {
      parse<op::SaveExecOp>(sdst, op::SaveExecOp::BitOp::eXOR, src0, OpSrc(eOperandKind::EXEC()), types().i64());
    } break;
    case eOpcode::S_ANDN2_SAVEEXEC_B64: {
      parse<op::SaveExecOp>(sdst, op::SaveExecOp::BitOp::eAND, src0, OpSrc(eOperandKind::EXEC(), {OpFlags::eNot}), types().i64());
    } break;
    case eOpcode::S_ORN2_SAVEEXEC_B64: {
      parse<op::SaveExecOp>(sdst, op::SaveExecOp::BitOp::eAND, src0, OpSrc(eOperandKind::EXEC(), {OpFlags::eNot}), types().i64());
    } break;
    case eOpcode::S_NAND_SAVEEXEC_B64: {
      parse<op::SaveExecOp>(OpDst(sdst.kind, {OpFlags::eNot}), op::SaveExecOp::BitOp::eAND, src0, OpSrc(eOperandKind::EXEC()), types().i64());
    } break;
    case eOpcode::S_NOR_SAVEEXEC_B64: {
      parse<op::SaveExecOp>(OpDst(sdst.kind, {OpFlags::eNot}), op::SaveExecOp::BitOp::eOR, src0, OpSrc(eOperandKind::EXEC()), types().i64());
    } break;
    case eOpcode::S_XNOR_SAVEEXEC_B64: {
      parse<op::SaveExecOp>(OpDst(sdst.kind, {OpFlags::eNot}), op::SaveExecOp::BitOp::eXOR, src0, OpSrc(eOperandKind::EXEC()), types().i64());
    } break;
      // case eOpcode::S_QUADMASK_B32: break; // todo,  might be same as wqm
      // case eOpcode::S_QUADMASK_B64: break; // todo, might be same as wqm
    // case eOpcode::S_MOVRELS_B32: {} break; // todo
    // case eOpcode::S_MOVRELS_B64: {} break; // todo
    // case eOpcode::S_MOVRELD_B32: {} break; // todo
    // case eOpcode::S_MOVRELD_B64: {} break; // todo
    // case eOpcode::S_CBRANCH_JOIN: {} break; // todo, make a block falltrough or handle data?
    //  case eOpcode::S_MOV_REGRD_B32: break; // Does not exist
    case eOpcode::S_ABS_I32: {
      parse<op::AbsIOp>(sdst, src0, types().i16());
    } break;
    // case eOpcode::S_MOV_FED_B32: break; // Does not exist
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }

  if (src0.kind.isLiteral()) {
    return sizeof(uint64_t);
  }
  return sizeof(uint32_t);
}

uint8_t Parser::handleSop2(CodeBlock& cb, pc_t pc, uint32_t const* pCode) {
  auto       inst = SOP2(*pCode);
  auto const op   = (eOpcode)(OPcodeStart_SOP2 + inst.template get<SOP2::Field::OP>());

  auto const sdst = OpDst(eOperandKind((eOperandKind_t)inst.template get<SOP2::Field::SDST>()));
  auto       src0 = OpSrc(eOperandKind((eOperandKind_t)inst.template get<SOP2::Field::SSRC0>()));
  auto       src1 = OpSrc(eOperandKind((eOperandKind_t)inst.template get<SOP2::Field::SSRC1>()));

  switch (op) {
    case eOpcode::S_ADD_U32: {
      parse<op::AddUIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_SUB_U32: {
      parse<op::SubUIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_ADD_I32: {
      parse<op::AddSIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_SUB_I32: {
      parse<op::SubSIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_ADDC_U32: {
      parse<op::AddUIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, OpSrc(eOperandKind::SCC()), types().i32());
    } break;
    case eOpcode::S_SUBB_U32: {
      parse<op::SubUIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, OpSrc(eOperandKind::SCC()), types().i32());
    } break;
    case eOpcode::S_MIN_I32: {
      parse<op::MinSIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_MIN_U32: {
      parse<op::MinUIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_MAX_I32: {
      parse<op::MaxSIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_MAX_U32: {
      parse<op::MaxUIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_CSELECT_B32: {
      parse<op::CMoveOp>(sdst, OpSrc(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_CSELECT_B64: {
      parse<op::CMoveOp>(sdst, OpSrc(eOperandKind::SCC()), src0, src1, types().i64());
    } break;
    case eOpcode::S_AND_B32: {
      parse<op::AndIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_AND_B64: {
      parse<op::AndIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i64());
    } break;
    case eOpcode::S_OR_B32: {
      parse<op::OrIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_OR_B64: {
      parse<op::OrIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i64());
    } break;
    case eOpcode::S_XOR_B32: {
      parse<op::XorIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_XOR_B64: {
      parse<op::XorIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i64());
    } break;
    case eOpcode::S_ANDN2_B32: {
      parse<op::AndIOp>(sdst, OpDst(eOperandKind::SCC()), src0, OpSrc(src1.kind, {OpFlags::eNot}), types().i32());
    } break;
    case eOpcode::S_ANDN2_B64: {
      parse<op::AndIOp>(sdst, OpDst(eOperandKind::SCC()), src0, OpSrc(src1.kind, {OpFlags::eNot}), types().i64());
    } break;
    case eOpcode::S_ORN2_B32: {
      parse<op::OrIOp>(sdst, OpDst(eOperandKind::SCC()), src0, OpSrc(src1.kind, {OpFlags::eNot}), types().i32());
    } break;
    case eOpcode::S_ORN2_B64: {
      parse<op::OrIOp>(sdst, OpDst(eOperandKind::SCC()), src0, OpSrc(src1.kind, {OpFlags::eNot}), types().i64());
    } break;
    case eOpcode::S_NAND_B32: {
      parse<op::AndIOp>(OpDst(sdst.kind, {OpFlags::eNot}), OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_NAND_B64: {
      parse<op::AndIOp>(OpDst(sdst.kind, {OpFlags::eNot}), OpDst(eOperandKind::SCC()), src0, src1, types().i64());
    } break;
    case eOpcode::S_NOR_B32: {
      parse<op::OrIOp>(OpDst(sdst.kind, {OpFlags::eNot}), OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_NOR_B64: {
      parse<op::OrIOp>(OpDst(sdst.kind, {OpFlags::eNot}), OpDst(eOperandKind::SCC()), src0, src1, types().i64());
    } break;
    case eOpcode::S_XNOR_B32: {
      parse<op::XorIOp>(OpDst(sdst.kind, {OpFlags::eNot}), OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_XNOR_B64: {
      parse<op::XorIOp>(OpDst(sdst.kind, {OpFlags::eNot}), OpDst(eOperandKind::SCC()), src0, src1, types().i64());
    } break;
    case eOpcode::S_LSHL_B32: {
      parse<op::LSHLOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_LSHL_B64: {
      parse<op::LSHLOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i64());
    } break;
    case eOpcode::S_LSHR_B32: {
      parse<op::LSHROp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_LSHR_B64: {
      parse<op::LSHROp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i64());
    } break;
    case eOpcode::S_ASHR_I32: {
      parse<op::ASHROp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_ASHR_I64: {
      parse<op::ASHROp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i64());
    } break;
    case eOpcode::S_BFM_B32: {
      parse<op::BitfieldMaskOp>(sdst, src0, src1, types().i32());
    } break;
    case eOpcode::S_BFM_B64: {
      parse<op::BitfieldMaskOp>(sdst, src0, src1, types().i64());
    } break;
    case eOpcode::S_MUL_I32: {
      parse<op::MulIOp>(sdst, src0, src1, types().i32());
    } break;
    case eOpcode::S_BFE_U32: {
      parse<op::BitfieldExtractUIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_BFE_I32: {
      parse<op::BitfieldExtractSIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    case eOpcode::S_BFE_U64: {
      parse<op::BitfieldExtractUIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i64());
    } break;
    case eOpcode::S_BFE_I64: {
      parse<op::BitfieldExtractSIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i64());
    } break;
    // case eOpcode::S_CBRANCH_G_FORK: {
    // } break;
    case eOpcode::S_ABSDIFF_I32: {
      parse<op::AbsDiffIOp>(sdst, OpDst(eOperandKind::SCC()), src0, src1, types().i32());
    } break;
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }

  if (src0.kind.isLiteral() || src1.kind.isLiteral()) {
    return sizeof(uint64_t);
  }
  return sizeof(uint32_t);
}

uint8_t Parser::handleSopc(CodeBlock& cb, pc_t pc, uint32_t const* pCode) {
  auto       inst = SOPC(*pCode);
  auto const op   = (eOpcode)(OPcodeStart_SOPC + inst.template get<SOPC::Field::OP>());

  auto const sdst = OpDst(eOperandKind::SCC());
  auto       src0 = OpSrc(eOperandKind((eOperandKind_t)inst.template get<SOPC::Field::SSRC0>()));
  auto       src1 = OpSrc(eOperandKind((eOperandKind_t)inst.template get<SOPC::Field::SSRC1>()));

  switch (op) {
    case eOpcode::S_CMP_EQ_I32: {
      parse<op::CmpOp>(op::eCmpIPredicate::eq, sdst, src0, src1, types().i32());
    } break;
    case eOpcode::S_CMP_LG_I32: {
      parse<op::CmpOp>(op::eCmpIPredicate::ne, sdst, src0, src1, types().i32());
    } break;
    case eOpcode::S_CMP_GT_I32: {
      parse<op::CmpOp>(op::eCmpIPredicate::sgt, sdst, src0, src1, types().i32());
    } break;
    case eOpcode::S_CMP_GE_I32: {
      parse<op::CmpOp>(op::eCmpIPredicate::sge, sdst, src0, src1, types().i32());
    } break;
    case eOpcode::S_CMP_LT_I32: {
      parse<op::CmpOp>(op::eCmpIPredicate::slt, sdst, src0, src1, types().i32());
    } break;
    case eOpcode::S_CMP_LE_I32: {
      parse<op::CmpOp>(op::eCmpIPredicate::sle, sdst, src0, src1, types().i32());
    } break;
    case eOpcode::S_CMP_EQ_U32: {
      parse<op::CmpOp>(op::eCmpIPredicate::eq, sdst, src0, src1, types().i32());
    } break;
    case eOpcode::S_CMP_LG_U32: {
      parse<op::CmpOp>(op::eCmpIPredicate::ne, sdst, src0, src1, types().i32());
    } break;
    case eOpcode::S_CMP_GT_U32: {
      parse<op::CmpOp>(op::eCmpIPredicate::ugt, sdst, src0, src1, types().i32());
    } break;
    case eOpcode::S_CMP_GE_U32: {
      parse<op::CmpOp>(op::eCmpIPredicate::uge, sdst, src0, src1, types().i32());
    } break;
    case eOpcode::S_CMP_LT_U32: {
      parse<op::CmpOp>(op::eCmpIPredicate::ult, sdst, src0, src1, types().i32());
    } break;
    case eOpcode::S_CMP_LE_U32: {
      parse<op::CmpOp>(op::eCmpIPredicate::ule, sdst, src0, src1, types().i32());
    } break;
    case eOpcode::S_BITCMP0_B32: {
      parse<op::IsBitClearOp>(sdst, src0, src1, types().i32());
    } break;
    case eOpcode::S_BITCMP1_B32: {
      parse<op::IsBitSetOp>(sdst, src0, src1, types().i32());
    } break;
    case eOpcode::S_BITCMP0_B64: {
      parse<op::IsBitClearOp>(sdst, src0, src1, types().i64());
    } break;
    case eOpcode::S_BITCMP1_B64: {
      parse<op::IsBitSetOp>(sdst, src0, src1, types().i64());
    } break;
    case eOpcode::S_SETVSKIP: {
      parse<op::IsBitSetOp>(OpDst(eOperandKind::VSKIP()), src0, src1, types().i32());
    } break;
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }

  if (src0.kind.isLiteral() || src1.kind.isLiteral()) {
    return sizeof(uint64_t);
  }

  return sizeof(uint32_t);
}

uint8_t Parser::handleSopk(CodeBlock& cb, pc_t pc, uint32_t const* pCode) {
  auto       inst = SOPK(*pCode);
  auto const op   = (eOpcode)(OPcodeStart_SOPK + inst.template get<SOPK::Field::OP>());

  auto const sdst = OpDst(eOperandKind((eOperandKind_t)inst.template get<SOPK::Field::SDST>()));
  auto const src0 = OpSrc((uint32_t)((int32_t)(int16_t)inst.template get<SOPK::Field::SIMM16>()));

  switch (op) {
    case eOpcode::S_MOVK_I32: {
      parse<op::MoveOp>(sdst, src0, types().i32());
    } break;
    case eOpcode::S_MOVK_HI_I32: {
      parse<op::BitfieldInsertOp>(sdst, OpSrc(sdst.kind), src0, OpSrc(16), OpSrc(16), types().i32());
    } break;
    case eOpcode::S_CMOVK_I32: {
      parse<op::CMoveOp>(sdst, OpSrc(eOperandKind::SCC()), src0, OpSrc(sdst.kind), types().i32());
    } break;
    case eOpcode::S_CMPK_EQ_I32: {
      parse<op::CmpOp>(op::eCmpIPredicate::eq, sdst, OpSrc(sdst.kind), src0, types().i32());
    } break;
    case eOpcode::S_CMPK_LG_I32: {
      parse<op::CmpOp>(op::eCmpIPredicate::ne, sdst, OpSrc(sdst.kind), src0, types().i32());
    } break;
    case eOpcode::S_CMPK_GT_I32: {
      parse<op::CmpOp>(op::eCmpIPredicate::sgt, sdst, OpSrc(sdst.kind), src0, types().i32());
    } break;
    case eOpcode::S_CMPK_GE_I32: {
      parse<op::CmpOp>(op::eCmpIPredicate::sge, sdst, OpSrc(sdst.kind), src0, types().i32());
    } break;
    case eOpcode::S_CMPK_LT_I32: {
      parse<op::CmpOp>(op::eCmpIPredicate::slt, sdst, OpSrc(sdst.kind), src0, types().i32());
    } break;
    case eOpcode::S_CMPK_LE_I32: {
      parse<op::CmpOp>(op::eCmpIPredicate::sle, sdst, OpSrc(sdst.kind), src0, types().i32());
    } break;
    case eOpcode::S_CMPK_EQ_U32: {
      parse<op::CmpOp>(op::eCmpIPredicate::eq, sdst, OpSrc(sdst.kind), src0, types().i32());
    } break;
    case eOpcode::S_CMPK_LG_U32: {
      parse<op::CmpOp>(op::eCmpIPredicate::ne, sdst, OpSrc(sdst.kind), src0, types().i32());
    } break;
    case eOpcode::S_CMPK_GT_U32: {
      parse<op::CmpOp>(op::eCmpIPredicate::ugt, sdst, OpSrc(sdst.kind), src0, types().i32());
    } break;
    case eOpcode::S_CMPK_GE_U32: {
      parse<op::CmpOp>(op::eCmpIPredicate::uge, sdst, OpSrc(sdst.kind), src0, types().i32());
    } break;
    case eOpcode::S_CMPK_LT_U32: {
      parse<op::CmpOp>(op::eCmpIPredicate::ult, sdst, OpSrc(sdst.kind), src0, types().i32());
    } break;
    case eOpcode::S_CMPK_LE_U32: {
      parse<op::CmpOp>(op::eCmpIPredicate::ule, sdst, OpSrc(sdst.kind), src0, types().i32());
    } break;
    case eOpcode::S_ADDK_I32: {
      parse<op::AddSIOp>(sdst, OpDst(eOperandKind::SCC()), OpSrc(sdst.kind), src0, types().i32());
    } break;
    case eOpcode::S_MULK_I32: {
      parse<op::MulIOp>(sdst, OpSrc(sdst.kind), src0, types().i32());
    } break;
    // case eOpcode::S_CBRANCH_I_FORK: {
    // } break;
    // case eOpcode::S_GETREG_B32: {
    // } break;
    // case eOpcode::S_SETREG_B32: {
    // } break;
    // case eOpcode::S_GETREG_REGRD_B32: {
    // } break;
    // case eOpcode::S_SETREG_IMM32_B32: {
    // } break;
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }
  return sizeof(uint32_t);
}

uint8_t Parser::handleSopp(CodeBlock& cb, pc_t pc, uint32_t const* pCode) {
  auto       inst = SOPP(*pCode);
  auto const op   = (eOpcode)(OPcodeStart_SOPP + inst.template get<SOPP::Field::OP>());

  auto const offset = (int16_t)inst.template get<SOPP::Field::SIMM16>();

  switch (op) {
    case eOpcode::S_NOP: break; // ignore
    case eOpcode::S_ENDPGM: {

      cb.pc_end = pc;
      _mlirBuilder.create<mlir::func::ReturnOp>(_defaultLocation);
    } break;
    case eOpcode::S_BRANCH: {
      cb.pc_end = pc;

      auto targetPc = (int64_t)(sizeof(uint32_t) + pc) + sizeof(uint32_t) * (int64_t)offset;
      auto target   = getOrCreateBlock(targetPc, cb.mlirBlock->getParent());

      _mlirBuilder.create<mlir::cf::BranchOp>(_defaultLocation, target->mlirBlock);
    } break;
    case eOpcode::S_CBRANCH_SCC0: {
      cb.pc_end = pc;

      auto target0 = getOrCreateBlock(sizeof(uint32_t) + pc, cb.mlirBlock->getParent());
      auto target1 = getOrCreateBlock((int64_t)(sizeof(uint32_t) + pc) + sizeof(uint32_t) * (int64_t)offset, cb.mlirBlock->getParent());

      auto predicate = loadRegister(OpSrc(eOperandKind::SCC()), types().i1());
      _mlirBuilder.create<mlir::cf::CondBranchOp>(_defaultLocation, predicate, target0->mlirBlock, target1->mlirBlock);
    } break;
    case eOpcode::S_CBRANCH_SCC1: {
      cb.pc_end = pc;

      auto target0 = getOrCreateBlock(sizeof(uint32_t) + pc, cb.mlirBlock->getParent());
      auto target1 = getOrCreateBlock((int64_t)(sizeof(uint32_t) + pc) + sizeof(uint32_t) * (int64_t)offset, cb.mlirBlock->getParent());

      auto predicate = loadRegister(OpSrc(eOperandKind::SCC()), types().i1());
      _mlirBuilder.create<mlir::cf::CondBranchOp>(_defaultLocation, predicate, target1->mlirBlock, target0->mlirBlock);
    } break;
    case eOpcode::S_CBRANCH_VCCZ: {
      cb.pc_end = pc;

      auto target0 = getOrCreateBlock(sizeof(uint32_t) + pc, cb.mlirBlock->getParent());
      auto target1 = getOrCreateBlock((int64_t)(sizeof(uint32_t) + pc) + sizeof(uint32_t) * (int64_t)offset, cb.mlirBlock->getParent());

      auto predicate = loadRegister(OpSrc(eOperandKind::VCC()), types().i1());
      _mlirBuilder.create<mlir::cf::CondBranchOp>(_defaultLocation, predicate, target0->mlirBlock, target1->mlirBlock);
    } break;
    case eOpcode::S_CBRANCH_VCCNZ: {
      cb.pc_end = pc;

      auto target0 = getOrCreateBlock(sizeof(uint32_t) + pc, cb.mlirBlock->getParent());
      auto target1 = getOrCreateBlock((int64_t)(sizeof(uint32_t) + pc) + sizeof(uint32_t) * (int64_t)offset, cb.mlirBlock->getParent());

      auto predicate = loadRegister(OpSrc(eOperandKind::VCC()), types().i1());
      _mlirBuilder.create<mlir::cf::CondBranchOp>(_defaultLocation, predicate, target1->mlirBlock, target0->mlirBlock);
    } break;
    case eOpcode::S_CBRANCH_EXECZ: {
      cb.pc_end = pc;

      auto target0 = getOrCreateBlock(sizeof(uint32_t) + pc, cb.mlirBlock->getParent());
      auto target1 = getOrCreateBlock((int64_t)(sizeof(uint32_t) + pc) + sizeof(uint32_t) * (int64_t)offset, cb.mlirBlock->getParent());

      // todo use getThreadExec() or so
      auto predicate = loadRegister(OpSrc(eOperandKind::EXEC()), types().i1());
      _mlirBuilder.create<mlir::cf::CondBranchOp>(_defaultLocation, predicate, target0->mlirBlock, target1->mlirBlock);
    } break;
    case eOpcode::S_CBRANCH_EXECNZ: {
      cb.pc_end = pc;

      auto target0 = getOrCreateBlock(sizeof(uint32_t) + pc, cb.mlirBlock->getParent());
      auto target1 = getOrCreateBlock((int64_t)(sizeof(uint32_t) + pc) + sizeof(uint32_t) * (int64_t)offset, cb.mlirBlock->getParent());

      // todo use getThreadExec() or so
      auto predicate = loadRegister(OpSrc(eOperandKind::EXEC()), types().i1());
      _mlirBuilder.create<mlir::cf::CondBranchOp>(_defaultLocation, predicate, target1->mlirBlock, target0->mlirBlock);
    } break;
    // case eOpcode::S_BARRIER: { // todo
    // } break;
    // case eOpcode::S_SETKILL: {} break; // Does not exist
    case eOpcode::S_WAITCNT:
      break; // ignore
    // case eOpcode::S_SETHALT: {} break; // Does not exist
    case eOpcode::S_SLEEP: break; // ignore
    case eOpcode::S_SETPRIO: {
    } break;
    case eOpcode::S_SENDMSG: {
    } break;
    // case eOpcode::S_SENDMSGHALT: {} break; // Does not exist
    // case eOpcode::S_TRAP: {} break; // Does not exist
    case eOpcode::S_ICACHE_INV: {
    } break;
    case eOpcode::S_INCPERFLEVEL: {
    } break;
    case eOpcode::S_DECPERFLEVEL: {
    } break;
    case eOpcode::S_TTRACEDATA: {
    } break;
    // case eOpcode::S_CBRANCH_CDBGSYS: {} break; // Does not exist
    // case eOpcode::S_CBRANCH_CDBGUSER: {} break; // Does not exist
    // case eOpcode::S_CBRANCH_CDBGSYS_OR_USER: {} break; // Does not exist
    // case eOpcode::S_CBRANCH_CDBGSYS_AND_USER: {} break;// Does not exist
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }

  return sizeof(uint32_t);
}
} // namespace compiler::frontend