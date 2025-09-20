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
bool handleSop2(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = SOP2(**pCode);
  auto const op   = (parser::eOpcode)(OPcodeStart_SOP2 + inst.template get<SOP2::Field::OP>());

  auto const sdst = (eOperandKind)inst.template get<SOP2::Field::SDST>();
  auto const src0 = (eOperandKind)inst.template get<SOP2::Field::SSRC0>();
  auto const src1 = (eOperandKind)inst.template get<SOP2::Field::SSRC1>();

  if (src0 == eOperandKind::Literal || src1 == eOperandKind::Literal) {
    *pCode += 1;
    builder.createInstruction(create::literalOp(**pCode));
  }

  *pCode += 1;

  switch (op) {
    case eOpcode::S_ADD_U32: {
      builder.createInstruction(create::addIOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(src0), ir::OperandType::i32(), CmpIPredicate::ult));
    } break;
    case eOpcode::S_SUB_U32: {
      builder.createInstruction(create::subIOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(src1), OpSrc(src0), ir::OperandType::i32(), CmpIPredicate::ugt));
    } break;
    case eOpcode::S_ADD_I32: {
      builder.createInstruction(create::addIOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(src0), ir::OperandType::i32(), CmpIPredicate::slt));
    } break;
    case eOpcode::S_SUB_I32: {
      builder.createInstruction(create::subIOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(src1), OpSrc(src0), ir::OperandType::i32(), CmpIPredicate::sgt));
    } break;
    case eOpcode::S_ADDC_U32: {
      builder.createInstruction(
          create::addcIOp(OpDst(sdst), OpDst(eOperandKind::Scc), OpSrc(src0), OpSrc(src1), OpSrc(eOperandKind::Scc), ir::OperandType::i32()));
    } break;
    case eOpcode::S_SUBB_U32: {
      builder.createInstruction(
          create::subbIOp(OpDst(sdst), OpDst(eOperandKind::Scc), OpSrc(src0), OpSrc(src1), OpSrc(eOperandKind::Scc), ir::OperandType::i32()));
    } break;
    case eOpcode::S_MIN_I32: {
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(src0), OpSrc(src1), ir::OperandType::i32(), CmpIPredicate::slt));
      builder.createInstruction(create::selectOp(OpDst(sdst), OpSrc(eOperandKind::Scc), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
    } break;
    case eOpcode::S_MIN_U32: {
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(src0), OpSrc(src1), ir::OperandType::i32(), CmpIPredicate::ult));
      builder.createInstruction(create::selectOp(OpDst(sdst), OpSrc(eOperandKind::Scc), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
    } break;
    case eOpcode::S_MAX_I32: {
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(src0), OpSrc(src1), ir::OperandType::i32(), CmpIPredicate::sgt));
      builder.createInstruction(create::selectOp(OpDst(sdst), OpSrc(eOperandKind::Scc), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
    } break;
    case eOpcode::S_MAX_U32: {
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(src0), OpSrc(src1), ir::OperandType::i32(), CmpIPredicate::ugt));
      builder.createInstruction(create::selectOp(OpDst(sdst), OpSrc(eOperandKind::Scc), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
    } break;
    case eOpcode::S_CSELECT_B32: {
      builder.createInstruction(create::selectOp(OpDst(sdst), OpSrc(eOperandKind::Scc), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
    } break;
    case eOpcode::S_CSELECT_B64: {
      builder.createInstruction(create::selectOp(OpDst(sdst), OpSrc(eOperandKind::Scc), OpSrc(src0), OpSrc(src1), ir::OperandType::i64()));
    } break;
    case eOpcode::S_AND_B32: {
      builder.createInstruction(create::bitAndOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_AND_B64: {
      builder.createInstruction(create::bitAndOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i64()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i64(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_OR_B32: {
      builder.createInstruction(create::bitOrOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_OR_B64: {
      builder.createInstruction(create::bitOrOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i64()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i64(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_XOR_B32: {
      builder.createInstruction(create::bitXorOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_XOR_B64: {
      builder.createInstruction(create::bitXorOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i64()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i64(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_ANDN2_B32: {
      builder.createInstruction(create::bitAndOp(OpDst(sdst), OpSrc(src0), OpSrc(src1, true, false), ir::OperandType::i32()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_ANDN2_B64: {
      builder.createInstruction(create::bitAndOp(OpDst(sdst), OpSrc(src0), OpSrc(src1, true, false), ir::OperandType::i64()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i64(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_ORN2_B32: {
      builder.createInstruction(create::bitOrOp(OpDst(sdst), OpSrc(src0), OpSrc(src1, true, false), ir::OperandType::i32()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_ORN2_B64: {
      builder.createInstruction(create::bitOrOp(OpDst(sdst), OpSrc(src0), OpSrc(src1, true, false), ir::OperandType::i64()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i64(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_NAND_B32: {
      builder.createInstruction(create::bitAndOp(OpDst(sdst, 0, false, true), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_NAND_B64: {
      builder.createInstruction(create::bitAndOp(OpDst(sdst, 0, false, true), OpSrc(src0), OpSrc(src1), ir::OperandType::i64()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i64(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_NOR_B32: {
      builder.createInstruction(create::bitOrOp(OpDst(sdst, 0, false, true), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_NOR_B64: {
      builder.createInstruction(create::bitOrOp(OpDst(sdst, 0, false, true), OpSrc(src0), OpSrc(src1), ir::OperandType::i64()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i64(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_XNOR_B32: {
      builder.createInstruction(create::bitXorOp(OpDst(sdst, 0, false, true), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_XNOR_B64: {
      builder.createInstruction(create::bitXorOp(OpDst(sdst, 0, false, true), OpSrc(src0), OpSrc(src1), ir::OperandType::i64()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i64(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_LSHL_B32: {
      builder.createInstruction(create::shiftLUIOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_LSHL_B64: {
      builder.createInstruction(create::shiftLUIOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i64()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i64(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_LSHR_B32: {
      builder.createInstruction(create::shiftRUIOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_LSHR_B64: {
      builder.createInstruction(create::shiftRUIOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i64()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i64(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_ASHR_I32: {
      builder.createInstruction(create::shiftRSIOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_ASHR_I64: {
      builder.createInstruction(create::shiftRSIOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i64()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i64(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_BFM_B32: {
      builder.createInstruction(create::bitFieldMaskOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
    } break;
    case eOpcode::S_BFM_B64: {
      builder.createInstruction(create::bitFieldMaskOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i64()));
    } break;
    case eOpcode::S_MUL_I32: {
      builder.createInstruction(create::mulIOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
    } break;
    case eOpcode::S_BFE_U32: {
      builder.createInstruction(create::bitUIExtractOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_BFE_I32: {
      builder.createInstruction(create::bitSIExtractOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_BFE_U64: {
      builder.createInstruction(create::bitSIExtractOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i64()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i64(), CmpIPredicate::ne));
    } break;
    case eOpcode::S_BFE_I64: {
      builder.createInstruction(create::bitSIExtractOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i64()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i64(), CmpIPredicate::ne));
    } break;
    // case eOpcode::S_CBRANCH_G_FORK: { } break; // todo, make a block falltrough or handle data?
    case eOpcode::S_ABSDIFF_I32: {
      builder.createInstruction(create::subIOp(OpDst(sdst), OpSrc(src0), OpSrc(src1), ir::OperandType::i32()));
      builder.createInstruction(create::moveOp(OpDst(sdst), OpSrc(sdst, false, true), ir::OperandType::i32()));
      builder.createInstruction(create::cmpIOp(OpDst(eOperandKind::Scc), OpSrc(sdst), OpSrc(getUImm(0)), ir::OperandType::i32(), CmpIPredicate::ne));
    } break;
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }

  return true;
}
} // namespace compiler::frontend::translate