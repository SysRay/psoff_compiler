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
bool handleVop1(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode, bool extended) {
  using namespace parser;

  parser::eOpcode op;
  eOperandKind    vdst;
  eOperandKind    src0;

  uint8_t omod   = 0;
  bool    negate = false; ///< for src
  bool    abs    = false; ///< only with vop3
  bool    clamp  = false; ///< only with vop3

  if (extended) {
    auto inst = VOP3(getU64(*pCode));
    op        = (parser::eOpcode)(OPcodeStart_VOP1 + inst.template get<VOP3::Field::OP>() - OpcodeOffset_VOP1_VOP3);

    vdst = (eOperandKind)inst.template get<VOP3::Field::VDST>();
    src0 = (eOperandKind)inst.template get<VOP3::Field::SRC0>();

    omod   = inst.template get<VOP3::Field::OMOD>();
    negate = inst.template get<VOP3::Field::NEG>();
    abs    = inst.template get<VOP3::Field::ABS>();
    clamp  = inst.template get<VOP3::Field::CLAMP>();
    *pCode += 1;
  } else {
    auto inst = VOP1(**pCode);
    op        = (parser::eOpcode)(OPcodeStart_VOP1 + inst.template get<VOP1::Field::OP>());

    vdst = (eOperandKind)inst.template get<VOP1::Field::VDST>();
    src0 = (eOperandKind)inst.template get<VOP1::Field::SRC0>();
    if (src0 == eOperandKind::Literal) {
      *pCode += 1;
      builder.createInstruction(create::literalOp(**pCode));
    }
  }

  *pCode += 1;

  switch (op) {
    case eOpcode::V_NOP: break; // ignore
    case eOpcode::V_MOV_B32: {
      builder.createVirtualInst(create::moveOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::i32()));
    } break;
    // case eOpcode::V_READFIRSTLANE_B32: {} break; // todo, needs lane elimination pass
    case eOpcode::V_CVT_I32_F64: {
      builder.createVirtualInst(
          create::convFPToSIOp(OpDst(vdst, omod, clamp, false), ir::OperandType::i32(), OpSrc(src0, negate, abs), ir::OperandType::f64()));
    } break;
    case eOpcode::V_CVT_F64_I32: {
      builder.createVirtualInst(
          create::convSIToFPOp(OpDst(vdst, omod, clamp, false), ir::OperandType::f64(), OpSrc(src0, negate, abs), ir::OperandType::i32()));
    } break;
    case eOpcode::V_CVT_F32_I32: {
      builder.createVirtualInst(
          create::convSIToFPOp(OpDst(vdst, omod, clamp, false), ir::OperandType::f32(), OpSrc(src0, negate, abs), ir::OperandType::i32()));
    } break;
    case eOpcode::V_CVT_F32_U32: {
      builder.createVirtualInst(
          create::convUIToFPOp(OpDst(vdst, omod, clamp, false), ir::OperandType::f32(), OpSrc(src0, negate, abs), ir::OperandType::i32()));
    } break;
    case eOpcode::V_CVT_U32_F32: {
      builder.createVirtualInst(
          create::convFPToUIOp(OpDst(vdst, omod, clamp, false), ir::OperandType::i32(), OpSrc(src0, negate, abs), ir::OperandType::f32()));
    } break;
    case eOpcode::V_CVT_I32_F32: {
      builder.createVirtualInst(
          create::convFPToSIOp(OpDst(vdst, omod, clamp, false), ir::OperandType::i32(), OpSrc(src0, negate, abs), ir::OperandType::f32()));
    } break;
      // case eOpcode::V_MOV_FED_B32: break; // Does not exist
    case eOpcode::V_CVT_F16_F32: {
      builder.createVirtualInst(create::packHalf2x16Op(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), OpSrc(eOperandKind::ConstZero)));
    } break;
    case eOpcode::V_CVT_F32_F16: {
      builder.createVirtualInst(
          create::unpackHalf2x16(OpDst(vdst, omod, clamp, false), OpDst(eOperandKind::CustomTemp0Lo, omod, clamp, false), OpSrc(src0, negate, abs)));
    } break;
    case eOpcode::V_CVT_RPI_I32_F32: {
      builder.createVirtualInst(create::addFOp(OpDst(vdst), OpSrc(src0, negate, abs), OpSrc(getFImm(FIMM::f0_5)), ir::OperandType::f32()));
      builder.createVirtualInst(create::floorOp(OpDst(vdst), OpSrc(vdst), ir::OperandType::f32()));
      builder.createVirtualInst(create::convFPToSIOp(OpDst(vdst, omod, clamp, false), ir::OperandType::i32(), OpSrc(vdst), ir::OperandType::f32()));
    } break;
    case eOpcode::V_CVT_FLR_I32_F32: {
      builder.createVirtualInst(create::floorOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::f32()));
      builder.createVirtualInst(create::convFPToSIOp(OpDst(vdst, omod, clamp, false), ir::OperandType::i32(), OpSrc(vdst), ir::OperandType::f32()));
    } break;
    case eOpcode::V_CVT_OFF_F32_I4: {
      builder.createVirtualInst(create::convSI4ToFloat(OpDst(vdst, omod, clamp, false), OpSrc(src0)));
    } break;
    case eOpcode::V_CVT_F32_F64: {
      builder.createVirtualInst(create::truncFOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs)));
    } break;
    case eOpcode::V_CVT_F64_F32: {
      builder.createVirtualInst(create::extFOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs)));
    } break;
    case eOpcode::V_CVT_F32_UBYTE0: {
      builder.createInstruction(
          create::bitUIExtractOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), OpSrc(getUImm(8)), OpSrc(getUImm(0)), ir::OperandType::i32()));
      builder.createVirtualInst(create::convUIToFPOp(OpDst(vdst, omod, clamp, false), ir::OperandType::f32(), OpSrc(vdst), ir::OperandType::i32()));
    } break;
    case eOpcode::V_CVT_F32_UBYTE1: {
      builder.createInstruction(
          create::bitUIExtractOp(OpDst(vdst, omod, clamp, false), OpSrc(src0), OpSrc(getUImm(8)), OpSrc(getUImm(8)), ir::OperandType::i32()));
      builder.createVirtualInst(create::convUIToFPOp(OpDst(vdst, omod, clamp, false), ir::OperandType::f32(), OpSrc(vdst), ir::OperandType::i32()));
    } break;
    case eOpcode::V_CVT_F32_UBYTE2: {
      builder.createInstruction(
          create::bitUIExtractOp(OpDst(vdst, omod, clamp, false), OpSrc(src0), OpSrc(getUImm(8)), OpSrc(getUImm(16)), ir::OperandType::i32()));
      builder.createVirtualInst(create::convUIToFPOp(OpDst(vdst, omod, clamp, false), ir::OperandType::f32(), OpSrc(vdst), ir::OperandType::i32()));
    } break;
    case eOpcode::V_CVT_F32_UBYTE3: {
      builder.createInstruction(
          create::bitUIExtractOp(OpDst(vdst, omod, clamp, false), OpSrc(src0), OpSrc(getUImm(8)), OpSrc(getUImm(24)), ir::OperandType::i32()));
      builder.createVirtualInst(create::convUIToFPOp(OpDst(vdst, omod, clamp, false), ir::OperandType::f32(), OpSrc(vdst), ir::OperandType::i32()));
    } break;
    case eOpcode::V_CVT_U32_F64: {
      builder.createVirtualInst(
          create::convFPToUIOp(OpDst(vdst, omod, clamp, false), ir::OperandType::i32(), OpSrc(src0, negate, abs), ir::OperandType::f64()));
    } break;
    case eOpcode::V_CVT_F64_U32: {
      builder.createVirtualInst(
          create::convUIToFPOp(OpDst(vdst, omod, clamp, false), ir::OperandType::f64(), OpSrc(src0, negate, abs), ir::OperandType::i32()));
    } break;
    case eOpcode::V_TRUNC_F64: {
      builder.createVirtualInst(create::truncOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::f64()));
    } break;
    case eOpcode::V_CEIL_F64: {
      builder.createVirtualInst(create::ceilOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::f64()));
    } break;
    case eOpcode::V_RNDNE_F64: {
      builder.createVirtualInst(create::roundEvenOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::f64()));
    } break;
    case eOpcode::V_FLOOR_F64: {
      builder.createVirtualInst(create::floorOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::f64()));
    } break;
    case eOpcode::V_FRACT_F32: {
      builder.createVirtualInst(create::fractOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::f32()));
    } break;
    case eOpcode::V_TRUNC_F32: {
      builder.createVirtualInst(create::truncOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::f32()));
    } break;
    case eOpcode::V_CEIL_F32: {
      builder.createVirtualInst(create::ceilOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::f32()));
    } break;
    case eOpcode::V_RNDNE_F32: {
      builder.createVirtualInst(create::roundEvenOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::f32()));
    } break;
    case eOpcode::V_FLOOR_F32: {
      builder.createVirtualInst(create::floorOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::f32()));
    } break;
    case eOpcode::V_EXP_F32: {
      builder.createVirtualInst(create::exp2Op(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs)));
    } break;
    case eOpcode::V_LOG_CLAMP_F32: {
      builder.createVirtualInst(create::log2Op(OpDst(vdst), OpSrc(src0, negate, abs)));
      builder.createVirtualInst(create::clampFMinMaxOp(OpDst(vdst, omod, clamp, false), OpSrc(vdst), ir::OperandType::f32()));
    } break;
    case eOpcode::V_LOG_F32: {
      builder.createVirtualInst(create::log2Op(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs)));
    } break;
    case eOpcode::V_RCP_CLAMP_F32: {
      builder.createVirtualInst(create::rcpOp(OpDst(vdst), OpSrc(src0, negate, abs), ir::OperandType::f32()));
      builder.createVirtualInst(create::clampFMinMaxOp(OpDst(vdst, omod, clamp, false), OpSrc(vdst), ir::OperandType::f32()));
    } break;
    case eOpcode::V_RCP_LEGACY_F32: {
      builder.createVirtualInst(create::rcpOp(OpDst(vdst), OpSrc(src0, negate, abs), ir::OperandType::f32()));
      builder.createVirtualInst(create::clampFZeroOp(OpDst(vdst, omod, clamp, false), OpSrc(vdst), ir::OperandType::f32()));
    } break;
    case eOpcode::V_RCP_F32: {
      builder.createVirtualInst(create::rcpOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::f32()));
    } break;
    case eOpcode::V_RCP_IFLAG_F32: { // todo exception?
      builder.createVirtualInst(create::rcpOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::f32()));
    } break;
    case eOpcode::V_RSQ_CLAMP_F32: {
      builder.createVirtualInst(create::rsqrtOp(OpDst(vdst), OpSrc(src0, negate, abs), ir::OperandType::f32()));
      builder.createVirtualInst(create::clampFMinMaxOp(OpDst(vdst, omod, clamp, false), OpSrc(vdst), ir::OperandType::f32()));
    } break;
    case eOpcode::V_RSQ_LEGACY_F32: {
      builder.createVirtualInst(create::rsqrtOp(OpDst(vdst), OpSrc(src0, negate, abs), ir::OperandType::f32()));
      builder.createVirtualInst(create::clampFZeroOp(OpDst(vdst, omod, clamp, false), OpSrc(vdst), ir::OperandType::f32()));
    } break;
    case eOpcode::V_RSQ_F32: {
      builder.createVirtualInst(create::rsqrtOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::f32()));
    } break;
    case eOpcode::V_RCP_F64: {
      builder.createVirtualInst(create::rcpOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::f64()));
    } break;
    case eOpcode::V_RCP_CLAMP_F64: {
      builder.createVirtualInst(create::rcpOp(OpDst(vdst), OpSrc(src0, negate, abs), ir::OperandType::f64()));
      builder.createVirtualInst(create::clampFMinMaxOp(OpDst(vdst, omod, clamp, false), OpSrc(vdst), ir::OperandType::f64()));
    } break;
    case eOpcode::V_RSQ_F64: {
      builder.createVirtualInst(create::rsqrtOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::f64()));
    } break;
    case eOpcode::V_RSQ_CLAMP_F64: {
      builder.createVirtualInst(create::rsqrtOp(OpDst(vdst), OpSrc(src0, negate, abs), ir::OperandType::f64()));
      builder.createVirtualInst(create::clampFMinMaxOp(OpDst(vdst, omod, clamp, false), OpSrc(vdst), ir::OperandType::f64()));
    } break;
    case eOpcode::V_SQRT_F32: {
      builder.createVirtualInst(create::sqrtOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::f32()));
    } break;
    case eOpcode::V_SQRT_F64: {
      builder.createVirtualInst(create::sqrtOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::f64()));
    } break;
    case eOpcode::V_SIN_F32: {
      builder.createVirtualInst(create::sinOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs)));
    } break;
    case eOpcode::V_COS_F32: {
      builder.createVirtualInst(create::cosOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs)));
    } break;
    case eOpcode::V_NOT_B32: {
      builder.createVirtualInst(create::moveOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, true, false), ir::OperandType::i32()));
    } break;
    case eOpcode::V_BFREV_B32: {
      builder.createVirtualInst(create::bitReverseOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::i32()));
    } break;
    case eOpcode::V_FFBH_U32: {
      // relative to msb
      builder.createVirtualInst(create::bitReverseOp(OpDst(vdst), OpSrc(src0, negate, abs), ir::OperandType::i32()));
      builder.createInstruction(create::findILsbOp(OpDst(vdst, omod, clamp, false), OpSrc(vdst, negate, abs), ir::OperandType::i32()));
    } break;
    case eOpcode::V_FFBL_B32: {
      builder.createInstruction(create::findILsbOp(OpDst(vdst, omod, clamp, false), OpSrc(vdst, negate, abs), ir::OperandType::i32()));
    } break;
    // case eOpcode::V_FFBH_I32: {} break; // todo
    case eOpcode::V_FREXP_EXP_I32_F64: {
      builder.createVirtualInst(create::frexpOp(OpDst(vdst), OpDst(eOperandKind::CustomTemp0Lo), OpSrc(src0, negate, abs), ir::OperandType::f64()));
    } break;
    case eOpcode::V_FREXP_MANT_F64: {
      builder.createVirtualInst(create::frexpOp(OpDst(eOperandKind::CustomTemp0Lo), OpDst(vdst), OpSrc(src0, negate, abs), ir::OperandType::f64()));
    } break;
    case eOpcode::V_FRACT_F64: {
      builder.createVirtualInst(create::fractOp(OpDst(vdst, omod, clamp, false), OpSrc(src0, negate, abs), ir::OperandType::f64()));
    } break;
    case eOpcode::V_FREXP_EXP_I32_F32: {
      // amdllpc translated frexp to
      // v_frexp_exp_i32_f32_e32 v1, v0
      // v_frexp_mant_f32_e32 v0, v0
      builder.createVirtualInst(create::frexpOp(OpDst(vdst), OpDst(eOperandKind::CustomTemp0Lo), OpSrc(src0, negate, abs), ir::OperandType::f32()));
    } break;
    case eOpcode::V_FREXP_MANT_F32: {
      builder.createVirtualInst(create::frexpOp(OpDst(eOperandKind::CustomTemp0Lo), OpDst(vdst), OpSrc(src0, negate, abs), ir::OperandType::f32()));
    } break;
    // case eOpcode::V_CLREXCP: {} break; // does not exist
    // case eOpcode::V_MOVRELD_B32: {} break; // todo
    // case eOpcode::V_MOVRELS_B32: {} break; //todo
    // case eOpcode::V_MOVRELSD_B32: {} break;// todo
    // case eOpcode::V_LOG_LEGACY_F32: {} break; // does not exist
    // case eOpcode::V_EXP_LEGACY_F32: {} break; // does not exist
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }
  return true;
}
} // namespace compiler::frontend::translate