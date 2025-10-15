#include "frontend/ir_types.h"
#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "../instruction_builder.h"
#include "translate.h"

#include <bitset>
#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
InstructionKind_t handleVop1(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode, bool extended) {
  using namespace parser;

  parser::eOpcode op;
  OpDst           vdst;
  OpSrc           src0;

  if (extended) {
    auto inst = VOP3(getU64(*pCode));
    op        = (parser::eOpcode)(OPcodeStart_VOP1 + inst.template get<VOP3::Field::OP>() - OpcodeOffset_VOP1_VOP3);

    auto const vdst_ = eOperandKind::VGPR(inst.template get<VOP3::Field::VDST>());
    auto const src0_ = eOperandKind((eOperandKind_t)inst.template get<VOP3::Field::SRC0>());

    auto const           omod   = inst.template get<VOP3::Field::OMOD>();
    std::bitset<3> const negate = inst.template get<VOP3::Field::NEG>();
    std::bitset<3> const abs    = inst.template get<VOP3::Field::ABS>();
    auto const           clamp  = inst.template get<VOP3::Field::CLAMP>();

    src0 = OpSrc(src0_, negate[0], abs[0]);
    vdst = OpDst(vdst_, omod, clamp, false);
    *pCode += 1;
  } else {
    auto inst = VOP1(**pCode);
    op        = (parser::eOpcode)(OPcodeStart_VOP1 + inst.template get<VOP1::Field::OP>());

    vdst = OpDst(eOperandKind::VGPR(inst.template get<VOP1::Field::VDST>()));
    src0 = OpSrc(eOperandKind((eOperandKind_t)inst.template get<VOP1::Field::SRC0>()));
    if (src0.kind.isLiteral()) {
      *pCode += 1;
      builder.createInstruction(create::literalOp(**pCode));
    }
  }

  *pCode += 1;

  switch (op) {
    case eOpcode::V_NOP: break; // ignore
    case eOpcode::V_MOV_B32: {
      builder.createVirtualInst(create::moveOp(vdst, src0, ir::OperandType::i32()));
    } break;
    // case eOpcode::V_READFIRSTLANE_B32: {} break; // todo, needs lane elimination pass
    case eOpcode::V_CVT_I32_F64: {
      builder.createVirtualInst(create::convFPToSIOp(vdst, ir::OperandType::i32(), src0, ir::OperandType::f64()));
    } break;
    case eOpcode::V_CVT_F64_I32: {
      builder.createVirtualInst(create::convSIToFPOp(vdst, ir::OperandType::f64(), src0, ir::OperandType::i32()));
    } break;
    case eOpcode::V_CVT_F32_I32: {
      builder.createVirtualInst(create::convSIToFPOp(vdst, ir::OperandType::f32(), src0, ir::OperandType::i32()));
    } break;
    case eOpcode::V_CVT_F32_U32: {
      builder.createVirtualInst(create::convUIToFPOp(vdst, ir::OperandType::f32(), src0, ir::OperandType::i32()));
    } break;
    case eOpcode::V_CVT_U32_F32: {
      builder.createVirtualInst(create::convFPToUIOp(vdst, ir::OperandType::i32(), src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_CVT_I32_F32: {
      builder.createVirtualInst(create::convFPToSIOp(vdst, ir::OperandType::i32(), src0, ir::OperandType::f32()));
    } break;
      // case eOpcode::V_MOV_FED_B32: break; // Does not exist
    case eOpcode::V_CVT_F16_F32: {
      builder.createVirtualInst(create::packHalf2x16Op(vdst, src0, OpSrc(eOperandKind::createImm(0))));
    } break;
    case eOpcode::V_CVT_F32_F16: {
      builder.createVirtualInst(create::unpackHalf2x16(vdst, OpDst(eOperandKind::Temp0()), src0));
    } break;
    case eOpcode::V_CVT_RPI_I32_F32: {
      builder.createVirtualInst(
          create::addFOp(OpDst(vdst.kind), src0, OpSrc(eOperandKind((eOperandKind_t)eOperandKind::eBase::ConstFloat_0_5)), ir::OperandType::f32()));
      builder.createVirtualInst(create::floorOp(OpDst(vdst.kind), OpSrc(vdst.kind), ir::OperandType::f32()));
      builder.createVirtualInst(create::convFPToSIOp(vdst, ir::OperandType::i32(), OpSrc(vdst.kind), ir::OperandType::f32()));
    } break;
    case eOpcode::V_CVT_FLR_I32_F32: {
      builder.createVirtualInst(create::floorOp(vdst, src0, ir::OperandType::f32()));
      builder.createVirtualInst(create::convFPToSIOp(vdst, ir::OperandType::i32(), OpSrc(vdst.kind), ir::OperandType::f32()));
    } break;
    case eOpcode::V_CVT_OFF_F32_I4: {
      builder.createVirtualInst(create::convSI4ToFloat(vdst, OpSrc(src0)));
    } break;
    case eOpcode::V_CVT_F32_F64: {
      builder.createVirtualInst(create::truncFOp(vdst, src0));
    } break;
    case eOpcode::V_CVT_F64_F32: {
      builder.createVirtualInst(create::extFOp(vdst, src0));
    } break;
    case eOpcode::V_CVT_F32_UBYTE0: {
      builder.createInstruction(
          create::bitUIExtractOp(vdst, src0, OpSrc(eOperandKind::createImm(0)), OpSrc(eOperandKind::createImm(8)), ir::OperandType::i32()));
      builder.createVirtualInst(create::convUIToFPOp(vdst, ir::OperandType::f32(), OpSrc(vdst.kind), ir::OperandType::i32()));
    } break;
    case eOpcode::V_CVT_F32_UBYTE1: {
      builder.createInstruction(
          create::bitUIExtractOp(vdst, OpSrc(src0), OpSrc(eOperandKind::createImm(8)), OpSrc(eOperandKind::createImm(8)), ir::OperandType::i32()));
      builder.createVirtualInst(create::convUIToFPOp(vdst, ir::OperandType::f32(), OpSrc(vdst.kind), ir::OperandType::i32()));
    } break;
    case eOpcode::V_CVT_F32_UBYTE2: {
      builder.createInstruction(
          create::bitUIExtractOp(vdst, OpSrc(src0), OpSrc(eOperandKind::createImm(16)), OpSrc(eOperandKind::createImm(8)), ir::OperandType::i32()));
      builder.createVirtualInst(create::convUIToFPOp(vdst, ir::OperandType::f32(), OpSrc(vdst.kind), ir::OperandType::i32()));
    } break;
    case eOpcode::V_CVT_F32_UBYTE3: {
      builder.createInstruction(
          create::bitUIExtractOp(vdst, OpSrc(src0), OpSrc(eOperandKind::createImm(24)), OpSrc(eOperandKind::createImm(8)), ir::OperandType::i32()));
      builder.createVirtualInst(create::convUIToFPOp(vdst, ir::OperandType::f32(), OpSrc(vdst.kind), ir::OperandType::i32()));
    } break;
    case eOpcode::V_CVT_U32_F64: {
      builder.createVirtualInst(create::convFPToUIOp(vdst, ir::OperandType::i32(), src0, ir::OperandType::f64()));
    } break;
    case eOpcode::V_CVT_F64_U32: {
      builder.createVirtualInst(create::convUIToFPOp(vdst, ir::OperandType::f64(), src0, ir::OperandType::i32()));
    } break;
    case eOpcode::V_TRUNC_F64: {
      builder.createVirtualInst(create::truncOp(vdst, src0, ir::OperandType::f64()));
    } break;
    case eOpcode::V_CEIL_F64: {
      builder.createVirtualInst(create::ceilOp(vdst, src0, ir::OperandType::f64()));
    } break;
    case eOpcode::V_RNDNE_F64: {
      builder.createVirtualInst(create::roundEvenOp(vdst, src0, ir::OperandType::f64()));
    } break;
    case eOpcode::V_FLOOR_F64: {
      builder.createVirtualInst(create::floorOp(vdst, src0, ir::OperandType::f64()));
    } break;
    case eOpcode::V_FRACT_F32: {
      builder.createVirtualInst(create::fractOp(vdst, src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_TRUNC_F32: {
      builder.createVirtualInst(create::truncOp(vdst, src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_CEIL_F32: {
      builder.createVirtualInst(create::ceilOp(vdst, src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_RNDNE_F32: {
      builder.createVirtualInst(create::roundEvenOp(vdst, src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_FLOOR_F32: {
      builder.createVirtualInst(create::floorOp(vdst, src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_EXP_F32: {
      builder.createVirtualInst(create::exp2Op(vdst, src0));
    } break;
    case eOpcode::V_LOG_CLAMP_F32: {
      builder.createVirtualInst(create::log2Op(OpDst(vdst.kind), src0));
      builder.createVirtualInst(create::clampFMinMaxOp(vdst, OpSrc(vdst.kind), ir::OperandType::f32()));
    } break;
    case eOpcode::V_LOG_F32: {
      builder.createVirtualInst(create::log2Op(vdst, src0));
    } break;
    case eOpcode::V_RCP_CLAMP_F32: {
      builder.createVirtualInst(create::rcpOp(OpDst(vdst.kind), src0, ir::OperandType::f32()));
      builder.createVirtualInst(create::clampFMinMaxOp(vdst, OpSrc(vdst.kind), ir::OperandType::f32()));
    } break;
    case eOpcode::V_RCP_LEGACY_F32: {
      builder.createVirtualInst(create::rcpOp(OpDst(vdst.kind), src0, ir::OperandType::f32()));
      builder.createVirtualInst(create::clampFZeroOp(vdst, OpSrc(vdst.kind), ir::OperandType::f32()));
    } break;
    case eOpcode::V_RCP_F32: {
      builder.createVirtualInst(create::rcpOp(vdst, src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_RCP_IFLAG_F32: { // todo exception?
      builder.createVirtualInst(create::rcpOp(vdst, src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_RSQ_CLAMP_F32: {
      builder.createVirtualInst(create::rsqrtOp(OpDst(vdst.kind), src0, ir::OperandType::f32()));
      builder.createVirtualInst(create::clampFMinMaxOp(vdst, OpSrc(vdst.kind), ir::OperandType::f32()));
    } break;
    case eOpcode::V_RSQ_LEGACY_F32: {
      builder.createVirtualInst(create::rsqrtOp(OpDst(vdst.kind), src0, ir::OperandType::f32()));
      builder.createVirtualInst(create::clampFZeroOp(vdst, OpSrc(vdst.kind), ir::OperandType::f32()));
    } break;
    case eOpcode::V_RSQ_F32: {
      builder.createVirtualInst(create::rsqrtOp(vdst, src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_RCP_F64: {
      builder.createVirtualInst(create::rcpOp(vdst, src0, ir::OperandType::f64()));
    } break;
    case eOpcode::V_RCP_CLAMP_F64: {
      builder.createVirtualInst(create::rcpOp(OpDst(vdst.kind), src0, ir::OperandType::f64()));
      builder.createVirtualInst(create::clampFMinMaxOp(vdst, OpSrc(vdst.kind), ir::OperandType::f64()));
    } break;
    case eOpcode::V_RSQ_F64: {
      builder.createVirtualInst(create::rsqrtOp(vdst, src0, ir::OperandType::f64()));
    } break;
    case eOpcode::V_RSQ_CLAMP_F64: {
      builder.createVirtualInst(create::rsqrtOp(OpDst(vdst.kind), src0, ir::OperandType::f64()));
      builder.createVirtualInst(create::clampFMinMaxOp(vdst, OpSrc(vdst.kind), ir::OperandType::f64()));
    } break;
    case eOpcode::V_SQRT_F32: {
      builder.createVirtualInst(create::sqrtOp(vdst, src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_SQRT_F64: {
      builder.createVirtualInst(create::sqrtOp(vdst, src0, ir::OperandType::f64()));
    } break;
    case eOpcode::V_SIN_F32: {
      builder.createVirtualInst(create::sinOp(vdst, src0));
    } break;
    case eOpcode::V_COS_F32: {
      builder.createVirtualInst(create::cosOp(vdst, src0));
    } break;
    case eOpcode::V_NOT_B32: {
      builder.createVirtualInst(create::moveOp(vdst, OpSrc(src0.kind, !src0.flags.getNegate(), false), ir::OperandType::i32()));
    } break;
    case eOpcode::V_BFREV_B32: {
      builder.createVirtualInst(create::bitReverseOp(vdst, src0, ir::OperandType::i32()));
    } break;
    case eOpcode::V_FFBH_U32: {
      // relative to msb
      builder.createVirtualInst(create::bitReverseOp(OpDst(vdst.kind), src0, ir::OperandType::i32()));
      builder.createInstruction(create::findILsbOp(vdst, OpSrc(vdst.kind), ir::OperandType::i32()));
    } break;
    case eOpcode::V_FFBL_B32: {
      builder.createInstruction(create::findILsbOp(vdst, src0, ir::OperandType::i32()));
    } break;
    // case eOpcode::V_FFBH_I32: {} break; // todo
    case eOpcode::V_FREXP_EXP_I32_F64: {
      builder.createVirtualInst(create::frexpOp(vdst, OpDst(eOperandKind::Temp0()), src0, ir::OperandType::f64()));
    } break;
    case eOpcode::V_FREXP_MANT_F64: {
      builder.createVirtualInst(create::frexpOp(OpDst(eOperandKind::Temp0()), vdst, src0, ir::OperandType::f64()));
    } break;
    case eOpcode::V_FRACT_F64: {
      builder.createVirtualInst(create::fractOp(vdst, src0, ir::OperandType::f64()));
    } break;
    case eOpcode::V_FREXP_EXP_I32_F32: {
      // amdllpc translates frexp to
      // v_frexp_exp_i32_f32_e32 v1, v0
      // v_frexp_mant_f32_e32 v0, v0
      builder.createVirtualInst(create::frexpOp(vdst, OpDst(eOperandKind::Temp0()), src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_FREXP_MANT_F32: {
      builder.createVirtualInst(create::frexpOp(OpDst(eOperandKind::Temp0()), vdst, src0, ir::OperandType::f32()));
    } break;
    // case eOpcode::V_CLREXCP: {} break; // does not exist
    // case eOpcode::V_MOVRELD_B32: {} break; // todo
    // case eOpcode::V_MOVRELS_B32: {} break; //todo
    // case eOpcode::V_MOVRELSD_B32: {} break;// todo
    // case eOpcode::V_LOG_LEGACY_F32: {} break; // does not exist
    // case eOpcode::V_EXP_LEGACY_F32: {} break; // does not exist
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }
  return conv(op);
}
} // namespace compiler::frontend::translate