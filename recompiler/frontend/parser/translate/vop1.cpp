#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "frontend/ir_types.h"
#include "ir/dialects/arith/builder.h"
#include "ir/dialects/core/builder.h"
#include "translate.h"

#include <bitset>
#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
InstructionKind_t handleVop1(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode, bool extended) {
  using namespace parser;
  using namespace ir::dialect;

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

    // todo: neg, abs, omod, clamp (each operation?)
    // src0 = createSrc(src0_, negate[0], abs[0]);
    // vdst = createDst(vdst_, omod, clamp, false);
    src0 = createSrc(src0_);
    vdst = createDst(vdst_);
    *pCode += 1;
  } else {
    auto inst = VOP1(**pCode);
    op        = (parser::eOpcode)(OPcodeStart_VOP1 + inst.template get<VOP1::Field::OP>());

    vdst = createDst(eOperandKind::VGPR(inst.template get<VOP1::Field::VDST>()));
    src0 = createSrc(eOperandKind((eOperandKind_t)inst.template get<VOP1::Field::SRC0>()));

    if (eOperandKind(src0.kind).isLiteral()) {
      *pCode += 1;
      src0 = createSrc(ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_u64 = **pCode}, ir::OperandType::i32()));
    }
  }
  *pCode += 1;

  using namespace ir::dialect;
  switch (op) {
    case eOpcode::V_NOP: break; // ignore
    case eOpcode::V_MOV_B32: {
      ctx.create<core::MoveOp>(vdst, src0, ir::OperandType::f32());
    } break;
    // case eOpcode::V_READFIRSTLANE_B32: {} break; // todo, needs lane elimination pass
    case eOpcode::V_CVT_I32_F64: {
      ctx.create<arith::ConvFPToSIOp>(vdst, ir::OperandType::i32(), src0, ir::OperandType::f64());
    } break;
    case eOpcode::V_CVT_F64_I32: {
      ctx.create<arith::ConvSIToFPOp>(vdst, ir::OperandType::f64(), src0, ir::OperandType::i32());
    } break;
    case eOpcode::V_CVT_F32_I32: {
      ctx.create<arith::ConvSIToFPOp>(vdst, ir::OperandType::f32(), src0, ir::OperandType::i32());
    } break;
    case eOpcode::V_CVT_F32_U32: {
      ctx.create<arith::ConvUIToFPOp>(vdst, ir::OperandType::f32(), src0, ir::OperandType::i32());
    } break;
    case eOpcode::V_CVT_U32_F32: {
      ctx.create<arith::ConvFPToUIOp>(vdst, ir::OperandType::i32(), src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_CVT_I32_F32: {
      ctx.create<arith::ConvFPToSIOp>(vdst, ir::OperandType::i32(), src0, ir::OperandType::f32());
    } break;
      // case eOpcode::V_MOV_FED_B32: break; // Does not exist
    case eOpcode::V_CVT_F16_F32: {
      ctx.create<arith::PackHalf2x16Op>(vdst, src0, createSrc(eOperandKind::createImm(0)));
    } break;
    case eOpcode::V_CVT_F32_F16: {
      ctx.create<arith::UnpackHalf2x16>(vdst, createDst(), src0);
    } break;
    case eOpcode::V_CVT_RPI_I32_F32: {
      ctx.create<arith::AddFOp>(createDst(eOperandKind(vdst.kind)), src0, createSrc(eOperandKind((eOperandKind_t)eOperandKind::eBase::ConstFloat_0_5)),
                                ir::OperandType::f32());
      ctx.create<arith::FloorOp>(createDst(eOperandKind(vdst.kind)), createSrc(eOperandKind(vdst.kind)), ir::OperandType::f32());
      ctx.create<arith::ConvFPToSIOp>(vdst, ir::OperandType::i32(), createSrc(eOperandKind(vdst.kind)), ir::OperandType::f32());
    } break;
    case eOpcode::V_CVT_FLR_I32_F32: {
      ctx.create<arith::FloorOp>(vdst, src0, ir::OperandType::f32());
      ctx.create<arith::ConvFPToSIOp>(vdst, ir::OperandType::i32(), createSrc(eOperandKind(vdst.kind)), ir::OperandType::f32());
    } break;
    case eOpcode::V_CVT_OFF_F32_I4: {
      ctx.create<arith::ConvSI4ToFloat>(vdst, src0);
    } break;
    case eOpcode::V_CVT_F32_F64: {
      ctx.create<arith::TruncFOp>(vdst, src0);
    } break;
    case eOpcode::V_CVT_F64_F32: {
      ctx.create<arith::ExtFOp>(vdst, src0);
    } break;
    case eOpcode::V_CVT_F32_UBYTE0: {
      ctx.create<arith::BitUIExtractOp>(vdst, src0, createSrc(eOperandKind::createImm(0)), createSrc(eOperandKind::createImm(8)), ir::OperandType::i32());
      ctx.create<arith::ConvUIToFPOp>(vdst, ir::OperandType::f32(), createSrc(eOperandKind(vdst.kind)), ir::OperandType::i32());
    } break;
    case eOpcode::V_CVT_F32_UBYTE1: {
      ctx.create<arith::BitUIExtractOp>(vdst, src0, createSrc(eOperandKind::createImm(8)), createSrc(eOperandKind::createImm(8)), ir::OperandType::i32());
      ctx.create<arith::ConvUIToFPOp>(vdst, ir::OperandType::f32(), createSrc(eOperandKind(vdst.kind)), ir::OperandType::i32());
    } break;
    case eOpcode::V_CVT_F32_UBYTE2: {
      ctx.create<arith::BitUIExtractOp>(vdst, src0, createSrc(eOperandKind::createImm(16)), createSrc(eOperandKind::createImm(8)), ir::OperandType::i32());
      ctx.create<arith::ConvUIToFPOp>(vdst, ir::OperandType::f32(), createSrc(eOperandKind(vdst.kind)), ir::OperandType::i32());
    } break;
    case eOpcode::V_CVT_F32_UBYTE3: {
      ctx.create<arith::BitUIExtractOp>(vdst, src0, createSrc(eOperandKind::createImm(24)), createSrc(eOperandKind::createImm(8)), ir::OperandType::i32());
      ctx.create<arith::ConvUIToFPOp>(vdst, ir::OperandType::f32(), createSrc(eOperandKind(vdst.kind)), ir::OperandType::i32());
    } break;
    case eOpcode::V_CVT_U32_F64: {
      ctx.create<arith::ConvFPToUIOp>(vdst, ir::OperandType::i32(), src0, ir::OperandType::f64());
    } break;
    case eOpcode::V_CVT_F64_U32: {
      ctx.create<arith::ConvUIToFPOp>(vdst, ir::OperandType::f64(), src0, ir::OperandType::i32());
    } break;
    case eOpcode::V_TRUNC_F64: {
      ctx.create<arith::TruncOp>(vdst, src0, ir::OperandType::f64());
    } break;
    case eOpcode::V_CEIL_F64: {
      ctx.create<arith::CeilOp>(vdst, src0, ir::OperandType::f64());
    } break;
    case eOpcode::V_RNDNE_F64: {
      ctx.create<arith::RoundEvenOp>(vdst, src0, ir::OperandType::f64());
    } break;
    case eOpcode::V_FLOOR_F64: {
      ctx.create<arith::FloorOp>(vdst, src0, ir::OperandType::f64());
    } break;
    case eOpcode::V_FRACT_F32: {
      ctx.create<arith::FractOp>(vdst, src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_TRUNC_F32: {
      ctx.create<arith::TruncOp>(vdst, src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_CEIL_F32: {
      ctx.create<arith::CeilOp>(vdst, src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_RNDNE_F32: {
      ctx.create<arith::RoundEvenOp>(vdst, src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_FLOOR_F32: {
      ctx.create<arith::FloorOp>(vdst, src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_EXP_F32: {
      ctx.create<arith::Exp2Op>(vdst, src0);
    } break;
    case eOpcode::V_LOG_CLAMP_F32: {
      ctx.create<arith::Log2Op>(createDst(eOperandKind(vdst.kind)), src0);
      ctx.create<arith::ClampFMinMaxOp>(vdst, createSrc(eOperandKind(vdst.kind)), ir::OperandType::f32());
    } break;
    case eOpcode::V_LOG_F32: {
      ctx.create<arith::Log2Op>(vdst, src0);
    } break;
    case eOpcode::V_RCP_CLAMP_F32: {
      ctx.create<arith::RcpOp>(createDst(eOperandKind(vdst.kind)), src0, ir::OperandType::f32());
      ctx.create<arith::ClampFMinMaxOp>(vdst, createSrc(eOperandKind(vdst.kind)), ir::OperandType::f32());
    } break;
    case eOpcode::V_RCP_LEGACY_F32: {
      ctx.create<arith::RcpOp>(createDst(eOperandKind(vdst.kind)), src0, ir::OperandType::f32());
      ctx.create<arith::ClampFZeroOp>(vdst, createSrc(eOperandKind(vdst.kind)), ir::OperandType::f32());
    } break;
    case eOpcode::V_RCP_F32: {
      ctx.create<arith::RcpOp>(vdst, src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_RCP_IFLAG_F32: { // todo exception?
      ctx.create<arith::RcpOp>(vdst, src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_RSQ_CLAMP_F32: {
      ctx.create<arith::RsqrtOp>(createDst(eOperandKind(vdst.kind)), src0, ir::OperandType::f32());
      ctx.create<arith::ClampFMinMaxOp>(vdst, createSrc(eOperandKind(vdst.kind)), ir::OperandType::f32());
    } break;
    case eOpcode::V_RSQ_LEGACY_F32: {
      ctx.create<arith::RsqrtOp>(createDst(eOperandKind(vdst.kind)), src0, ir::OperandType::f32());
      ctx.create<arith::ClampFZeroOp>(vdst, createSrc(eOperandKind(vdst.kind)), ir::OperandType::f32());
    } break;
    case eOpcode::V_RSQ_F32: {
      ctx.create<arith::RsqrtOp>(vdst, src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_RCP_F64: {
      ctx.create<arith::RcpOp>(vdst, src0, ir::OperandType::f64());
    } break;
    case eOpcode::V_RCP_CLAMP_F64: {
      ctx.create<arith::RcpOp>(createDst(eOperandKind(vdst.kind)), src0, ir::OperandType::f64());
      ctx.create<arith::ClampFMinMaxOp>(vdst, createSrc(eOperandKind(vdst.kind)), ir::OperandType::f64());
    } break;
    case eOpcode::V_RSQ_F64: {
      ctx.create<arith::RsqrtOp>(vdst, src0, ir::OperandType::f64());
    } break;
    case eOpcode::V_RSQ_CLAMP_F64: {
      ctx.create<arith::RsqrtOp>(createDst(eOperandKind(vdst.kind)), src0, ir::OperandType::f64());
      ctx.create<arith::ClampFMinMaxOp>(vdst, createSrc(eOperandKind(vdst.kind)), ir::OperandType::f64());
    } break;
    case eOpcode::V_SQRT_F32: {
      ctx.create<arith::SqrtOp>(vdst, src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_SQRT_F64: {
      ctx.create<arith::SqrtOp>(vdst, src0, ir::OperandType::f64());
    } break;
    case eOpcode::V_SIN_F32: {
      ctx.create<arith::SinOp>(vdst, src0);
    } break;
    case eOpcode::V_COS_F32: {
      ctx.create<arith::CosOp>(vdst, src0);
    } break;
    case eOpcode::V_NOT_B32: {
      ctx.create<arith::NotOp>(vdst, src0, ir::OperandType::i32());
    } break;
    case eOpcode::V_BFREV_B32: {
      ctx.create<arith::BitReverseOp>(vdst, src0, ir::OperandType::i32());
    } break;
    case eOpcode::V_FFBH_U32: {
      // relative to msb
      ctx.create<arith::BitReverseOp>(createDst(eOperandKind(vdst.kind)), src0, ir::OperandType::i32());
      ctx.create<arith::FindILsbOp>(vdst, createSrc(eOperandKind(vdst.kind)), ir::OperandType::i32());
    } break;
    case eOpcode::V_FFBL_B32: {
      ctx.create<arith::FindILsbOp>(vdst, src0, ir::OperandType::i32());
    } break;
    // case eOpcode::V_FFBH_I32: {} break; // todo
    case eOpcode::V_FREXP_EXP_I32_F64: {
      ctx.create<arith::FrexpOp>(vdst, createDst(), src0, ir::OperandType::f64());
    } break;
    case eOpcode::V_FREXP_MANT_F64: {
      ctx.create<arith::FrexpOp>(createDst(), vdst, src0, ir::OperandType::f64());
    } break;
    case eOpcode::V_FRACT_F64: {
      ctx.create<arith::FractOp>(vdst, src0, ir::OperandType::f64());
    } break;
    case eOpcode::V_FREXP_EXP_I32_F32: {
      // amdllpc translates frexp to
      // v_frexp_exp_i32_f32_e32 v1, v0
      // v_frexp_mant_f32_e32 v0, v0
      ctx.create<arith::FrexpOp>(vdst, createDst(), src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_FREXP_MANT_F32: {
      ctx.create<arith::FrexpOp>(createDst(), vdst, src0, ir::OperandType::f32());
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