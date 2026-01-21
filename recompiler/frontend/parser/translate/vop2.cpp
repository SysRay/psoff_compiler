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

static bool isSDST(parser::eOpcode op) {
  using namespace parser;

  if (op == eOpcode::V_ADD_I32 || op == eOpcode::V_SUB_I32 || op == eOpcode::V_SUBREV_I32 || op == eOpcode::V_ADDC_U32 || op == eOpcode::V_SUBB_U32 ||
      op == eOpcode::V_SUBBREV_U32)
    return true;
  return false;
}

InstructionKind_t handleVop2(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode, bool extended) {
  using namespace parser;
  using namespace ir::dialect;

  eOpcode op;

  OpDst vdst, sdst;
  OpSrc src0, src1, src2;

  if (extended) {
    auto inst  = VOP3(getU64(*pCode));
    auto instS = VOP3_SDST(getU64(*pCode));
    op         = (eOpcode)(OPcodeStart_VOP2 + inst.template get<VOP3::Field::OP>() - OpcodeOffset_VOP2_VOP3);

    auto const vdst_ = eOperandKind::VGPR(inst.template get<VOP3::Field::VDST>());
    auto const src0_ = eOperandKind((eOperandKind_t)inst.template get<VOP3::Field::SRC0>());
    auto const src1_ = eOperandKind((eOperandKind_t)inst.template get<VOP3::Field::SRC1>());
    auto const src2_ = eOperandKind((eOperandKind_t)inst.template get<VOP3::Field::SRC2>());

    auto const sdst_ = eOperandKind((eOperandKind_t)instS.template get<VOP3_SDST::Field::SDST>());

    auto const           omod   = inst.template get<VOP3::Field::OMOD>();
    std::bitset<3> const negate = inst.template get<VOP3::Field::NEG>();
    std::bitset<3> const abs    = inst.template get<VOP3::Field::ABS>();
    auto const           clamp  = inst.template get<VOP3::Field::CLAMP>();

    // todo: neg, abs, omod, clamp (each operation?)
    // if (isSDST(op)) {
    //   src0 = createSrc(src0_, negate[0], false);
    //   src1 = createSrc(src1_, negate[1], false);
    //   src2 = createSrc(src2_, negate[2], false);
    //   vdst = createDst(vdst_, omod, false, false);
    //   sdst = createDst(sdst_, omod, false, false);
    // } else {
    //   src0 = createSrc(src0_, negate[0], abs[0]);
    //   src1 = createSrc(src1_, negate[1], abs[1]);
    //   src2 = createSrc(src2_, negate[2], abs[2]);
    //   vdst = createDst(vdst_, omod, clamp, false);
    // }

    if (isSDST(op)) {
      src0 = createSrc(src0_);
      src1 = createSrc(src1_);
      src2 = createSrc(src2_);
      vdst = createDst(vdst_);
      sdst = createDst(sdst_);
    } else {
      src0 = createSrc(src0_);
      src1 = createSrc(src1_);
      src2 = createSrc(src2_);
      vdst = createDst(vdst_);
    }

    *pCode += 1;
  } else {
    auto inst = VOP2(**pCode);
    op        = (eOpcode)(OPcodeStart_VOP2 + inst.template get<VOP2::Field::OP>());

    vdst = createDst(eOperandKind::VGPR(inst.template get<VOP2::Field::VDST>()));
    sdst = createDst(eOperandKind::VCC());
    src0 = createSrc(eOperandKind((eOperandKind_t)inst.template get<VOP2::Field::SRC0>()));
    src1 = createSrc(eOperandKind::VGPR(inst.template get<VOP2::Field::VSRC1>()));
    src2 = createSrc(eOperandKind::VCC());

    if (eOperandKind(src0.kind).isLiteral()) {
      *pCode += 1;
      src0 = createSrc(ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_u64 = **pCode}, ir::OperandType::i32()));
    } else if (eOperandKind(src1.kind).isLiteral()) {
      *pCode += 1;
      src1 = createSrc(ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_u64 = **pCode}, ir::OperandType::i32()));
    } else if (eOperandKind(src2.kind).isLiteral()) {
      *pCode += 1;
      src2 = createSrc(ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_u64 = **pCode}, ir::OperandType::i32()));
    }
  }

  *pCode += 1;

  switch (op) {
    case eOpcode::V_CNDMASK_B32: {
      ctx.create<core::SelectOp>(vdst, src2, src0, src1, ir::OperandType::f32());
    } break;
    case eOpcode::V_ADD_F32: {
      ctx.create<arith::AddFOp>(vdst, src0, src1, ir::OperandType::f32());
    } break;
    case eOpcode::V_SUB_F32: {
      ctx.create<arith::SubFOp>(vdst, src0, src1, ir::OperandType::f32());
    } break;
    case eOpcode::V_SUBREV_F32: {
      ctx.create<arith::SubFOp>(vdst, src1, src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_MAC_LEGACY_F32: {
      // todo src2 flags
      auto res = ctx.create<arith::FmaFOp>(createDst(), src0, src1, createSrc(eOperandKind(vdst.kind)), ir::OperandType::f32());
      ctx.create<arith::ClampFZeroOp>(vdst, res, ir::OperandType::f32());
    } break;
    case eOpcode::V_MUL_LEGACY_F32: {
      auto res = ctx.create<arith::MulFOp>(createDst(), src0, src1, ir::OperandType::f32());
      ctx.create<arith::ClampFZeroOp>(vdst, res, ir::OperandType::f32());
    } break;
    case eOpcode::V_MUL_F32: {
      ctx.create<arith::MulFOp>(vdst, src0, src1, ir::OperandType::f32());
    } break;
    case eOpcode::V_MUL_I32_I24: {
      auto s0 = ctx.create<arith::BitSIExtractOp>(createDst(), src0, createSrc(eOperandKind::createImm(0)), createSrc(eOperandKind::createImm(24)),
                                                  ir::OperandType::i32());
      auto s1 = ctx.create<arith::BitSIExtractOp>(createDst(), src1, createSrc(eOperandKind::createImm(0)), createSrc(eOperandKind::createImm(24)),
                                                  ir::OperandType::i32());
      ctx.create<arith::MulIOp>(vdst, s0, s1, ir::OperandType::i32());
    } break;
    case eOpcode::V_MUL_HI_I32_I24: {
      auto s0 = createSrc(ctx.create<arith::BitSIExtractOp>(createDst(), src0, createSrc(eOperandKind::createImm(0)), createSrc(eOperandKind::createImm(24)),
                                                            ir::OperandType::i32()));
      auto s1 = createSrc(ctx.create<arith::BitSIExtractOp>(createDst(), src1, createSrc(eOperandKind::createImm(0)), createSrc(eOperandKind::createImm(24)),
                                                            ir::OperandType::i32()));
      ctx.create<arith::MulIExtendedOp>(createDst(), vdst, s0, s1, ir::OperandType::i32());
    } break;
    case eOpcode::V_MUL_U32_U24: {
      auto s0 = ctx.create<arith::BitUIExtractOp>(createDst(), src0, createSrc(eOperandKind::createImm(0)), createSrc(eOperandKind::createImm(24)),
                                                  ir::OperandType::i32());
      auto s1 = ctx.create<arith::BitUIExtractOp>(createDst(), src1, createSrc(eOperandKind::createImm(0)), createSrc(eOperandKind::createImm(24)),
                                                  ir::OperandType::i32());
      ctx.create<arith::MulIOp>(vdst, s0, s1, ir::OperandType::i32());
    } break;
    case eOpcode::V_MUL_HI_U32_U24: {
      auto s0 = ctx.create<arith::BitUIExtractOp>(createDst(), src0, createSrc(eOperandKind::createImm(0)), createSrc(eOperandKind::createImm(24)),
                                                  ir::OperandType::i32());
      auto s1 = ctx.create<arith::BitUIExtractOp>(createDst(), src1, createSrc(eOperandKind::createImm(0)), createSrc(eOperandKind::createImm(24)),
                                                  ir::OperandType::i32());
      ctx.create<arith::MulIExtendedOp>(createDst(), vdst, s0, s1, ir::OperandType::i32());
    } break;
    case eOpcode::V_MIN_LEGACY_F32: {
      ctx.create<arith::MinNOp>(vdst, src1, src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_MAX_LEGACY_F32: {
      ctx.create<arith::MaxNOp>(vdst, src1, src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_MIN_F32: {
      ctx.create<arith::MinFOp>(vdst, src1, src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_MAX_F32: {
      ctx.create<arith::MaxFOp>(vdst, src1, src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_MIN_I32: {
      ctx.create<arith::MinSIOp>(vdst, src1, src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_MAX_I32: {
      ctx.create<arith::MaxSIOp>(vdst, src1, src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_MIN_U32: {
      ctx.create<arith::MinUIOp>(vdst, src1, src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_MAX_U32: {
      ctx.create<arith::MaxUIOp>(vdst, src1, src0, ir::OperandType::f32());
    } break;
    case eOpcode::V_LSHR_B32: {
      ctx.create<arith::ShiftRUIOp>(vdst, src0, src1, ir::OperandType::i32());
    } break;
    case eOpcode::V_LSHRREV_B32: {
      ctx.create<arith::ShiftRUIOp>(vdst, src1, src0, ir::OperandType::i32());
    } break;
    case eOpcode::V_ASHR_I32: {
      ctx.create<arith::ShiftRSIOp>(vdst, src0, src1, ir::OperandType::i32());
    } break;
    case eOpcode::V_ASHRREV_I32: {
      ctx.create<arith::ShiftRSIOp>(vdst, src1, src0, ir::OperandType::i32());
    } break;
    case eOpcode::V_LSHL_B32: {
      ctx.create<arith::ShiftLUIOp>(vdst, src0, src1, ir::OperandType::i32());
    } break;
    case eOpcode::V_LSHLREV_B32: {
      ctx.create<arith::ShiftLUIOp>(vdst, src1, src0, ir::OperandType::i32());
    } break;
    case eOpcode::V_AND_B32: {
      ctx.create<arith::BitAndOp>(vdst, src0, src1, ir::OperandType::i32());
    } break;
    case eOpcode::V_OR_B32: {
      ctx.create<arith::BitOrOp>(vdst, src0, src1, ir::OperandType::i32());
    } break;
    case eOpcode::V_XOR_B32: {
      ctx.create<arith::BitXorOp>(vdst, src0, src1, ir::OperandType::i32());
    } break;
    case eOpcode::V_BFM_B32: {
      ctx.create<arith::BitFieldMaskOp>(vdst, src0, src1, ir::OperandType::i32());
    } break;
    case eOpcode::V_MAC_F32: {
      // todo src2 flags
      ctx.create<arith::FmaFOp>(vdst, src0, src1, createSrc(eOperandKind(vdst.kind)), ir::OperandType::f32());
    } break;
    case eOpcode::V_MADMK_F32: {
      auto K = ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_f64 = std::bit_cast<float>(**pCode)}, ir::OperandType::f32());
      // todo src2 flags
      *pCode += 1;
      ctx.create<arith::FmaFOp>(vdst, src0, K, src1, ir::OperandType::f32());
    } break;
    case eOpcode::V_MADAK_F32: {
      auto K = ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_f64 = std::bit_cast<float>(**pCode)}, ir::OperandType::f32());
      // todo src2 flags
      *pCode += 1;
      ctx.create<arith::FmaFOp>(vdst, src0, src1, K, ir::OperandType::f32());
    } break;
    case eOpcode::V_BCNT_U32_B32: {
      auto res = ctx.create<arith::BitCountOp>(createDst(), src0, ir::OperandType::i32());
      ctx.create<arith::AddIOp>(vdst, res, src1, ir::OperandType::i32());
    } break;
    // case eOpcode::V_MBCNT_LO_U32_B32: {} break; // todo
    // case eOpcode::V_MBCNT_HI_U32_B32: {} break; // todo
    case eOpcode::V_ADD_I32: {
      auto res = ctx.create<arith::AddIOp>(vdst, src0, src1, ir::OperandType::i32());
      ctx.create<arith::CmpIOp>(sdst, res, src0, ir::OperandType::i32(), arith::CmpIPredicate::slt);
    } break;
    case eOpcode::V_SUB_I32: {
      auto res = ctx.create<arith::SubIOp>(vdst, src0, src1, ir::OperandType::i32());
      ctx.create<arith::CmpIOp>(sdst, res, src0, ir::OperandType::i32(), arith::CmpIPredicate::sgt);
    } break;
    case eOpcode::V_SUBREV_I32: {
      auto res = ctx.create<arith::SubIOp>(vdst, src1, src0, ir::OperandType::i32());
      ctx.create<arith::CmpIOp>(sdst, res, src1, ir::OperandType::i32(), arith::CmpIPredicate::sgt);
    } break;
    case eOpcode::V_ADDC_U32: {
      ctx.create<arith::AddCarryIOp>(vdst, sdst, src0, src1, src2, ir::OperandType::i32());
    } break;
    case eOpcode::V_SUBB_U32: {
      ctx.create<arith::SubBurrowIOp>(vdst, sdst, src0, src1, src2, ir::OperandType::i32());
    } break;
    case eOpcode::V_SUBBREV_U32: {
      ctx.create<arith::SubBurrowIOp>(vdst, sdst, src1, src0, src2, ir::OperandType::i32());
    } break;
    case eOpcode::V_LDEXP_F32: {
      ctx.create<arith::LdexpOp>(vdst, src0, src1, ir::OperandType::f32());
    } break;
    // case eOpcode::V_CVT_PKACCUM_U8_F32: {} break; // does not exist
    case eOpcode::V_CVT_PKNORM_I16_F32: {
      ctx.create<arith::PackSnorm2x16Op>(vdst, src0, src1);
    } break;
    case eOpcode::V_CVT_PKNORM_U16_F32: {
      ctx.create<arith::PackUnorm2x16Op>(vdst, src0, src1);
    } break;
    case eOpcode::V_CVT_PKRTZ_F16_F32: {
      ctx.create<arith::PackHalf2x16Op>(vdst, src0, src1);
    } break;
      // case eOpcode::V_CVT_PK_U16_U32: { } break; // todo
      // case eOpcode::V_CVT_PK_I16_I32: { } break; // todo
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }
  return conv(op);
}
} // namespace compiler::frontend::translate