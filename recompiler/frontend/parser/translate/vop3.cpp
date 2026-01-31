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

  if (op == eOpcode::V_MAD_U64_U32 || op == eOpcode::V_MAD_I64_I32) return true;
  return false;
}

InstructionKind_t handleVop3(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;
  using namespace ir::dialect;

  auto inst  = VOP3(getU64(*pCode));
  auto instS = VOP3_SDST(getU64(*pCode));

  auto const op = (parser::eOpcode)(OPcodeStart_VOP3 + inst.template get<VOP3::Field::OP>());

  OpDst vdst, sdst;
  OpSrc src0, src1, src2;

  {
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
  }

  *pCode += 2;

  switch (op) {
    case eOpcode::V_MAD_LEGACY_F32: {
      auto res = ctx.create<arith::FmaFOp>(createDst(), src0, src1, src2, ir::OperandType::f32());
      ctx.create<arith::ClampFZeroOp>(vdst, res, ir::OperandType::f32());
    } break;
    case eOpcode::V_MAD_F32: {
      ctx.create<arith::FmaFOp>(vdst, src0, src1, src2, ir::OperandType::f32());
    } break;
    case eOpcode::V_MAD_I32_I24: {
      auto s0 = ctx.create<arith::BitSIExtractOp>(createDst(), src0, createSrc(eOperandKind::createImm(0)), createSrc(eOperandKind::createImm(24)),
                                                  ir::OperandType::i32());
      auto s1 = ctx.create<arith::BitSIExtractOp>(createDst(), src1, createSrc(eOperandKind::createImm(0)), createSrc(eOperandKind::createImm(24)),
                                                  ir::OperandType::i32());
      ctx.create<arith::FmaIOp>(vdst, s0, s1, src2, ir::OperandType::i32());
    } break;
    case eOpcode::V_MAD_U32_U24: {
      auto s0 = ctx.create<arith::BitUIExtractOp>(createDst(), src0, createSrc(eOperandKind::createImm(0)), createSrc(eOperandKind::createImm(24)),
                                                  ir::OperandType::i32());
      auto s1 = ctx.create<arith::BitUIExtractOp>(createDst(), src1, createSrc(eOperandKind::createImm(0)), createSrc(eOperandKind::createImm(24)),
                                                  ir::OperandType::i32());
      ctx.create<arith::FmaIOp>(vdst, s0, s1, src2, ir::OperandType::i32());
    } break;
    // case eOpcode::V_CUBEID_F32: {} break;  // todo
    // case eOpcode::V_CUBESC_F32: {} break;  // todo
    // case eOpcode::V_CUBETC_F32: {} break; // todo
    // case eOpcode::V_CUBEMA_F32: {} break; // todo
    case eOpcode::V_BFE_U32: {
      ctx.create<arith::BitUIExtractOp>(vdst, src0, src1, src2, ir::OperandType::i32());
    } break;
    case eOpcode::V_BFE_I32: {
      ctx.create<arith::BitSIExtractOp>(vdst, src0, src1, src2, ir::OperandType::i32());
    } break;
    case eOpcode::V_BFI_B32: {
      ctx.create<arith::BitFieldInsertOp>(vdst, src0, src1, src2, ir::OperandType::i32());
    } break;
    case eOpcode::V_FMA_F32: {
      ctx.create<arith::FmaFOp>(vdst, src0, src1, src2, ir::OperandType::f32());
    } break;
    case eOpcode::V_FMA_F64: {
      ctx.create<arith::FmaFOp>(vdst, src0, src1, src2, ir::OperandType::f64());
    } break;
    // case eOpcode::V_LERP_U8: {} break; // todo
    // case eOpcode::V_ALIGNBIT_B32: {} break; // todo
    // case eOpcode::V_ALIGNBYTE_B32: {} break; // todo
    //  case eOpcode::V_MULLIT_F32: {} break; // todo
    case eOpcode::V_MIN3_F32: {
      ctx.create<arith::Min3FOp>(vdst, src0, src1, src2, ir::OperandType::f32());
    } break;
    case eOpcode::V_MIN3_I32: {
      ctx.create<arith::Min3SIOp>(vdst, src0, src1, src2, ir::OperandType::i32());
    } break;
    case eOpcode::V_MIN3_U32: {
      ctx.create<arith::Min3FOp>(vdst, src0, src1, src2, ir::OperandType::i32());
    } break;
    case eOpcode::V_MAX3_F32: {
      ctx.create<arith::Max3FOp>(vdst, src0, src1, src2, ir::OperandType::f32());
    } break;
    case eOpcode::V_MAX3_I32: {
      ctx.create<arith::Max3SIOp>(vdst, src0, src1, src2, ir::OperandType::i32());
    } break;
    case eOpcode::V_MAX3_U32: {
      ctx.create<arith::Min3FOp>(vdst, src0, src1, src2, ir::OperandType::i32());
    } break;
    case eOpcode::V_MED3_F32: {
      ctx.create<arith::Min3FOp>(vdst, src0, src1, src2, ir::OperandType::f32());
    } break;
    case eOpcode::V_MED3_I32: {
      ctx.create<arith::Min3FOp>(vdst, src0, src1, src2, ir::OperandType::i32());
    } break;
    case eOpcode::V_MED3_U32: {
      ctx.create<arith::Min3FOp>(vdst, src0, src1, src2, ir::OperandType::i32());
    } break;
      // case eOpcode::V_SAD_U8: { } break;// todo
      // case eOpcode::V_SAD_HI_U8: {} break;// todo
      // case eOpcode::V_SAD_U16: { } break;// todo
    case eOpcode::V_SAD_U32: {
      auto res_ = ctx.create<arith::SubIOp>(createDst(), src0, src1, ir::OperandType::i32());
      auto res  = ctx.create<arith::AbsoluteOp>(createDst(), res_, ir::OperandType::i32());
      ctx.create<arith::AddIOp>(vdst, res, src2, ir::OperandType::i32());
    } break;
    // case eOpcode::V_CVT_PK_U8_F32: {} break;  // todo
    //  case eOpcode::V_DIV_FIXUP_F32: {} break; // todo
    //  case eOpcode::V_DIV_FIXUP_F64: {} break; // todo
    case eOpcode::V_LSHL_B64: {
      ctx.create<arith::ShiftLUIOp>(vdst, src0, src1, ir::OperandType::i64());
    } break;
    case eOpcode::V_LSHR_B64: {
      ctx.create<arith::ShiftRUIOp>(vdst, src0, src1, ir::OperandType::i64());
    } break;
    case eOpcode::V_ASHR_I64: {
      ctx.create<arith::ShiftRSIOp>(vdst, src0, src1, ir::OperandType::i64());
    } break;
    case eOpcode::V_ADD_F64: {
      ctx.create<arith::AddFOp>(vdst, src0, src1, ir::OperandType::f64());
    } break;
    case eOpcode::V_MUL_F64: {
      ctx.create<arith::MulFOp>(vdst, src0, src1, ir::OperandType::f64());
    } break;
    case eOpcode::V_MIN_F64: {
      ctx.create<arith::MinFOp>(vdst, src0, src1, ir::OperandType::f64());
    } break;
    case eOpcode::V_MAX_F64: {
      ctx.create<arith::MaxFOp>(vdst, src0, src1, ir::OperandType::f64());
    } break;
    case eOpcode::V_LDEXP_F64: {
      ctx.create<arith::LdexpOp>(vdst, src0, src1, ir::OperandType::f64());
    } break;
    case eOpcode::V_MUL_LO_U32: {
      ctx.create<arith::MulIExtendedOp>(vdst, createDst(), src0, src1, ir::OperandType::i32());
    } break;
    case eOpcode::V_MUL_HI_U32: {
      ctx.create<arith::MulIExtendedOp>(createDst(), vdst, src0, src1, ir::OperandType::i32());
    } break;
    case eOpcode::V_MUL_LO_I32: {
      ctx.create<arith::MulIExtendedOp>(vdst, createDst(), src0, src1, ir::OperandType::i32());
    } break;
    case eOpcode::V_MUL_HI_I32: {
      ctx.create<arith::MulIExtendedOp>(createDst(), vdst, src0, src1, ir::OperandType::i32());
    } break;
      // case eOpcode::V_DIV_SCALE_F32: {} break; // todo
      // case eOpcode::V_DIV_SCALE_F64: {} break; // todo
      // case eOpcode::V_DIV_FMAS_F32: {} break; // todo
      // case eOpcode::V_DIV_FMAS_F64: {} break; // todo
      // case eOpcode::V_MSAD_U8: {} break; // todo
      // case eOpcode::V_QSAD_U8: {} break; // todo
      // case eOpcode::V_MQSAD_U8: {} break; // todo
      // case eOpcode::V_TRIG_PREOP_F64: {} break; // todo
      // case eOpcode::V_MQSAD_U32_U8: {} break; // todo
      // case eOpcode::V_MAD_U64_U32: {} break; // todo
      // case eOpcode::V_MAD_I64_I32: {} break;// todo
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }
  return conv(op);
}
} // namespace compiler::frontend::translate