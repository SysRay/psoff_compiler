#include "../../frontend.h"
#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "instruction_builder.h"
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

bool handleVop2(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode, bool extended) {
  using namespace parser;

  eOpcode op;

  OpDst vdst;
  OpDst sdst;
  OpSrc src0, src1, src2;

  if (extended) {
    auto inst  = VOP3(getU64(*pCode));
    auto instS = VOP3_SDST(getU64(*pCode));
    op         = (eOpcode)(OPcodeStart_VOP2 + inst.template get<VOP3::Field::OP>() - OpcodeOffset_VOP2_VOP3);

    auto const vdst_ = eOperandKind::VGPR(inst.template get<VOP3::Field::VDST>());
    auto const src0_ = eOperandKind::create((OperandKind_t)inst.template get<VOP3::Field::SRC0>());
    auto const src1_ = eOperandKind::create((OperandKind_t)inst.template get<VOP3::Field::SRC1>());
    auto const src2_ = eOperandKind::create((OperandKind_t)inst.template get<VOP3::Field::SRC2>());

    auto const sdst_ = eOperandKind::create((OperandKind_t)instS.template get<VOP3_SDST::Field::SDST>());

    auto const           omod   = inst.template get<VOP3::Field::OMOD>();
    std::bitset<3> const negate = inst.template get<VOP3::Field::NEG>();
    std::bitset<3> const abs    = inst.template get<VOP3::Field::ABS>();
    auto const           clamp  = inst.template get<VOP3::Field::CLAMP>();

    if (isSDST(op)) {
      src0 = OpSrc(src0_, negate[0], false);
      src1 = OpSrc(src1_, negate[1], false);
      src2 = OpSrc(src2_, negate[2], false);
      vdst = OpDst(vdst_, omod, false, false);
      sdst = OpDst(sdst_, omod, false, false);
    } else {
      src0 = OpSrc(src0_, negate[0], abs[0]);
      src1 = OpSrc(src1_, negate[1], abs[1]);
      src2 = OpSrc(src2_, negate[2], abs[2]);
      vdst = OpDst(vdst_, omod, clamp, false);
    }
    *pCode += 1;
  } else {
    auto inst = VOP2(**pCode);
    op        = (eOpcode)(OPcodeStart_VOP2 + inst.template get<VOP2::Field::OP>());

    vdst = OpDst(eOperandKind::VGPR(inst.template get<VOP2::Field::VDST>()));
    sdst = OpDst(eOperandKind::VCC());
    src0 = OpSrc(eOperandKind::create((OperandKind_t)inst.template get<VOP2::Field::SRC0>()));
    src1 = OpSrc(eOperandKind::VGPR(inst.template get<VOP2::Field::VSRC1>()));
    src2 = OpSrc(eOperandKind::VCC());
    if (src0.kind.isLiteral() || src1.kind.isLiteral()) {
      *pCode += 1;
      builder.createVirtualInst(create::literalOp(**pCode));
    }
  }

  *pCode += 1;

  switch (op) {
    case eOpcode::V_CNDMASK_B32: {
      builder.createVirtualInst(create::selectOp(vdst, src2, src0, src1, ir::OperandType::f32()));
    } break;
    case eOpcode::V_ADD_F32: {
      builder.createVirtualInst(create::addFOp(vdst, src0, src1, ir::OperandType::f32()));
    } break;
    case eOpcode::V_SUB_F32: {
      builder.createVirtualInst(create::subFOp(vdst, src0, src1, ir::OperandType::f32()));
    } break;
    case eOpcode::V_SUBREV_F32: {
      builder.createVirtualInst(create::subFOp(vdst, src1, src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_MAC_LEGACY_F32: {
      builder.createVirtualInst(create::fmaFOp(OpDst(vdst.kind), src0, src1, OpSrc(vdst.kind, src2.flags), ir::OperandType::f32()));
      builder.createVirtualInst(create::clampFZeroOp(vdst, OpSrc(vdst.kind), ir::OperandType::f32()));
    } break;
    case eOpcode::V_MUL_LEGACY_F32: {
      builder.createVirtualInst(create::mulFOp(OpDst(vdst.kind), src0, src1, ir::OperandType::f32()));
      builder.createVirtualInst(create::clampFZeroOp(vdst, OpSrc(vdst.kind), ir::OperandType::f32()));
    } break;
    case eOpcode::V_MUL_F32: {
      builder.createVirtualInst(create::mulFOp(vdst, src0, src1, ir::OperandType::f32()));
    } break;
    case eOpcode::V_MUL_I32_I24: {
      builder.createVirtualInst(create::bitSIExtractOp(OpDst(eOperandKind::Temp0()), src0, OpSrc(eOperandKind::createImm(0)),
                                                       OpSrc(eOperandKind::createImm(24)), ir::OperandType::i32()));
      builder.createVirtualInst(create::bitSIExtractOp(OpDst(eOperandKind::Temp1()), src1, OpSrc(eOperandKind::createImm(0)),
                                                       OpSrc(eOperandKind::createImm(24)), ir::OperandType::i32()));
      builder.createVirtualInst(create::mulIOp(vdst, OpSrc(eOperandKind::Temp0()), OpSrc(eOperandKind::Temp1()), ir::OperandType::i32()));
    } break;
    case eOpcode::V_MUL_HI_I32_I24: {
      builder.createVirtualInst(create::bitSIExtractOp(OpDst(eOperandKind::Temp0()), src0, OpSrc(eOperandKind::createImm(0)),
                                                       OpSrc(eOperandKind::createImm(24)), ir::OperandType::i32()));
      builder.createVirtualInst(create::bitSIExtractOp(OpDst(eOperandKind::Temp1()), src1, OpSrc(eOperandKind::createImm(0)),
                                                       OpSrc(eOperandKind::createImm(24)), ir::OperandType::i32()));
      builder.createVirtualInst(
          create::mulIExtendedOp(OpDst(eOperandKind::Temp0()), vdst, OpSrc(eOperandKind::Temp0()), OpSrc(eOperandKind::Temp1()), ir::OperandType::i32()));
    } break;
    case eOpcode::V_MUL_U32_U24: {
      builder.createVirtualInst(create::bitUIExtractOp(OpDst(eOperandKind::Temp0()), src0, OpSrc(eOperandKind::createImm(0)),
                                                       OpSrc(eOperandKind::createImm(24)), ir::OperandType::i32()));
      builder.createVirtualInst(create::bitUIExtractOp(OpDst(eOperandKind::Temp1()), src1, OpSrc(eOperandKind::createImm(0)),
                                                       OpSrc(eOperandKind::createImm(24)), ir::OperandType::i32()));
      builder.createVirtualInst(create::mulIOp(vdst, OpSrc(eOperandKind::Temp0()), OpSrc(eOperandKind::Temp1()), ir::OperandType::i32()));
    } break;
    case eOpcode::V_MUL_HI_U32_U24: {
      builder.createVirtualInst(create::bitUIExtractOp(OpDst(eOperandKind::Temp0()), src0, OpSrc(eOperandKind::createImm(0)),
                                                       OpSrc(eOperandKind::createImm(24)), ir::OperandType::i32()));
      builder.createVirtualInst(create::bitUIExtractOp(OpDst(eOperandKind::Temp1()), src1, OpSrc(eOperandKind::createImm(0)),
                                                       OpSrc(eOperandKind::createImm(24)), ir::OperandType::i32()));
      builder.createVirtualInst(
          create::mulIExtendedOp(OpDst(eOperandKind::Temp0()), vdst, OpSrc(eOperandKind::Temp0()), OpSrc(eOperandKind::Temp1()), ir::OperandType::i32()));
    } break;
    case eOpcode::V_MIN_LEGACY_F32: {
      builder.createVirtualInst(create::minNOp(vdst, src1, src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_MAX_LEGACY_F32: {
      builder.createVirtualInst(create::maxNOp(vdst, src1, src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_MIN_F32: {
      builder.createVirtualInst(create::minFOp(vdst, src1, src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_MAX_F32: {
      builder.createVirtualInst(create::maxFOp(vdst, src1, src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_MIN_I32: {
      builder.createVirtualInst(create::minSIOp(vdst, src1, src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_MAX_I32: {
      builder.createVirtualInst(create::maxSIOp(vdst, src1, src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_MIN_U32: {
      builder.createVirtualInst(create::minUIOp(vdst, src1, src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_MAX_U32: {
      builder.createVirtualInst(create::maxUIOp(vdst, src1, src0, ir::OperandType::f32()));
    } break;
    case eOpcode::V_LSHR_B32: {
      builder.createVirtualInst(create::shiftRUIOp(vdst, src0, src1, ir::OperandType::i32()));
    } break;
    case eOpcode::V_LSHRREV_B32: {
      builder.createVirtualInst(create::shiftRUIOp(vdst, src1, src0, ir::OperandType::i32()));
    } break;
    case eOpcode::V_ASHR_I32: {
      builder.createVirtualInst(create::shiftRSIOp(vdst, src0, src1, ir::OperandType::i32()));
    } break;
    case eOpcode::V_ASHRREV_I32: {
      builder.createVirtualInst(create::shiftRSIOp(vdst, src1, src0, ir::OperandType::i32()));
    } break;
    case eOpcode::V_LSHL_B32: {
      builder.createVirtualInst(create::shiftLUIOp(vdst, src0, src1, ir::OperandType::i32()));
    } break;
    case eOpcode::V_LSHLREV_B32: {
      builder.createVirtualInst(create::shiftLUIOp(vdst, src1, src0, ir::OperandType::i32()));
    } break;
    case eOpcode::V_AND_B32: {
      builder.createVirtualInst(create::bitAndOp(vdst, src0, src1, ir::OperandType::i32()));
    } break;
    case eOpcode::V_OR_B32: {
      builder.createVirtualInst(create::bitOrOp(vdst, src0, src1, ir::OperandType::i32()));
    } break;
    case eOpcode::V_XOR_B32: {
      builder.createVirtualInst(create::bitXorOp(vdst, src0, src1, ir::OperandType::i32()));
    } break;
    case eOpcode::V_BFM_B32: {
      builder.createVirtualInst(create::bitFieldMaskOp(vdst, src0, src1, ir::OperandType::i32()));
    } break;
    case eOpcode::V_MAC_F32: {
      builder.createVirtualInst(create::fmaFOp(vdst, src0, src1, OpSrc(vdst.kind, src2.flags), ir::OperandType::f32()));
    } break;
    case eOpcode::V_MADMK_F32: {
      builder.createVirtualInst(create::literalOp(**pCode));
      *pCode += 1;
      builder.createVirtualInst(create::fmaFOp(vdst, src0, OpSrc(eOperandKind::Literal(), src2.flags), src1, ir::OperandType::f32()));
    } break;
    case eOpcode::V_MADAK_F32: {
      builder.createVirtualInst(create::literalOp(**pCode));
      *pCode += 1;
      builder.createVirtualInst(create::fmaFOp(vdst, src0, src1, OpSrc(eOperandKind::Literal(), src2.flags), ir::OperandType::f32()));
    } break;
    case eOpcode::V_BCNT_U32_B32: {
      builder.createVirtualInst(create::bitCountOp(OpDst(vdst.kind), src0, ir::OperandType::i32()));
      builder.createVirtualInst(create::addIOp(vdst, OpSrc(vdst.kind), src1, ir::OperandType::i32()));
    } break;
    // case eOpcode::V_MBCNT_LO_U32_B32: {} break; // todo
    // case eOpcode::V_MBCNT_HI_U32_B32: {} break; // todo
    case eOpcode::V_ADD_I32: {
      builder.createVirtualInst(create::addIOp(vdst, src0, src1, ir::OperandType::i32()));
      builder.createVirtualInst(create::cmpIOp(sdst, OpSrc(vdst.kind), src0, ir::OperandType::i32(), CmpIPredicate::slt));
    } break;
    case eOpcode::V_SUB_I32: {
      builder.createVirtualInst(create::subIOp(vdst, src0, src1, ir::OperandType::i32()));
      builder.createVirtualInst(create::cmpIOp(sdst, OpSrc(vdst.kind), src0, ir::OperandType::i32(), CmpIPredicate::sgt));
    } break;
    case eOpcode::V_SUBREV_I32: {
      builder.createVirtualInst(create::subIOp(vdst, src1, src0, ir::OperandType::i32()));
      builder.createVirtualInst(create::cmpIOp(sdst, OpSrc(vdst.kind), src1, ir::OperandType::i32(), CmpIPredicate::sgt));
    } break;
    case eOpcode::V_ADDC_U32: {
      builder.createVirtualInst(create::addcIOp(vdst, sdst, src0, src1, src2, ir::OperandType::i32()));
    } break;
    case eOpcode::V_SUBB_U32: {
      builder.createVirtualInst(create::subbIOp(vdst, sdst, src0, src1, src2, ir::OperandType::i32()));
    } break;
    case eOpcode::V_SUBBREV_U32: {
      builder.createVirtualInst(create::subbIOp(vdst, sdst, src1, src0, src2, ir::OperandType::i32()));
    } break;
    case eOpcode::V_LDEXP_F32: {
      builder.createVirtualInst(create::ldexpOp(vdst, src0, src1, ir::OperandType::f32()));
    } break;
    // case eOpcode::V_CVT_PKACCUM_U8_F32: {} break; // does not exist
    case eOpcode::V_CVT_PKNORM_I16_F32: {
      builder.createVirtualInst(create::packSnorm2x16Op(vdst, src0, src1));
    } break;
    case eOpcode::V_CVT_PKNORM_U16_F32: {
      builder.createVirtualInst(create::packUnorm2x16Op(vdst, src0, src1));
    } break;
    case eOpcode::V_CVT_PKRTZ_F16_F32: {
      builder.createVirtualInst(create::packHalf2x16Op(vdst, src0, src1));
    } break;
      // case eOpcode::V_CVT_PK_U16_U32: { } break; // todo
      // case eOpcode::V_CVT_PK_I16_I32: { } break; // todo
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }
  return true;
}
} // namespace compiler::frontend::translate