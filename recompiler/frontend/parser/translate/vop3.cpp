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
bool handleVop3(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = VOP3(getU64(*pCode));
  auto const op   = (parser::eOpcode)(OPcodeStart_VOP3 + inst.template get<VOP3::Field::OP>());

  OpDst vdst;
  OpSrc src0, src1, src2;
  {
    auto const vdst_ = eOperandKind::VGPR(inst.template get<VOP3::Field::VDST>());
    auto const src0_ = eOperandKind((eOperandKind_t)inst.template get<VOP3::Field::SRC0>());
    auto const src1_ = eOperandKind((eOperandKind_t)inst.template get<VOP3::Field::SRC1>());
    auto const src2_ = eOperandKind((eOperandKind_t)inst.template get<VOP3::Field::SRC2>());

    auto const           omod   = inst.template get<VOP3::Field::OMOD>();
    std::bitset<3> const negate = inst.template get<VOP3::Field::NEG>();
    std::bitset<3> const abs    = inst.template get<VOP3::Field::ABS>();
    auto const           clamp  = inst.template get<VOP3::Field::CLAMP>();

    src0 = OpSrc(src0_, negate[0], abs[0]);
    src1 = OpSrc(src1_, negate[1], abs[1]);
    src2 = OpSrc(src2_, negate[2], abs[2]);
    vdst = OpDst(vdst_, omod, clamp, false);
  }

  *pCode += 2;

  switch (op) {
    case eOpcode::V_MAD_LEGACY_F32: {
      builder.createVirtualInst(create::fmaFOp(OpDst(vdst.kind), src0, src1, src2, ir::OperandType::f32()));
      builder.createVirtualInst(create::clampFZeroOp(vdst, OpSrc(vdst.kind), ir::OperandType::f32()));
    } break;
    case eOpcode::V_MAD_F32: {
      builder.createVirtualInst(create::fmaFOp(vdst, src0, src1, src2, ir::OperandType::f32()));
    } break;
    case eOpcode::V_MAD_I32_I24: {
      builder.createVirtualInst(create::bitSIExtractOp(OpDst(eOperandKind::Temp0()), src0, OpSrc(eOperandKind::createImm(0)),
                                                       OpSrc(eOperandKind::createImm(24)), ir::OperandType::i32()));
      builder.createVirtualInst(create::bitSIExtractOp(OpDst(eOperandKind::Temp1()), src1, OpSrc(eOperandKind::createImm(0)),
                                                       OpSrc(eOperandKind::createImm(24)), ir::OperandType::i32()));
      builder.createVirtualInst(create::fmaIOp(vdst, OpSrc(eOperandKind::Temp0()), OpSrc(eOperandKind::Temp0()), src2, ir::OperandType::i32()));
    } break;
    case eOpcode::V_MAD_U32_U24: {
      builder.createVirtualInst(create::bitUIExtractOp(OpDst(eOperandKind::Temp0()), src0, OpSrc(eOperandKind::createImm(0)),
                                                       OpSrc(eOperandKind::createImm(24)), ir::OperandType::i32()));
      builder.createVirtualInst(create::bitUIExtractOp(OpDst(eOperandKind::Temp1()), src1, OpSrc(eOperandKind::createImm(0)),
                                                       OpSrc(eOperandKind::createImm(24)), ir::OperandType::i32()));
      builder.createVirtualInst(create::fmaIOp(vdst, OpSrc(eOperandKind::Temp0()), OpSrc(eOperandKind::Temp0()), src2, ir::OperandType::i32()));
    } break;
      // case eOpcode::V_CUBEID_F32: {} break;  // todo
      // case eOpcode::V_CUBESC_F32: {} break;  // todo
      // case eOpcode::V_CUBETC_F32: {} break; // todo
      // case eOpcode::V_CUBEMA_F32: {} break; // todo
    case eOpcode::V_BFE_U32: {
      builder.createVirtualInst(create::bitUIExtractOp(OpDst(eOperandKind::Temp0()), src0, src1, src2, ir::OperandType::i32()));
    } break;
    case eOpcode::V_BFE_I32: {
      builder.createVirtualInst(create::bitSIExtractOp(OpDst(eOperandKind::Temp0()), src0, src1, src2, ir::OperandType::i32()));
    } break;
    case eOpcode::V_BFI_B32: {
      builder.createVirtualInst(create::bitFieldInsertOp(OpDst(eOperandKind::Temp0()), src0, src1, src2, ir::OperandType::i32()));
    } break;
    case eOpcode::V_FMA_F32: {
      builder.createVirtualInst(create::fmaFOp(vdst, src0, src1, src2, ir::OperandType::f32()));
    } break;
    case eOpcode::V_FMA_F64: {
      builder.createVirtualInst(create::fmaFOp(vdst, src0, src1, src2, ir::OperandType::f64()));
    } break;
    // case eOpcode::V_LERP_U8: {} break; // todo
    // case eOpcode::V_ALIGNBIT_B32: {} break; // todo
    // case eOpcode::V_ALIGNBYTE_B32: {} break; // todo
    //  case eOpcode::V_MULLIT_F32: {} break; // todo
    case eOpcode::V_MIN3_F32: {
      builder.createVirtualInst(create::min3FOp(vdst, src0, src1, src2, ir::OperandType::f32()));
    } break;
    case eOpcode::V_MIN3_I32: {
      builder.createVirtualInst(create::min3SIOp(vdst, src0, src1, src2, ir::OperandType::i32()));
    } break;
    case eOpcode::V_MIN3_U32: {
      builder.createVirtualInst(create::min3FOp(vdst, src0, src1, src2, ir::OperandType::i32()));
    } break;
    case eOpcode::V_MAX3_F32: {
      builder.createVirtualInst(create::max3FOp(vdst, src0, src1, src2, ir::OperandType::f32()));
    } break;
    case eOpcode::V_MAX3_I32: {
      builder.createVirtualInst(create::max3SIOp(vdst, src0, src1, src2, ir::OperandType::i32()));
    } break;
    case eOpcode::V_MAX3_U32: {
      builder.createVirtualInst(create::min3FOp(vdst, src0, src1, src2, ir::OperandType::i32()));
    } break;
    case eOpcode::V_MED3_F32: {
      builder.createVirtualInst(create::min3FOp(vdst, src0, src1, src2, ir::OperandType::f32()));
    } break;
    case eOpcode::V_MED3_I32: {
      builder.createVirtualInst(create::min3FOp(vdst, src0, src1, src2, ir::OperandType::i32()));
    } break;
    case eOpcode::V_MED3_U32: {
      builder.createVirtualInst(create::min3FOp(vdst, src0, src1, src2, ir::OperandType::i32()));
    } break;
      // case eOpcode::V_SAD_U8: { } break;// todo
      // case eOpcode::V_SAD_HI_U8: {} break;// todo
      // case eOpcode::V_SAD_U16: { } break;// todo
    case eOpcode::V_SAD_U32: {
      builder.createVirtualInst(create::subIOp(OpDst(vdst.kind), src0, src1, ir::OperandType::i32()));
      builder.createVirtualInst(create::addIOp(vdst, OpSrc(vdst.kind, false, true), src2, ir::OperandType::i32()));
    } break;
    // case eOpcode::V_CVT_PK_U8_F32: {} break;  // todo
    //  case eOpcode::V_DIV_FIXUP_F32: {} break; // todo
    //  case eOpcode::V_DIV_FIXUP_F64: {} break; // todo
    case eOpcode::V_LSHL_B64: {
      builder.createVirtualInst(create::shiftLUIOp(vdst, src0, src1, ir::OperandType::i64()));
    } break;
    case eOpcode::V_LSHR_B64: {
      builder.createVirtualInst(create::shiftRUIOp(vdst, src0, src1, ir::OperandType::i64()));
    } break;
    case eOpcode::V_ASHR_I64: {
      builder.createVirtualInst(create::shiftRSIOp(vdst, src0, src1, ir::OperandType::i64()));
    } break;
    case eOpcode::V_ADD_F64: {
      builder.createVirtualInst(create::addFOp(vdst, src0, src1, ir::OperandType::f64()));
    } break;
    case eOpcode::V_MUL_F64: {
      builder.createVirtualInst(create::mulFOp(vdst, src0, src1, ir::OperandType::f64()));
    } break;
    case eOpcode::V_MIN_F64: {
      builder.createVirtualInst(create::minFOp(vdst, src0, src1, ir::OperandType::f64()));
    } break;
    case eOpcode::V_MAX_F64: {
      builder.createVirtualInst(create::maxFOp(vdst, src0, src1, ir::OperandType::f64()));
    } break;
    case eOpcode::V_LDEXP_F64: {
      builder.createVirtualInst(create::ldexpOp(vdst, src0, src1, ir::OperandType::f64()));
    } break;
    case eOpcode::V_MUL_LO_U32: {
      builder.createVirtualInst(create::mulIExtendedOp(vdst, OpDst(eOperandKind::Temp0()), src0, src1, ir::OperandType::i32()));
    } break;
    case eOpcode::V_MUL_HI_U32: {
      builder.createVirtualInst(create::mulIExtendedOp(OpDst(eOperandKind::Temp0()), vdst, src0, src1, ir::OperandType::i32()));
    } break;
    case eOpcode::V_MUL_LO_I32: {
      builder.createVirtualInst(create::mulIExtendedOp(vdst, OpDst(eOperandKind::Temp0()), src0, src1, ir::OperandType::i32()));
    } break;
    case eOpcode::V_MUL_HI_I32: {
      builder.createVirtualInst(create::mulIExtendedOp(OpDst(eOperandKind::Temp0()), vdst, src0, src1, ir::OperandType::i32()));
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
  return true;
}
} // namespace compiler::frontend::translate