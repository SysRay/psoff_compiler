#include "../debug_strings.h"
#include "../gfx/encoding_types.h"
#include "../operations.h"
#include "../parser.h"
#include "opcodes_table.h"

#include <bitset>
#include <format>
#include <stdexcept>

// mlir
#include "mlir/custom.h"

#include <mlir/Dialect/Arith/IR/Arith.h>

namespace compiler::frontend {

uint8_t Parser::handleVop1(CodeBlock& cb, pc_t pc, uint32_t const* pCode, bool extended) {
  eOpcode op;

  OpDst vdst;
  OpSrc src0;

  uint8_t size = sizeof(uint32_t);
  if (extended) {
    auto inst = VOP3(getU64(pCode));
    op        = (eOpcode)(OPcodeStart_VOP1 + inst.template get<VOP3::Field::OP>() - OpcodeOffset_VOP1_VOP3);

    auto const           omod   = inst.template get<VOP3::Field::OMOD>();
    std::bitset<3> const negate = inst.template get<VOP3::Field::NEG>();
    std::bitset<3> const abs    = inst.template get<VOP3::Field::ABS>();
    auto const           clamp  = inst.template get<VOP3::Field::CLAMP>();

    vdst = OpDst(eOperandKind::VGPR(inst.template get<VOP3::Field::VDST>()), omod, clamp);
    src0 = OpSrc(eOperandKind((eOperandKind_t)inst.template get<VOP3::Field::SRC0>()), negate[0], abs[0]);

    size = sizeof(uint64_t);
  } else {
    auto inst = VOP1(*pCode);
    op        = (eOpcode)(OPcodeStart_VOP1 + inst.template get<VOP1::Field::OP>());

    vdst = OpDst(eOperandKind::VGPR(inst.template get<VOP1::Field::VDST>()));
    src0 = OpSrc(eOperandKind((eOperandKind_t)inst.template get<VOP1::Field::SRC0>()));

    if (src0.kind.isLiteral()) {
      size = sizeof(uint64_t);
    }
  }

  switch (op) {
    case eOpcode::V_NOP: {
    } break;
    case eOpcode::V_MOV_B32: {
      parse<op::MoveOp>(vdst, src0, types().i32());
    } break;
    // case eOpcode::V_READFIRSTLANE_B32: { // todo
    // } break;
    case eOpcode::V_CVT_I32_F64: {
      parse<op::ConvertFtoSIOp>(vdst, types().i32(), src0, types().f64());
    } break;
    case eOpcode::V_CVT_F64_I32: {
      parse<op::ConvertSItoFOp>(vdst, types().f64(), src0, types().i32());
    } break;
    case eOpcode::V_CVT_F32_I32: {
      parse<op::ConvertSItoFOp>(vdst, types().f32(), src0, types().i32());
    } break;
    case eOpcode::V_CVT_F32_U32: {
      parse<op::ConvertUItoFOp>(vdst, types().f32(), src0, types().i32());
    } break;
    case eOpcode::V_CVT_U32_F32: {
      parse<op::ConvertFtoUIOp>(vdst, types().i32(), src0, types().f32());
    } break;
    case eOpcode::V_CVT_I32_F32: {
      parse<op::ConvertFtoSIOp>(vdst, types().i32(), src0, types().f32());
    } break;
      // case eOpcode::V_MOV_FED_B32: break; // Does not exist
    case eOpcode::V_CVT_F16_F32: {
      parse<op::ConvertFtoFOp>(vdst, types().f16(), src0, types().f32());
    } break;
    case eOpcode::V_CVT_F32_F16: {
      parse<op::ConvertFtoFOp>(vdst, types().f32(), src0, types().f16());
    } break;
    case eOpcode::V_CVT_RPI_I32_F32: {
      parse<op::ConvertFtoSIOp>(vdst, types().i32(), src0, types().f32(), op::ConvertFtoSIOp::eMode::RPI);
    } break;
    case eOpcode::V_CVT_FLR_I32_F32: {
      parse<op::ConvertFtoSIOp>(vdst, types().i32(), src0, types().f32(), op::ConvertFtoSIOp::eMode::Floor);
    } break;
    case eOpcode::V_CVT_OFF_F32_I4: {
      parse<op::ConvertSubPixelOffsetToFOp>(vdst, types().f32(), src0, types().i32());
    } break;
    case eOpcode::V_CVT_F32_F64: {
      parse<op::ConvertFtoFOp>(vdst, types().f32(), src0, types().f64());
    } break;
    case eOpcode::V_CVT_F64_F32: {
      parse<op::ConvertFtoFOp>(vdst, types().f64(), src0, types().f32());
    } break;
    case eOpcode::V_CVT_F32_UBYTE0: {
      parse<op::ConvertByteToFOp>(vdst, types().f32(), src0, types().i32(), 0);
    } break;
    case eOpcode::V_CVT_F32_UBYTE1: {
      parse<op::ConvertByteToFOp>(vdst, types().f32(), src0, types().i32(), 1);
    } break;
    case eOpcode::V_CVT_F32_UBYTE2: {
      parse<op::ConvertByteToFOp>(vdst, types().f32(), src0, types().i32(), 2);
    } break;
    case eOpcode::V_CVT_F32_UBYTE3: {
      parse<op::ConvertByteToFOp>(vdst, types().f32(), src0, types().i32(), 3);
    } break;
    case eOpcode::V_CVT_U32_F64: {
      parse<op::ConvertFtoUIOp>(vdst, types().i32(), src0, types().f64());
    } break;
    case eOpcode::V_CVT_F64_U32: {
      parse<op::ConvertUItoFOp>(vdst, types().f64(), src0, types().i32());
    } break;
    case eOpcode::V_TRUNC_F64: {
      parse<op::TruncOp>(vdst, src0, types().f64());
    } break;
    case eOpcode::V_CEIL_F64: {
      parse<op::CeilOp>(vdst, src0, types().f64());
    } break;
    case eOpcode::V_RNDNE_F64: {
      parse<op::RoundEvenOp>(vdst, src0, types().f64());
    } break;
    case eOpcode::V_FLOOR_F64: {
      parse<op::FloorOp>(vdst, src0, types().f64());
    } break;
    case eOpcode::V_FRACT_F32: {
      parse<op::FloorOp>(vdst, src0, types().f32());
    } break;
    case eOpcode::V_TRUNC_F32: {
      parse<op::TruncOp>(vdst, src0, types().f32());
    } break;
    case eOpcode::V_CEIL_F32: {
      parse<op::CeilOp>(vdst, src0, types().f32());
    } break;
    case eOpcode::V_RNDNE_F32: {
      parse<op::RoundEvenOp>(vdst, src0, types().f32());
    } break;
    case eOpcode::V_FLOOR_F32: {
      parse<op::FloorOp>(vdst, src0, types().f32());
    } break;
    case eOpcode::V_EXP_F32: {
      parse<op::Exp2Op>(vdst, src0, types().f32());
    } break;
    case eOpcode::V_LOG_CLAMP_F32: {
      vdst.flags |= OpFlags::eClampFMinMax;
      parse<op::Log2Op>(vdst, src0, types().f32());
    } break;
    case eOpcode::V_LOG_F32: {
      parse<op::Log2Op>(vdst, src0, types().f32());
    } break;
    case eOpcode::V_RCP_CLAMP_F32: {
      vdst.flags |= OpFlags::eClampFMinMax;
      parse<op::RcpOp>(vdst, src0, types().f32());
    } break;
    case eOpcode::V_RCP_LEGACY_F32: {
      vdst.flags |= OpFlags::eClampFZero;
      parse<op::RcpOp>(vdst, src0, types().f32());
    } break;
    case eOpcode::V_RCP_F32: {
      parse<op::RcpOp>(vdst, src0, types().f32());
    } break;
    case eOpcode::V_RCP_IFLAG_F32: {
      parse<op::RcpOp>(vdst, src0, types().f32());
    } break;
    case eOpcode::V_RSQ_CLAMP_F32: {
      vdst.flags |= OpFlags::eClampFMinMax;
      parse<op::RsqOp>(vdst, src0, types().f32());
    } break;
    case eOpcode::V_RSQ_LEGACY_F32: {
      vdst.flags |= OpFlags::eClampFZero;
      parse<op::RsqOp>(vdst, src0, types().f32());
    } break;
    case eOpcode::V_RSQ_F32: {
      parse<op::RsqOp>(vdst, src0, types().f32());
    } break;
    case eOpcode::V_RCP_F64: {
      parse<op::RcpOp>(vdst, src0, types().f64());
    } break;
    case eOpcode::V_RCP_CLAMP_F64: {
      vdst.flags |= OpFlags::eClampFMinMax;
      parse<op::RcpOp>(vdst, src0, types().f64());
    } break;
    case eOpcode::V_RSQ_F64: {
      parse<op::RsqOp>(vdst, src0, types().f64());
    } break;
    case eOpcode::V_RSQ_CLAMP_F64: {
      vdst.flags |= OpFlags::eClampFMinMax;
      parse<op::RsqOp>(vdst, src0, types().f64());
    } break;
    case eOpcode::V_SQRT_F32: {
      parse<op::SqrtOp>(vdst, src0, types().f32());
    } break;
    case eOpcode::V_SQRT_F64: {
      parse<op::SqrtOp>(vdst, src0, types().f64());
    } break;
    case eOpcode::V_SIN_F32: {
      parse<op::SinOp>(vdst, src0, types().f32());
    } break;
    case eOpcode::V_COS_F32: {
      parse<op::CosOp>(vdst, src0, types().f32());
    } break;
    case eOpcode::V_NOT_B32: {
      src0.setNot();
      parse<op::MoveOp>(vdst, src0, types().i32());
    } break;
    case eOpcode::V_BFREV_B32: {
      parse<op::BrevOp>(vdst, src0, types().i32());
    } break;
    case eOpcode::V_FFBH_U32: {
      parse<op::FindFirstLsbBitOp>(vdst, src0, types().i32());
    } break;
    case eOpcode::V_FFBL_B32: {
      src0.setNot();
      parse<op::FindFirstLsbBitOp>(vdst, src0, types().i32());
    } break;
    case eOpcode::V_FFBH_I32: {
      parse<op::FindFirstSMsbBitOp>(vdst, src0, types().i32());
    } break;
    case eOpcode::V_FREXP_EXP_I32_F64: {
      parse<op::GetExpOp>(vdst, types().i32(), src0, types().f64());
    } break;
    case eOpcode::V_FREXP_MANT_F64: {
      parse<op::GetMantOp>(vdst, types().i32(), src0, types().f64());
    } break;
    case eOpcode::V_FRACT_F64: {
      parse<op::FractOp>(vdst, src0, types().f64());
    } break;
    case eOpcode::V_FREXP_EXP_I32_F32: {
      parse<op::GetExpOp>(vdst, types().i32(), src0, types().f32());
    } break;
    case eOpcode::V_FREXP_MANT_F32: {
      parse<op::GetMantOp>(vdst, types().i32(), src0, types().f32());
    } break;
    // case eOpcode::V_CLREXCP: {} break; // does not exist
    // todo
    // case eOpcode::V_MOVRELD_B32: {
    // } break;
    // case eOpcode::V_MOVRELS_B32: {
    // } break;
    // case eOpcode::V_MOVRELSD_B32: {
    // } break;
    // case eOpcode::V_LOG_LEGACY_F32: {} break; // does not exist
    // case eOpcode::V_EXP_LEGACY_F32: {} break; // does not exist
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }
  return size;
}

uint8_t Parser::handleVop2(CodeBlock& cb, pc_t pc, uint32_t const* pCode, bool extended) {
  eOpcode op;

  OpDst vdst, sdst = OpDst(eOperandKind::VCC());
  OpSrc src0, src1, src2;

  auto isSDST = [op] {
    return op == eOpcode::V_ADD_I32 || op == eOpcode::V_SUB_I32 || op == eOpcode::V_SUBREV_I32 || op == eOpcode::V_ADDC_U32 || op == eOpcode::V_SUBB_U32 ||
           op == eOpcode::V_SUBBREV_U32;
  };

  uint8_t size = sizeof(uint32_t);

  if (extended) {
    auto inst = VOP3(getU64(pCode));
    op        = (eOpcode)(OPcodeStart_VOP2 + inst.template get<VOP3::Field::OP>() - OpcodeOffset_VOP2_VOP3);

    auto const           omod   = inst.template get<VOP3::Field::OMOD>();
    std::bitset<3> const negate = inst.template get<VOP3::Field::NEG>();
    std::bitset<3>       abs    = inst.template get<VOP3::Field::ABS>();
    auto                 clamp  = inst.template get<VOP3::Field::CLAMP>();

    if (isSDST()) {
      auto instS = VOP3_SDST(getU64(pCode));
      sdst       = OpDst(eOperandKind((eOperandKind_t)instS.template get<VOP3_SDST::Field::SDST>()));
      abs        = 0;
      clamp      = false;
    }

    vdst = OpDst(eOperandKind::VGPR(inst.template get<VOP3::Field::VDST>()), omod, clamp);
    src0 = OpSrc(eOperandKind((eOperandKind_t)inst.template get<VOP3::Field::SRC0>()), negate[0], abs[0]);
    src1 = OpSrc(eOperandKind((eOperandKind_t)inst.template get<VOP3::Field::SRC1>()), negate[1], abs[1]);
    src2 = OpSrc(eOperandKind((eOperandKind_t)inst.template get<VOP3::Field::SRC2>()), negate[2], abs[2]);

    size = sizeof(uint64_t);
  } else {
    auto inst = VOP2(*pCode);
    op        = (eOpcode)(OPcodeStart_VOP2 + inst.template get<VOP2::Field::OP>());

    vdst = OpDst(eOperandKind::VGPR(inst.template get<VOP2::Field::VDST>()));
    src0 = OpSrc(eOperandKind((eOperandKind_t)inst.template get<VOP2::Field::SRC0>()));
    src1 = OpSrc(eOperandKind(eOperandKind::VGPR(inst.template get<VOP2::Field::VSRC1>())));
    src2 = OpSrc(eOperandKind::VCC());

    if (src0.kind.isLiteral() || src1.kind.isLiteral()) {
      size = sizeof(uint64_t);
    }
  }

  switch (op) {
    case eOpcode::V_CNDMASK_B32: {
      parse<op::CMoveOp>(vdst, src2, src0, src1, types().i32());
    } break;
    // case eOpcode::V_READLANE_B32: { // todo
    // } break;
    // case eOpcode::V_WRITELANE_B32: {
    // } break;
    case eOpcode::V_ADD_F32: {
      parse<op::AddFOp>(vdst, src0, src1, types().f32());
    } break;
    case eOpcode::V_SUB_F32: {
      parse<op::SubFOp>(vdst, src0, src1, types().f32());
    } break;
    case eOpcode::V_SUBREV_F32: {
      parse<op::SubFOp>(vdst, src1, src0, types().f32());
    } break;
    case eOpcode::V_MAC_LEGACY_F32: {
      parse<op::FmaOp>(vdst, src0, src1, OpSrc(vdst.kind, src2.flags), types().f32()); // todo
    } break;
    case eOpcode::V_MUL_LEGACY_F32: {
      parse<op::MulFOp>(vdst, src0, src1, types().f32()); // todo
    } break;
    case eOpcode::V_MUL_F32: {
      parse<op::MulFOp>(vdst, src0, src1, types().f32());
    } break;
    case eOpcode::V_MUL_I32_I24: {
      parse<op::Mul24IOp>(vdst, src0, src1, types().i32(), true, false);
    } break;
    case eOpcode::V_MUL_HI_I32_I24: {
      parse<op::Mul24IOp>(vdst, src0, src1, types().i32(), true, true);
    } break;
    case eOpcode::V_MUL_U32_U24: {
      parse<op::Mul24IOp>(vdst, src0, src1, types().i32(), false, false);
    } break;
    case eOpcode::V_MUL_HI_U32_U24: {
      parse<op::Mul24IOp>(vdst, src0, src1, types().i32(), false, true);
    } break;
    case eOpcode::V_MIN_LEGACY_F32: {
      parse<op::MinFOp>(vdst, src0, src1, types().f32(), true);
    } break;
    case eOpcode::V_MAX_LEGACY_F32: {
      parse<op::MaxFOp>(vdst, src0, src1, types().f32(), true);
    } break;
    case eOpcode::V_MIN_F32: {
      parse<op::MinFOp>(vdst, src0, src1, types().f32());
    } break;
    case eOpcode::V_MAX_F32: {
      parse<op::MaxFOp>(vdst, src0, src1, types().f32());
    } break;
    case eOpcode::V_MIN_I32: {
      parse<op::MinSIOp>(vdst, OpDst(eOperandKind::Unset()), src0, src1, types().i32());
    } break;
    case eOpcode::V_MAX_I32: {
      parse<op::MaxSIOp>(vdst, OpDst(eOperandKind::Unset()), src0, src1, types().i32());
    } break;
    case eOpcode::V_MIN_U32: {
      parse<op::MinUIOp>(vdst, OpDst(eOperandKind::Unset()), src0, src1, types().i32());
    } break;
    case eOpcode::V_MAX_U32: {
      parse<op::MaxUIOp>(vdst, OpDst(eOperandKind::Unset()), src0, src1, types().i32());
    } break;
    case eOpcode::V_LSHR_B32: {
      parse<op::LSHROp>(vdst, OpDst(eOperandKind::Unset()), src0, src1, types().i32());
    } break;
    case eOpcode::V_LSHRREV_B32: {
      parse<op::LSHROp>(vdst, OpDst(eOperandKind::Unset()), src1, src0, types().i32());
    } break;
    case eOpcode::V_ASHR_I32: {
      parse<op::ASHROp>(vdst, OpDst(eOperandKind::Unset()), src0, src1, types().i32());
    } break;
    case eOpcode::V_ASHRREV_I32: {
      parse<op::ASHROp>(vdst, OpDst(eOperandKind::Unset()), src1, src0, types().i32());
    } break;
    case eOpcode::V_LSHL_B32: {
      parse<op::LSHLOp>(vdst, OpDst(eOperandKind::Unset()), src0, src1, types().i32());
    } break;
    case eOpcode::V_LSHLREV_B32: {
      parse<op::LSHLOp>(vdst, OpDst(eOperandKind::Unset()), src1, src0, types().i32());
    } break;
    case eOpcode::V_AND_B32: {
      parse<op::AndIOp>(vdst, OpDst(eOperandKind::Unset()), src0, src1, types().i32());
    } break;
    case eOpcode::V_OR_B32: {
      parse<op::OrIOp>(vdst, OpDst(eOperandKind::Unset()), src0, src1, types().i32());
    } break;
    case eOpcode::V_XOR_B32: {
      parse<op::XorIOp>(vdst, OpDst(eOperandKind::Unset()), src0, src1, types().i32());
    } break;
    case eOpcode::V_BFM_B32: {
      parse<op::BitfieldMaskOp>(vdst, src0, src1, types().i32());
    } break;
    case eOpcode::V_MAC_F32: {
      parse<op::FmaOp>(vdst, src0, src1, OpSrc(vdst.kind, src2.flags), types().f32());
    } break;
    case eOpcode::V_MADMK_F32: {
      parse<op::FmaOp>(vdst, src0, src2, src1, types().f32());
    } break;
    case eOpcode::V_MADAK_F32: {
      parse<op::FmaOp>(vdst, src0, src1, src2, types().f32());
    } break;
    case eOpcode::V_BCNT_U32_B32: {
      parse<op::BitCountOp>(vdst, src0, src1, types().i32());
    } break;
    // todo (threadid or normal handling)
    // case eOpcode::V_MBCNT_LO_U32_B32: {
    // } break;
    // case eOpcode::V_MBCNT_HI_U32_B32: {
    // } break;
    case eOpcode::V_ADD_I32: {
      parse<op::AddSIOp>(vdst, sdst, src0, src1, types().i32());
    } break;
    case eOpcode::V_SUB_I32: {
      parse<op::SubSIOp>(vdst, sdst, src0, src1, types().i32());
    } break;
    case eOpcode::V_SUBREV_I32: {
      parse<op::SubSIOp>(vdst, sdst, src1, src0, types().i32());
    } break;
    case eOpcode::V_ADDC_U32: {
      parse<op::AddUIOp>(vdst, sdst, src0, src1, src2, types().i32());
    } break;
    case eOpcode::V_SUBB_U32: {
      parse<op::SubUIOp>(vdst, sdst, src0, src1, src2, types().i32());
    } break;
    case eOpcode::V_SUBBREV_U32: {
      parse<op::SubUIOp>(vdst, sdst, src1, src0, src2, types().i32());
    } break;
    case eOpcode::V_LDEXP_F32: {
      parse<op::LDExpOp>(vdst, src1, src0, types().f32());
    } break;
    // case eOpcode::V_CVT_PKACCUM_U8_F32: {} break; // does not exist
    case eOpcode::V_CVT_PKNORM_I16_F32: {
      parse<op::ConvertPackSnormOp>(vdst, src1, src0);
    } break;
    case eOpcode::V_CVT_PKNORM_U16_F32: {
      parse<op::ConvertPackUnormOp>(vdst, src1, src0);
    } break;
    case eOpcode::V_CVT_PKRTZ_F16_F32: {
      parse<op::ConvertPackF32Op>(vdst, src1, src0);
    } break;
    case eOpcode::V_CVT_PK_U16_U32: {
      parse<op::ConvertPackUI32Op>(vdst, src1, src0);
    } break;
    case eOpcode::V_CVT_PK_I16_I32: {
      parse<op::ConvertPackSI32Op>(vdst, src1, src0);
    } break;
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }

  return size;
}

uint8_t Parser::handleVop3(CodeBlock& cb, pc_t pc, uint32_t const* pCode) {
  eOpcode op;

  auto isSDST = [op] { return op == eOpcode::V_MAD_U64_U32 || op == eOpcode::V_MAD_I64_I32; };

  auto inst  = VOP3(getU64(pCode));
  auto instS = VOP3_SDST(getU64(pCode));
  op         = (eOpcode)(OPcodeStart_VOP3 + inst.template get<VOP3::Field::OP>());

  auto const vdst_ = eOperandKind::VGPR(inst.template get<VOP3::Field::VDST>());
  auto const src0_ = eOperandKind((eOperandKind_t)inst.template get<VOP3::Field::SRC0>());
  auto const src1_ = eOperandKind((eOperandKind_t)inst.template get<VOP3::Field::SRC1>());
  auto const src2_ = eOperandKind((eOperandKind_t)inst.template get<VOP3::Field::SRC2>());

  auto const sdst_ = eOperandKind((eOperandKind_t)instS.template get<VOP3_SDST::Field::SDST>());

  auto const           omod   = inst.template get<VOP3::Field::OMOD>();
  std::bitset<3> const negate = inst.template get<VOP3::Field::NEG>();
  std::bitset<3> const abs    = inst.template get<VOP3::Field::ABS>();
  auto const           clamp  = inst.template get<VOP3::Field::CLAMP>();

  switch (op) {
    case eOpcode::V_MAD_LEGACY_F32: {
    } break;
    case eOpcode::V_MAD_F32: {
    } break;
    case eOpcode::V_MAD_I32_I24: {
    } break;
    case eOpcode::V_MAD_U32_U24: {
    } break;
    case eOpcode::V_CUBEID_F32: {
    } break;
    case eOpcode::V_CUBESC_F32: {
    } break;
    case eOpcode::V_CUBETC_F32: {
    } break;
    case eOpcode::V_CUBEMA_F32: {
    } break;
    case eOpcode::V_BFE_U32: {
    } break;
    case eOpcode::V_BFE_I32: {
    } break;
    case eOpcode::V_BFI_B32: {
    } break;
    case eOpcode::V_FMA_F32: {
    } break;
    case eOpcode::V_FMA_F64: {
    } break;
    case eOpcode::V_LERP_U8: {
    } break;
    case eOpcode::V_ALIGNBIT_B32: {
    } break;
    case eOpcode::V_ALIGNBYTE_B32: {
    } break;
    case eOpcode::V_MULLIT_F32: {
    } break;
    case eOpcode::V_MIN3_F32: {
    } break;
    case eOpcode::V_MIN3_I32: {
    } break;
    case eOpcode::V_MIN3_U32: {
    } break;
    case eOpcode::V_MAX3_F32: {
    } break;
    case eOpcode::V_MAX3_I32: {
    } break;
    case eOpcode::V_MAX3_U32: {
    } break;
    case eOpcode::V_MED3_F32: {
    } break;
    case eOpcode::V_MED3_I32: {
    } break;
    case eOpcode::V_MED3_U32: {
    } break;
    case eOpcode::V_SAD_U8: {
    } break;
    case eOpcode::V_SAD_HI_U8: {
    } break;
    case eOpcode::V_SAD_U16: {
    } break;
    case eOpcode::V_SAD_U32: {
    } break;
    case eOpcode::V_CVT_PK_U8_F32: {
    } break;
    case eOpcode::V_DIV_FIXUP_F32: {
    } break;
    case eOpcode::V_DIV_FIXUP_F64: {
    } break;
    case eOpcode::V_LSHL_B64: {
    } break;
    case eOpcode::V_LSHR_B64: {
    } break;
    case eOpcode::V_ASHR_I64: {
    } break;
    case eOpcode::V_ADD_F64: {
    } break;
    case eOpcode::V_MUL_F64: {
    } break;
    case eOpcode::V_MIN_F64: {
    } break;
    case eOpcode::V_MAX_F64: {
    } break;
    case eOpcode::V_LDEXP_F64: {
    } break;
    case eOpcode::V_MUL_LO_U32: {
    } break;
    case eOpcode::V_MUL_HI_U32: {
    } break;
    case eOpcode::V_MUL_LO_I32: {
    } break;
    case eOpcode::V_MUL_HI_I32: {
    } break;
    case eOpcode::V_DIV_SCALE_F32: {
    } break;
    case eOpcode::V_DIV_SCALE_F64: {
    } break;
    case eOpcode::V_DIV_FMAS_F32: {
    } break;
    case eOpcode::V_DIV_FMAS_F64: {
    } break;
    case eOpcode::V_MSAD_U8: {
    } break;
    case eOpcode::V_QSAD_U8: {
    } break;
    case eOpcode::V_MQSAD_U8: {
    } break;
    case eOpcode::V_TRIG_PREOP_F64: {
    } break;
    case eOpcode::V_MQSAD_U32_U8: {
    } break;
    case eOpcode::V_MAD_U64_U32: {
    } break;
    case eOpcode::V_MAD_I64_I32: {
    } break;
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }
  return sizeof(uint64_t);
}

uint8_t Parser::handleVopc(CodeBlock& cb, pc_t pc, uint32_t const* pCode, bool extended) {
  eOpcode op;
  OpDst   sdst;
  OpSrc   src0, src1;

  uint8_t size = sizeof(uint32_t);
  if (extended) {
    auto inst = VOP3_SDST(getU64(pCode));
    op        = (eOpcode)(OPcodeStart_VOPC + inst.template get<VOP3_SDST::Field::OP>() - OpcodeOffset_VOPC_VOP3);

    auto const           omod   = inst.template get<VOP3_SDST::Field::OMOD>();
    std::bitset<3> const negate = inst.template get<VOP3_SDST::Field::NEG>();

    sdst = OpDst(eOperandKind((eOperandKind_t)inst.template get<VOP3_SDST::Field::SDST>()), omod, false);
    src0 = OpSrc(eOperandKind((eOperandKind_t)inst.template get<VOP3_SDST::Field::SRC0>()), negate[0], false);
    src1 = OpSrc(eOperandKind((eOperandKind_t)inst.template get<VOP3_SDST::Field::SRC1>()), negate[1], false);

    size = sizeof(uint64_t);
  } else {
    auto inst = VOPC(*pCode);
    op        = (eOpcode)(OPcodeStart_VOPC + inst.template get<VOPC::Field::OP>());
    sdst      = OpDst(eOperandKind::VCC());
    src0      = OpSrc(eOperandKind((eOperandKind_t)inst.template get<VOPC::Field::SRC0>()));
    src1      = OpSrc(eOperandKind::VGPR(inst.template get<VOPC::Field::VSRC1>()));

    if (src0.kind.isLiteral() || src1.kind.isLiteral()) {
      size = sizeof(uint64_t);
    }
  }

  constexpr std::array cmpOpsF = {op::eCmpFPredicate::AlwaysFalse, op::eCmpFPredicate::OLT, op::eCmpFPredicate::OEQ, op::eCmpFPredicate::OLE,
                                  op::eCmpFPredicate::OGT,         op::eCmpFPredicate::ONE, op::eCmpFPredicate::OGE, op::eCmpFPredicate::ORD,
                                  op::eCmpFPredicate::UNO,         op::eCmpFPredicate::OLT, op::eCmpFPredicate::OEQ, op::eCmpFPredicate::OLE,
                                  op::eCmpFPredicate::OGT,         op::eCmpFPredicate::ONE, op::eCmpFPredicate::OGE, op::eCmpFPredicate::AlwaysTrue};

  constexpr std::array cmpOpsSI = {
      op::eCmpIPredicate::AlwaysFalse, op::eCmpIPredicate::slt, op::eCmpIPredicate::eq,  op::eCmpIPredicate::sle,
      op::eCmpIPredicate::sgt,         op::eCmpIPredicate::ne,  op::eCmpIPredicate::sge, op::eCmpIPredicate::AlwaysTrue,
  };

  constexpr std::array cmpOpsUI = {
      op::eCmpIPredicate::AlwaysFalse, op::eCmpIPredicate::ult, op::eCmpIPredicate::eq,  op::eCmpIPredicate::ule,
      op::eCmpIPredicate::ugt,         op::eCmpIPredicate::ne,  op::eCmpIPredicate::uge, op::eCmpIPredicate::AlwaysTrue,

  };

  // // compare float
  if (op >= eOpcode::V_CMP_F_F32 && op <= eOpcode::V_CMP_T_F32) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMP_F_F32;
    parse<op::CmpOp>(cmpOpsF[opIndex], sdst, src0, src1, types().f32());
  } else if (op >= eOpcode::V_CMPX_F_F32 && op <= eOpcode::V_CMPX_T_F32) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPX_F_F32;
    storeRegister(OpDst(eOperandKind::EXEC()), parse<op::CmpOp>(cmpOpsF[opIndex], sdst, src0, src1, types().f32()));
  } else if (op >= eOpcode::V_CMP_F_F64 && op <= eOpcode::V_CMP_T_F64) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMP_F_F64;
    parse<op::CmpOp>(cmpOpsF[opIndex], sdst, src0, src1, types().f64());
  } else if (op >= eOpcode::V_CMPX_F_F64 && op <= eOpcode::V_CMPX_T_F64) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPX_F_F64;
    storeRegister(OpDst(eOperandKind::EXEC()), parse<op::CmpOp>(cmpOpsF[opIndex], sdst, src0, src1, types().f64()));
  } else if (op >= eOpcode::V_CMPS_F_F32 && op <= eOpcode::V_CMPS_T_F32) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPS_F_F32;
    parse<op::CmpOp>(cmpOpsF[opIndex], sdst, src0, src1, types().f32());
  } else if (op >= eOpcode::V_CMPSX_F_F32 && op <= eOpcode::V_CMPSX_T_F32) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPSX_F_F32;
    storeRegister(OpDst(eOperandKind::EXEC()), parse<op::CmpOp>(cmpOpsF[opIndex], sdst, src0, src1, types().f32()));
  } else if (op >= eOpcode::V_CMPS_F_F64 && op <= eOpcode::V_CMPSX_T_F64) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPS_F_F64;
    parse<op::CmpOp>(cmpOpsF[opIndex], sdst, src0, src1, types().f64());
  } else if (op >= eOpcode::V_CMPSX_F_F64 && op <= eOpcode::V_CMPSX_T_F64) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPSX_F_F64;
    storeRegister(OpDst(eOperandKind::EXEC()), parse<op::CmpOp>(cmpOpsF[opIndex], sdst, src0, src1, types().f64()));
  }

  // // Compare integer
  else if (op >= eOpcode::V_CMP_F_I32 && op <= eOpcode::V_CMP_T_I32) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMP_F_I32;
    parse<op::CmpOp>(cmpOpsSI[opIndex], sdst, src0, src1, types().i32());
  } else if (op == eOpcode::V_CMP_CLASS_F32) { // todo
    throw std::runtime_error("class comp");
  } else if (op >= eOpcode::V_CMPX_F_I32 && op <= eOpcode::V_CMPX_T_I32) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPX_F_I32;
    storeRegister(OpDst(eOperandKind::EXEC()), parse<op::CmpOp>(cmpOpsSI[opIndex], sdst, src0, src1, types().i32()));
  } else if (op == eOpcode::V_CMPX_CLASS_F32) {
    throw std::runtime_error("class comp"); // todo
  } else if (op >= eOpcode::V_CMP_F_I64 && op <= eOpcode::V_CMP_T_I64) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMP_F_I64;
    parse<op::CmpOp>(cmpOpsSI[opIndex], sdst, src0, src1, types().i64());
  } else if (op == eOpcode::V_CMP_CLASS_F64) {
    throw std::runtime_error("class comp"); // todo
  } else if (op >= eOpcode::V_CMPX_F_I64 && op <= eOpcode::V_CMPX_T_I64) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPX_F_I64;
    storeRegister(OpDst(eOperandKind::EXEC()), parse<op::CmpOp>(cmpOpsSI[opIndex], sdst, src0, src1, types().i64()));
  } else if (op == eOpcode::V_CMPX_CLASS_F64) {
    throw std::runtime_error("class comp"); // todo
  }

  // // unsigned integers
  else if (op >= eOpcode::V_CMP_F_U32 && op <= eOpcode::V_CMP_T_U32) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMP_F_U32;
    parse<op::CmpOp>(cmpOpsUI[opIndex], sdst, src0, src1, types().i32());
  } else if (op >= eOpcode::V_CMPX_F_U32 && op <= eOpcode::V_CMPX_T_U32) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPX_F_U32;
    storeRegister(OpDst(eOperandKind::EXEC()), parse<op::CmpOp>(cmpOpsUI[opIndex], sdst, src0, src1, types().i32()));
  } else if (op >= eOpcode::V_CMP_F_U64 && op <= eOpcode::V_CMP_T_U64) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMP_F_U64;
    parse<op::CmpOp>(cmpOpsUI[opIndex], sdst, src0, src1, types().i64());
  } else if (op >= eOpcode::V_CMPX_F_U64 && op <= eOpcode::V_CMPX_T_U64) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPX_F_U64;
    storeRegister(OpDst(eOperandKind::EXEC()), parse<op::CmpOp>(cmpOpsUI[opIndex], sdst, src0, src1, types().i64()));
  } else {
    throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op)));
  }

  return size;
}

uint8_t Parser::handleVintrp(CodeBlock& cb, pc_t pc, uint32_t const* pCode) {
  uint8_t size = sizeof(uint32_t);

  auto       inst = VINTRP(*pCode);
  auto const op   = (eOpcode)(OPcodeStart_VINTRP + inst.template get<VINTRP::Field::OP>());

  auto const vdst    = eOperandKind((eOperandKind_t)inst.template get<VINTRP::Field::VDST>());
  auto       src0    = eOperandKind((eOperandKind_t)inst.template get<VINTRP::Field::VSRC>());
  auto const channel = (uint8_t)inst.template get<VINTRP::Field::ATTRCHAN>();
  auto const attr    = (uint8_t)inst.template get<VINTRP::Field::ATTR>();

  if (src0.isLiteral()) {
    size = sizeof(uint64_t);
  }

  return size;
}
} // namespace compiler::frontend