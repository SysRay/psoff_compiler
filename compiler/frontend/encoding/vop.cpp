#include "../debug_strings.h"
#include "../gfx/encoding_types.h"
#include "../parser.h"
#include "opcodes_table.h"

#include <bitset>
#include <format>
#include <stdexcept>

// mlir
#include <mlir/Dialect/Arith/IR/Arith.h>

namespace compiler::frontend {

uint8_t Parser::handleVop1(CodeBlock& cb, pc_t pc, uint32_t const* pCode, bool extended) {
  eOpcode op;

  uint8_t size = sizeof(uint32_t);
  if (extended) {
    auto inst = VOP3(getU64(pCode));
    op        = (eOpcode)(OPcodeStart_VOP1 + inst.template get<VOP3::Field::OP>() - OpcodeOffset_VOP1_VOP3);

    auto const vdst_ = eOperandKind::VGPR(inst.template get<VOP3::Field::VDST>());
    auto const src0_ = eOperandKind((eOperandKind_t)inst.template get<VOP3::Field::SRC0>());

    auto const           omod   = inst.template get<VOP3::Field::OMOD>();
    std::bitset<3> const negate = inst.template get<VOP3::Field::NEG>();
    std::bitset<3> const abs    = inst.template get<VOP3::Field::ABS>();
    auto const           clamp  = inst.template get<VOP3::Field::CLAMP>();

    size = sizeof(uint64_t);
  } else {
    auto inst = VOP1(*pCode);
    op        = (eOpcode)(OPcodeStart_VOP1 + inst.template get<VOP1::Field::OP>());

    auto vdst = eOperandKind::VGPR(inst.template get<VOP1::Field::VDST>());
    auto src0 = eOperandKind((eOperandKind_t)inst.template get<VOP1::Field::SRC0>());

    if (src0.isLiteral()) {
      size = sizeof(uint64_t);
    }
  }

  switch (op) {
    case eOpcode::V_NOP: {
    } break;
    case eOpcode::V_MOV_B32: {
    } break;
    case eOpcode::V_READFIRSTLANE_B32: {
    } break;
    case eOpcode::V_CVT_I32_F64: {
    } break;
    case eOpcode::V_CVT_F64_I32: {
    } break;
    case eOpcode::V_CVT_F32_I32: {
    } break;
    case eOpcode::V_CVT_F32_U32: {
    } break;
    case eOpcode::V_CVT_U32_F32: {
    } break;
    case eOpcode::V_CVT_I32_F32: {
    } break;
    case eOpcode::V_MOV_FED_B32: {
    } break;
    case eOpcode::V_CVT_F16_F32: {
    } break;
    case eOpcode::V_CVT_F32_F16: {
    } break;
    case eOpcode::V_CVT_RPI_I32_F32: {
    } break;
    case eOpcode::V_CVT_FLR_I32_F32: {
    } break;
    case eOpcode::V_CVT_OFF_F32_I4: {
    } break;
    case eOpcode::V_CVT_F32_F64: {
    } break;
    case eOpcode::V_CVT_F64_F32: {
    } break;
    case eOpcode::V_CVT_F32_UBYTE0: {
    } break;
    case eOpcode::V_CVT_F32_UBYTE1: {
    } break;
    case eOpcode::V_CVT_F32_UBYTE2: {
    } break;
    case eOpcode::V_CVT_F32_UBYTE3: {
    } break;
    case eOpcode::V_CVT_U32_F64: {
    } break;
    case eOpcode::V_CVT_F64_U32: {
    } break;
    case eOpcode::V_TRUNC_F64: {
    } break;
    case eOpcode::V_CEIL_F64: {
    } break;
    case eOpcode::V_RNDNE_F64: {
    } break;
    case eOpcode::V_FLOOR_F64: {
    } break;
    case eOpcode::V_FRACT_F32: {
    } break;
    case eOpcode::V_TRUNC_F32: {
    } break;
    case eOpcode::V_CEIL_F32: {
    } break;
    case eOpcode::V_RNDNE_F32: {
    } break;
    case eOpcode::V_FLOOR_F32: {
    } break;
    case eOpcode::V_EXP_F32: {
    } break;
    case eOpcode::V_LOG_CLAMP_F32: {
    } break;
    case eOpcode::V_LOG_F32: {
    } break;
    case eOpcode::V_RCP_CLAMP_F32: {
    } break;
    case eOpcode::V_RCP_LEGACY_F32: {
    } break;
    case eOpcode::V_RCP_F32: {
    } break;
    case eOpcode::V_RCP_IFLAG_F32: {
    } break;
    case eOpcode::V_RSQ_CLAMP_F32: {
    } break;
    case eOpcode::V_RSQ_LEGACY_F32: {
    } break;
    case eOpcode::V_RSQ_F32: {
    } break;
    case eOpcode::V_RCP_F64: {
    } break;
    case eOpcode::V_RCP_CLAMP_F64: {
    } break;
    case eOpcode::V_RSQ_F64: {
    } break;
    case eOpcode::V_RSQ_CLAMP_F64: {
    } break;
    case eOpcode::V_SQRT_F32: {
    } break;
    case eOpcode::V_SQRT_F64: {
    } break;
    case eOpcode::V_SIN_F32: {
    } break;
    case eOpcode::V_COS_F32: {
    } break;
    case eOpcode::V_NOT_B32: {
    } break;
    case eOpcode::V_BFREV_B32: {
    } break;
    case eOpcode::V_FFBH_U32: {
    } break;
    case eOpcode::V_FFBL_B32: {
    } break;
    case eOpcode::V_FFBH_I32: {
    } break;
    case eOpcode::V_FREXP_EXP_I32_F64: {
    } break;
    case eOpcode::V_FREXP_MANT_F64: {
    } break;
    case eOpcode::V_FRACT_F64: {
    } break;
    case eOpcode::V_FREXP_EXP_I32_F32: {
    } break;
    case eOpcode::V_FREXP_MANT_F32: {
    } break;
    case eOpcode::V_CLREXCP: {
    } break;
    case eOpcode::V_MOVRELD_B32: {
    } break;
    case eOpcode::V_MOVRELS_B32: {
    } break;
    case eOpcode::V_MOVRELSD_B32: {
    } break;
    case eOpcode::V_LOG_LEGACY_F32: {
    } break;
    case eOpcode::V_EXP_LEGACY_F32: {
    } break;
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }
  return size;
}

uint8_t Parser::handleVop2(CodeBlock& cb, pc_t pc, uint32_t const* pCode, bool extended) {
  eOpcode op;

  auto isSDST = [op] {
    return op == eOpcode::V_ADD_I32 || op == eOpcode::V_SUB_I32 || op == eOpcode::V_SUBREV_I32 || op == eOpcode::V_ADDC_U32 || op == eOpcode::V_SUBB_U32 ||
           op == eOpcode::V_SUBBREV_U32;
  };

  uint8_t size = sizeof(uint32_t);
  if (extended) {
    auto inst  = VOP3(getU64(pCode));
    auto instS = VOP3_SDST(getU64(pCode));
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

    size = sizeof(uint64_t);
  } else {
    auto inst = VOP2(*pCode);
    op        = (eOpcode)(OPcodeStart_VOP2 + inst.template get<VOP2::Field::OP>());

    auto vdst = eOperandKind::VGPR(inst.template get<VOP2::Field::VDST>());
    auto sdst = eOperandKind::VCC();
    auto src0 = eOperandKind((eOperandKind_t)inst.template get<VOP2::Field::SRC0>());
    auto src1 = eOperandKind(eOperandKind::VGPR(inst.template get<VOP2::Field::VSRC1>()));
    auto src2 = eOperandKind::VCC();

    if (src0.isLiteral() || src1.isLiteral()) {
      size = sizeof(uint64_t);
    }
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

  return sizeof(uint64_t);
}

uint8_t Parser::handleVopc(CodeBlock& cb, pc_t pc, uint32_t const* pCode, bool extended) {
  eOpcode      op;
  eOperandKind sdst {}, src0 {}, src1 {};

  uint8_t size = sizeof(uint32_t);
  if (extended) {
    auto inst = VOP3_SDST(getU64(pCode));
    op        = (eOpcode)(OPcodeStart_VOPC + inst.template get<VOP3_SDST::Field::OP>() - OpcodeOffset_VOPC_VOP3);

    sdst = eOperandKind((eOperandKind_t)inst.template get<VOP3_SDST::Field::SDST>());
    src0 = eOperandKind((eOperandKind_t)inst.template get<VOP3_SDST::Field::SRC0>());
    src1 = eOperandKind((eOperandKind_t)inst.template get<VOP3_SDST::Field::SRC1>());

    auto const           omod   = inst.template get<VOP3_SDST::Field::OMOD>();
    std::bitset<3> const negate = inst.template get<VOP3_SDST::Field::NEG>();

    size = sizeof(uint64_t);
  } else {
    auto inst = VOPC(*pCode);
    op        = (eOpcode)(OPcodeStart_VOPC + inst.template get<VOPC::Field::OP>());
    sdst      = eOperandKind::VCC();
    src0      = eOperandKind((eOperandKind_t)inst.template get<VOPC::Field::SRC0>());
    src1      = eOperandKind::VGPR(inst.template get<VOPC::Field::VSRC1>());

    if (src0.isLiteral() || src1.isLiteral()) {
      size = sizeof(uint64_t);
    }
  }

  using namespace mlir::arith;

  auto compf = [&](CmpFPredicate comp, mlir::Type type, bool exec, bool signaling) {
    // auto value0 = loadRegister(src0, type);
    // auto value1 = loadRegister(src1, type);

    // auto res = _mlirBuilder.create<mlir::arith::CmpFOp>(_defaultLocation, comp, value0, value1);

    // storeRegister(sdst, res);
    // if (exec) storeRegister(eOperandKind::EXEC(), res);
  };

  auto compI = [&](CmpIPredicate comp, mlir::Type type, bool exec) {
    // auto value0 = loadRegister(src0, type);
    // auto value1 = loadRegister(src1, type);

    // auto res = _mlirBuilder.create<mlir::arith::CmpIOp>(_defaultLocation, comp, value0, value1);

    // storeRegister(sdst, res);
    // if (exec) storeRegister(eOperandKind::EXEC(), res);
  };

  constexpr std::array cmpOpsF = {CmpFPredicate::AlwaysFalse, CmpFPredicate::OLT, CmpFPredicate::OEQ, CmpFPredicate::OLE,
                                  CmpFPredicate::OGT,         CmpFPredicate::ONE, CmpFPredicate::OGE, CmpFPredicate::ORD,
                                  CmpFPredicate::UNO,         CmpFPredicate::OLT, CmpFPredicate::OEQ, CmpFPredicate::OLE,
                                  CmpFPredicate::OGT,         CmpFPredicate::ONE, CmpFPredicate::OGE, CmpFPredicate::AlwaysTrue};

  constexpr std::array cmdOpsSI = {
      CmpIPredicate::slt, // CmpIPredicate::AlwaysFalse
      CmpIPredicate::slt, CmpIPredicate::eq, CmpIPredicate::sle, CmpIPredicate::sgt, CmpIPredicate::ne, CmpIPredicate::sge,
      CmpIPredicate::slt, // CmpIPredicate::AlwaysFalse
  };

  constexpr std::array cmdOpsUI = {
      CmpIPredicate::ult, // CmpIPredicate::AlwaysFalse
      CmpIPredicate::ult, CmpIPredicate::eq, CmpIPredicate::ule, CmpIPredicate::ugt, CmpIPredicate::ne, CmpIPredicate::uge,
      CmpIPredicate::ult, // CmpIPredicate::AlwaysTrue

  };

  // // compare float

  // if (op >= eOpcode::V_CMP_F_F32 && op <= eOpcode::V_CMP_T_F32) {
  //   auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMP_F_F32;
  //   compf(cmpOpsF[opIndex], types().f32(), false, false);
  // } else if (op >= eOpcode::V_CMPX_F_F32 && op <= eOpcode::V_CMPX_T_F32) {
  //   auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPX_F_F32;
  //   compf(cmpOpsF[opIndex], types().f32(), true, false);
  // } else if (op >= eOpcode::V_CMP_F_F64 && op <= eOpcode::V_CMP_T_F64) {
  //   auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMP_F_F64;
  //   compf(cmpOpsF[opIndex], types().f32(), false, false);
  // } else if (op >= eOpcode::V_CMPX_F_F64 && op <= eOpcode::V_CMPX_T_F64) {
  //   auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPX_F_F64;
  //   compf(cmpOpsF[opIndex], types().f64(), true, false);
  // } else if (op >= eOpcode::V_CMPS_F_F32 && op <= eOpcode::V_CMPS_T_F32) {
  //   auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPS_F_F32;
  //   compf(cmpOpsF[opIndex], types().f32(), false, true);
  // } else if (op >= eOpcode::V_CMPSX_F_F32 && op <= eOpcode::V_CMPSX_T_F32) {
  //   auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPSX_F_F32;
  //   compf(cmpOpsF[opIndex], types().f32(), true, true);
  // } else if (op >= eOpcode::V_CMPS_F_F64 && op <= eOpcode::V_CMPSX_T_F64) {
  //   auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPS_F_F64;
  //   compf(cmpOpsF[opIndex], types().f64(), false, true);
  // } else if (op >= eOpcode::V_CMPSX_F_F64 && op <= eOpcode::V_CMPSX_T_F64) {
  //   auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPSX_F_F64;
  //   compf(cmpOpsF[opIndex], types().f64(), true, true);
  // }

  // // Compare integer
  // else if (op == eOpcode::V_CMP_F_I32) {
  //   auto res = loadRegister(eOperandKind::createImm(0), types().i1());
  //   storeRegister(sdst, res);
  // } else if (op > eOpcode::V_CMP_F_I32 && op < eOpcode::V_CMP_T_I32) {
  //   auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMP_F_I32;
  //   compI(cmdOpsSI[opIndex], types().i32(), false);
  // } else if (op == eOpcode::V_CMP_T_I32) {
  //   auto res = loadRegister(eOperandKind::createImm(1), types().i1());
  //   storeRegister(sdst, res);
  // } else if (op == eOpcode::V_CMP_CLASS_F32) { // todo
  //   throw std::runtime_error("class comp");
  // } else if (op == eOpcode::V_CMPX_F_I32) {
  //   auto res = loadRegister(eOperandKind::createImm(0), types().i1());
  //   storeRegister(sdst, res);
  //   storeRegister(eOperandKind::EXEC(), res);
  // } else if (op > eOpcode::V_CMPX_F_I32 && op < eOpcode::V_CMPX_T_I32) {
  //   auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPX_F_I32;
  //   compI(cmdOpsSI[opIndex], types().i32(), true);
  // } else if (op == eOpcode::V_CMPX_T_I32) {
  //   auto res = loadRegister(eOperandKind::createImm(1), types().i1());
  //   storeRegister(sdst, res);
  //   storeRegister(eOperandKind::EXEC(), res);
  // } else if (op == eOpcode::V_CMPX_CLASS_F32) {
  //   throw std::runtime_error("class comp"); // todo
  // } else if (op == eOpcode::V_CMP_F_I64) {
  //   auto res = loadRegister(eOperandKind::createImm(0), types().i1());
  //   storeRegister(sdst, res);
  // } else if (op > eOpcode::V_CMP_F_I64 && op < eOpcode::V_CMP_T_I64) {
  //   auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMP_F_I64;
  //   compI(cmdOpsSI[opIndex], types().i64(), false);
  // } else if (op == eOpcode::V_CMP_T_I64) {
  //   auto res = loadRegister(eOperandKind::createImm(1), types().i1());
  //   storeRegister(sdst, res);
  // } else if (op == eOpcode::V_CMP_CLASS_F64) {
  //   throw std::runtime_error("class comp"); // todo
  // } else if (op == eOpcode::V_CMPX_F_I64) {
  //   auto res = loadRegister(eOperandKind::createImm(0), types().i1());
  //   storeRegister(sdst, res);
  //   storeRegister(eOperandKind::EXEC(), res);
  // } else if (op > eOpcode::V_CMPX_F_I64 && op < eOpcode::V_CMPX_T_I64) {
  //   auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPX_F_I64;
  //   compI(cmdOpsSI[opIndex], types().i64(), true);
  // } else if (op == eOpcode::V_CMPX_T_I64) {
  //   auto res = loadRegister(eOperandKind::createImm(1), types().i1());
  //   storeRegister(sdst, res);
  //   storeRegister(eOperandKind::EXEC(), res);
  // } else if (op == eOpcode::V_CMPX_CLASS_F64) {
  //   throw std::runtime_error("class comp"); // todo
  // }

  // // // unsigned integers
  // else if (op == eOpcode::V_CMP_F_U32) {
  //   auto res = loadRegister(eOperandKind::createImm(0), types().i1());
  //   storeRegister(sdst, res);
  // } else if (op > eOpcode::V_CMP_F_U32 && op < eOpcode::V_CMP_T_U32) {
  //   auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMP_F_U32;
  //   compI(cmdOpsUI[opIndex], types().i32(), false);
  // } else if (op == eOpcode::V_CMP_T_U32) {
  //   auto res = loadRegister(eOperandKind::createImm(1), types().i1());
  //   storeRegister(sdst, res);
  // } else if (op == eOpcode::V_CMPX_F_U32) {
  //   auto res = loadRegister(eOperandKind::createImm(0), types().i1());
  //   storeRegister(sdst, res);
  //   storeRegister(eOperandKind::EXEC(), res);
  // } else if (op > eOpcode::V_CMPX_F_U32 && op < eOpcode::V_CMPX_T_U32) {
  //   auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPX_F_U32;
  //   compI(cmdOpsUI[opIndex], types().i32(), true);
  // } else if (op == eOpcode::V_CMPX_T_U32) {
  //   auto res = loadRegister(eOperandKind::createImm(1), types().i1());
  //   storeRegister(sdst, res);
  //   storeRegister(eOperandKind::EXEC(), res);
  // } else if (op == eOpcode::V_CMP_F_U64) {
  //   auto res = loadRegister(eOperandKind::createImm(0), types().i1());
  //   storeRegister(sdst, res);
  // } else if (op > eOpcode::V_CMP_F_U64 && op < eOpcode::V_CMP_T_U64) {
  //   auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMP_F_U64;
  //   compI(cmdOpsUI[opIndex], types().i64(), false);
  // } else if (op == eOpcode::V_CMP_F_U64) {
  //   auto res = loadRegister(eOperandKind::createImm(1), types().i1());
  //   storeRegister(sdst, res);
  // } else if (op == eOpcode::V_CMPX_F_U64) {
  //   auto res = loadRegister(eOperandKind::createImm(0), types().i1());
  //   storeRegister(sdst, res);
  //   storeRegister(eOperandKind::EXEC(), res);
  // } else if (op > eOpcode::V_CMPX_F_U64 && op < eOpcode::V_CMPX_T_U64) {
  //   auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPX_F_U64;
  //   compI(cmdOpsUI[opIndex], types().i64(), true);
  // } else if (op == eOpcode::V_CMPX_T_U64) {
  //   auto res = loadRegister(eOperandKind::createImm(1), types().i1());
  //   storeRegister(sdst, res);
  //   storeRegister(eOperandKind::EXEC(), res);
  // } else {
  //   throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op)));
  // }

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