#include "../debug_strings.h"
#include "../gfx/encoding_types.h"
#include "../gfx/operand_types.h"
#include "../parser.h"
#include "opcodes_table.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend {

uint8_t Parser::handleSop1(CodeBlock& cb, pc_t pc, uint32_t const* pCode) {
  auto       inst = SOP1(*pCode);
  auto const op   = (eOpcode)(OPcodeStart_SOP1 + inst.template get<SOP1::Field::OP>());

  auto const sdst = eOperandKind((eOperandKind_t)inst.template get<SOP1::Field::SDST>());
  auto       src0 = eOperandKind((eOperandKind_t)inst.template get<SOP1::Field::SSRC0>());

  uint8_t size = sizeof(uint32_t);
  if (src0.isLiteral()) {
    size = sizeof(uint64_t);
    // src0 = createSrc(ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_u64 = **pCode}, ir::OperandType::i32()));
  }

  // *pCode += 1;

  switch (op) {
    case eOpcode::S_MOV_B32: {
    } break;
    case eOpcode::S_MOV_B64: {
    } break;
    case eOpcode::S_CMOV_B32: {
    } break;
    case eOpcode::S_CMOV_B64: {
    } break;
    case eOpcode::S_NOT_B32: {
    } break;
    case eOpcode::S_NOT_B64: {
    } break;
    case eOpcode::S_WQM_B32:
    case eOpcode::S_WQM_B64: {
    } break;
    case eOpcode::S_BREV_B32: {
    } break;
    case eOpcode::S_BREV_B64: {
    } break;
    case eOpcode::S_BCNT0_I32_B32: {
    } break;
    case eOpcode::S_BCNT0_I32_B64: {
    } break;
    case eOpcode::S_BCNT1_I32_B32: {
    } break;
    case eOpcode::S_BCNT1_I32_B64: {
    } break;
    case eOpcode::S_FF0_I32_B32: {
    } break;
    case eOpcode::S_FF0_I32_B64: {
    } break;
    case eOpcode::S_FF1_I32_B32: {
    } break;
    case eOpcode::S_FF1_I32_B64: {
    } break;
    case eOpcode::S_FLBIT_I32_B32: {
    } break;
    case eOpcode::S_FLBIT_I32_B64: {
    } break;
    case eOpcode::S_FLBIT_I32: {
    } break;
    case eOpcode::S_FLBIT_I32_I64: {
    } break;
    case eOpcode::S_SEXT_I32_I8: {
    } break;
    case eOpcode::S_SEXT_I32_I16: {
    } break;
    case eOpcode::S_BITSET0_B32: {
    } break;
    case eOpcode::S_BITSET0_B64: {
    } break;
    case eOpcode::S_BITSET1_B32: {
    } break;
    case eOpcode::S_BITSET1_B64: {
    } break;
    case eOpcode::S_GETPC_B64: {
    } break;
    case eOpcode::S_SETPC_B64: {

      cb.pc_end = pc;
    } break;
    case eOpcode::S_SWAPPC_B64: {

      cb.pc_end = pc;
    } break;
    // case eOpcode::S_RFE_B64: break; // Does not exist
    case eOpcode::S_AND_SAVEEXEC_B64: {
    } break;
    case eOpcode::S_OR_SAVEEXEC_B64: {
    } break;
    case eOpcode::S_XOR_SAVEEXEC_B64: {
    } break;
    case eOpcode::S_ANDN2_SAVEEXEC_B64: {
    } break;
    case eOpcode::S_ORN2_SAVEEXEC_B64: {
    } break;
    case eOpcode::S_NAND_SAVEEXEC_B64: {
    } break;
    case eOpcode::S_NOR_SAVEEXEC_B64: {
    } break;
    case eOpcode::S_XNOR_SAVEEXEC_B64: {
    } break;
      // case eOpcode::S_QUADMASK_B
      // 32: break; // todo,  might be same as wqm
      // case eOpcode::S_QUADMASK_B64: break; // todo, might be same as wqm
    // case eOpcode::S_MOVRELS_B32: {} break; // todo
    // case eOpcode::S_MOVRELS_B64: {} break; // todo
    // case eOpcode::S_MOVRELD_B32: {} break; // todo
    // case eOpcode::S_MOVRELD_B64: {} break; // todo
    // case eOpcode::S_CBRANCH_JOIN: {} break; // todo, make a block falltrough or handle data?
    //  case eOpcode::S_MOV_REGRD_B32: break; // Does not exist
    case eOpcode::S_ABS_I32: {
    } break;
    // case eOpcode::S_MOV_FED_B32: break; // Does not exist
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }

  return size;
}

uint8_t Parser::handleSop2(CodeBlock& cb, pc_t pc, uint32_t const* pCode) {
  auto       inst = SOP2(*pCode);
  auto const op   = (eOpcode)(OPcodeStart_SOP2 + inst.template get<SOP2::Field::OP>());

  auto const sdst = eOperandKind((eOperandKind_t)inst.template get<SOP2::Field::SDST>());
  auto       src0 = eOperandKind((eOperandKind_t)inst.template get<SOP2::Field::SSRC0>());
  auto       src1 = eOperandKind((eOperandKind_t)inst.template get<SOP2::Field::SSRC1>());

  uint8_t size = sizeof(uint32_t);
  if (src0.isLiteral() || src1.isLiteral()) {
    size = sizeof(uint64_t);
    // src0 = createSrc(ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_u64 = **pCode}, ir::OperandType::i32()));
  }

  // switch (op) {
  //   case eOpcode::S_ADD_U32: {
  //     auto res = ctx.create<AddIOp>(sdst, src0, src1, ir::OperandType::i32());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, src0, ir::OperandType::i32(), CmpIPredicate::ult);
  //   } break;
  //   case eOpcode::S_SUB_U32: {
  //     ctx.create<SubIOp>(sdst, src0, src1, ir::OperandType::i32());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src1, src0, ir::OperandType::i32(), CmpIPredicate::ugt);
  //   } break;
  //   case eOpcode::S_ADD_I32: {
  //     auto res = ctx.create<AddIOp>(sdst, src0, src1, ir::OperandType::i32());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, src0, ir::OperandType::i32(), CmpIPredicate::slt);
  //   } break;
  //   case eOpcode::S_SUB_I32: {
  //     ctx.create<SubIOp>(sdst, src0, src1, ir::OperandType::i32());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src1, src0, ir::OperandType::i32(), CmpIPredicate::sgt);
  //   } break;
  //   case eOpcode::S_ADDC_U32: {
  //     ctx.create<AddCarryIOp>(sdst, createDst(eOperandKind::SCC()), src0, src1, createSrc(eOperandKind::SCC()), ir::OperandType::i32());
  //   } break;
  //   case eOpcode::S_SUBB_U32: {
  //     ctx.create<SubBurrowIOp>(sdst, createDst(eOperandKind::SCC()), src0, src1, createSrc(eOperandKind::SCC()), ir::OperandType::i32());
  //   } break;
  //   case eOpcode::S_MIN_I32: {
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::slt);
  //     ctx.create<ir::dialect::core::SelectOp>(sdst, createSrc(eOperandKind::SCC()), src0, src1, ir::OperandType::i32());
  //   } break;
  //   case eOpcode::S_MIN_U32: {
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::ult);
  //     ctx.create<ir::dialect::core::SelectOp>(sdst, createSrc(eOperandKind::SCC()), src0, src1, ir::OperandType::i32());
  //   } break;
  //   case eOpcode::S_MAX_I32: {
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::sgt);
  //     ctx.create<ir::dialect::core::SelectOp>(sdst, createSrc(eOperandKind::SCC()), src0, src1, ir::OperandType::i32());
  //   } break;
  //   case eOpcode::S_MAX_U32: {
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::ugt);
  //     ctx.create<ir::dialect::core::SelectOp>(sdst, createSrc(eOperandKind::SCC()), src0, src1, ir::OperandType::i32());
  //   } break;
  //   case eOpcode::S_CSELECT_B32: {
  //     ctx.create<ir::dialect::core::SelectOp>(sdst, createSrc(eOperandKind::SCC()), src0, src1, ir::OperandType::i32());
  //   } break;
  //   case eOpcode::S_CSELECT_B64: {
  //     ctx.create<ir::dialect::core::SelectOp>(sdst, createSrc(eOperandKind::SCC()), src0, src1, ir::OperandType::i64());
  //   } break;
  //   case eOpcode::S_AND_B32: {
  //     auto res = ctx.create<BitAndOp>(sdst, src0, src1, ir::OperandType::i32());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_AND_B64: {
  //     auto res = ctx.create<BitAndOp>(sdst, src0, src1, ir::OperandType::i64());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_OR_B32: {
  //     auto res = ctx.create<BitOrOp>(sdst, src0, src1, ir::OperandType::i32());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_OR_B64: {
  //     auto res = ctx.create<BitOrOp>(sdst, src0, src1, ir::OperandType::i64());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_XOR_B32: {
  //     auto res = ctx.create<BitXorOp>(sdst, src0, src1, ir::OperandType::i32());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_XOR_B64: {
  //     auto res = ctx.create<BitXorOp>(sdst, src0, src1, ir::OperandType::i64());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_ANDN2_B32: {
  //     auto in1 = ctx.create<NotOp>(createDst(), src1, ir::OperandType::i32());
  //     auto res = ctx.create<BitAndOp>(sdst, src0, in1, ir::OperandType::i32());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_ANDN2_B64: {
  //     auto in1 = ctx.create<NotOp>(createDst(), src1, ir::OperandType::i64());
  //     auto res = ctx.create<BitAndOp>(sdst, src0, in1, ir::OperandType::i64());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_ORN2_B32: {
  //     auto in1 = ctx.create<NotOp>(createDst(), src1, ir::OperandType::i32());
  //     auto res = ctx.create<BitOrOp>(sdst, src0, in1, ir::OperandType::i32());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_ORN2_B64: {
  //     auto in1 = ctx.create<NotOp>(createDst(), src1, ir::OperandType::i64());
  //     auto res = ctx.create<BitOrOp>(sdst, src0, in1, ir::OperandType::i64());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_NAND_B32: {
  //     auto res_ = ctx.create<BitAndOp>(createDst(), src0, src1, ir::OperandType::i32());
  //     auto res  = ctx.create<NotOp>(sdst, res_, ir::OperandType::i32());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_NAND_B64: {
  //     auto res_ = ctx.create<BitAndOp>(createDst(), src0, src1, ir::OperandType::i64());
  //     auto res  = ctx.create<NotOp>(sdst, res_, ir::OperandType::i64());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_NOR_B32: {
  //     auto res_ = ctx.create<BitOrOp>(createDst(), src0, src1, ir::OperandType::i32());
  //     auto res  = ctx.create<NotOp>(sdst, res_, ir::OperandType::i32());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_NOR_B64: {
  //     auto res_ = ctx.create<BitOrOp>(createDst(), src0, src1, ir::OperandType::i64());
  //     auto res  = ctx.create<NotOp>(sdst, res_, ir::OperandType::i64());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_XNOR_B32: {
  //     auto res_ = ctx.create<BitXorOp>(createDst(), src0, src1, ir::OperandType::i32());
  //     auto res  = ctx.create<NotOp>(sdst, res_, ir::OperandType::i32());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_XNOR_B64: {
  //     auto res_ = ctx.create<BitXorOp>(createDst(), src0, src1, ir::OperandType::i64());
  //     auto res  = ctx.create<NotOp>(sdst, res_, ir::OperandType::i64());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_LSHL_B32: {
  //     auto res = ctx.create<ShiftLUIOp>(sdst, src0, src1, ir::OperandType::i32());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_LSHL_B64: {
  //     auto res = ctx.create<ShiftLUIOp>(sdst, src0, src1, ir::OperandType::i64());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_LSHR_B32: {
  //     auto res = ctx.create<ShiftRUIOp>(sdst, src0, src1, ir::OperandType::i32());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_LSHR_B64: {
  //     auto res = ctx.create<ShiftRUIOp>(sdst, src0, src1, ir::OperandType::i64());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_ASHR_I32: {
  //     auto res = ctx.create<ShiftRSIOp>(sdst, src0, src1, ir::OperandType::i32());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_ASHR_I64: {
  //     auto res = ctx.create<ShiftRSIOp>(sdst, src0, src1, ir::OperandType::i64());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_BFM_B32: {
  //     auto res = ctx.create<BitFieldMaskOp>(sdst, src0, src1, ir::OperandType::i32());
  //   } break;
  //   case eOpcode::S_BFM_B64: {
  //     auto res = ctx.create<BitFieldMaskOp>(sdst, src0, src1, ir::OperandType::i64());
  //   } break;
  //   case eOpcode::S_MUL_I32: {
  //     auto res = ctx.create<MulIOp>(sdst, src0, src1, ir::OperandType::i32());
  //   } break;
  //   case eOpcode::S_BFE_U32: {
  //     auto res = ctx.create<BitUIExtractOp>(sdst, src0, src1, ir::OperandType::i32());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_BFE_I32: {
  //     auto res = ctx.create<BitSIExtractOp>(sdst, src0, src1, ir::OperandType::i32());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_BFE_U64: {
  //     auto res = ctx.create<BitSIExtractOp>(sdst, src0, src1, ir::OperandType::i64());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
  //   } break;
  //   case eOpcode::S_BFE_I64: {
  //     auto res = ctx.create<BitSIExtractOp>(sdst, src0, src1, ir::OperandType::i64());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
  //   } break;
  //   // case eOpcode::S_CBRANCH_G_FORK: { } break; // todo, make a block falltrough or handle data?
  //   case eOpcode::S_ABSDIFF_I32: {
  //     auto res_ = ctx.create<SubIOp>(sdst, src0, src1, ir::OperandType::i32());
  //     auto res  = ctx.create<AbsoluteOp>(sdst, res_, ir::OperandType::i32());
  //     ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
  //   } break;
  //   default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  // }
  return size;
}

uint8_t Parser::handleSopc(CodeBlock& cb, pc_t pc, uint32_t const* pCode) {
  auto       inst = SOPC(*pCode);
  auto const op   = (eOpcode)(OPcodeStart_SOPC + inst.template get<SOPC::Field::OP>());

  auto const sdst = eOperandKind::SCC();
  auto       src0 = eOperandKind((eOperandKind_t)inst.template get<SOPC::Field::SSRC0>());
  auto       src1 = eOperandKind((eOperandKind_t)inst.template get<SOPC::Field::SSRC1>());

  uint8_t size = sizeof(uint32_t);
  if (src0.isLiteral() || src1.isLiteral()) {
    size = sizeof(uint64_t);
    // src0 = createSrc(ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_u64 = **pCode}, ir::OperandType::i32()));
  }

  // switch (op) {
  //     case eOpcode::S_CMP_EQ_I32: {
  //       ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::eq);
  //     } break;
  //     case eOpcode::S_CMP_LG_I32: {
  //       ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::ne);
  //     } break;
  //     case eOpcode::S_CMP_GT_I32: {
  //       ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::sgt);
  //     } break;
  //     case eOpcode::S_CMP_GE_I32: {
  //       ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::sge);
  //     } break;
  //     case eOpcode::S_CMP_LT_I32: {
  //       ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::slt);
  //     } break;
  //     case eOpcode::S_CMP_LE_I32: {
  //       ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::sle);
  //     } break;
  //     case eOpcode::S_CMP_EQ_U32: {
  //       ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::eq);
  //     } break;
  //     case eOpcode::S_CMP_LG_U32: {
  //       ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::ne);
  //     } break;
  //     case eOpcode::S_CMP_GT_U32: {
  //       ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::ugt);
  //     } break;
  //     case eOpcode::S_CMP_GE_U32: {
  //       ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::uge);
  //     } break;
  //     case eOpcode::S_CMP_LT_U32: {
  //       ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::ult);
  //     } break;
  //     case eOpcode::S_CMP_LE_U32: {
  //       ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::ule);
  //     } break;
  //     case eOpcode::S_BITCMP0_B32: {
  //       auto in0 = ctx.create<ir::dialect::arith::NotOp>(createDst(), src0, ir::OperandType::i32());
  //       ctx.create<BitCmpOp>(createDst(eOperandKind::SCC()), in0, src1, ir::OperandType::i32());
  //     } break;
  //     case eOpcode::S_BITCMP1_B32: {
  //       auto in0 = ctx.create<ir::dialect::arith::NotOp>(createDst(), src0, ir::OperandType::i32());
  //       ctx.create<BitCmpOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32());
  //     } break;
  //     case eOpcode::S_BITCMP0_B64: {
  //       ctx.create<BitCmpOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i64());
  //     } break;
  //     case eOpcode::S_BITCMP1_B64: {
  //       ctx.create<BitCmpOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i64());
  //     } break;
  //     case eOpcode::S_SETVSKIP: {
  //       ctx.create<BitCmpOp>(createDst(eOperandKind::VSKIP()), src0, src1, ir::OperandType::i32());
  //     } break;
  //     default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  //   }
  return size;
}

uint8_t Parser::handleSopk(CodeBlock& cb, pc_t pc, uint32_t const* pCode) {
  auto       inst = SOPK(*pCode);
  auto const op   = (eOpcode)(OPcodeStart_SOPK + inst.template get<SOPK::Field::OP>());

  auto const sdst  = eOperandKind((eOperandKind_t)inst.template get<SOPK::Field::SDST>());
  auto const imm16 = (int16_t)inst.template get<SOPK::Field::SIMM16>();

  return sizeof(uint32_t);
}

uint8_t Parser::handleSopp(CodeBlock& cb, pc_t pc, uint32_t const* pCode) {
  auto       inst = SOPP(*pCode);
  auto const op   = (eOpcode)(OPcodeStart_SOPP + inst.template get<SOPP::Field::OP>());

  auto const offset = (int16_t)inst.template get<SOPP::Field::SIMM16>();

  switch (op) {
    case eOpcode::S_NOP: break; // ignore
    case eOpcode::S_ENDPGM: {

      cb.pc_end = pc;
    } break;
    case eOpcode::S_BRANCH: {

      cb.pc_end = pc;

      auto targetPc = (int64_t)(sizeof(uint32_t) + pc) + sizeof(uint32_t) * (int64_t)offset;
      getOrCreateBlock(targetPc);
    } break;
    case eOpcode::S_CBRANCH_SCC0: {

      cb.pc_end = pc;

      auto targetIfPc   = sizeof(uint32_t) + pc;
      auto targetElsePc = (int64_t)(sizeof(uint32_t) + pc) + sizeof(uint32_t) * (int64_t)offset;
      getOrCreateBlock(targetElsePc);
      getOrCreateBlock(targetIfPc);
    } break;
    case eOpcode::S_CBRANCH_SCC1: {

      cb.pc_end = pc;

      auto targetIfPc   = sizeof(uint32_t) + pc;
      auto targetElsePc = (int64_t)(sizeof(uint32_t) + pc) + sizeof(uint32_t) * (int64_t)offset;
      getOrCreateBlock(targetElsePc);
      getOrCreateBlock(targetIfPc);
    } break;
    case eOpcode::S_CBRANCH_VCCZ: {

      cb.pc_end = pc;

      auto targetIfPc   = sizeof(uint32_t) + pc;
      auto targetElsePc = (int64_t)(sizeof(uint32_t) + pc) + sizeof(uint32_t) * (int64_t)offset;
      getOrCreateBlock(targetElsePc);
      getOrCreateBlock(targetIfPc);
    } break;
    case eOpcode::S_CBRANCH_VCCNZ: {

      cb.pc_end = pc;

      auto targetIfPc   = sizeof(uint32_t) + pc;
      auto targetElsePc = (int64_t)(sizeof(uint32_t) + pc) + sizeof(uint32_t) * (int64_t)offset;
      getOrCreateBlock(targetElsePc);
      getOrCreateBlock(targetIfPc);
    } break;
    case eOpcode::S_CBRANCH_EXECZ: {

      cb.pc_end = pc;

      auto targetIfPc   = sizeof(uint32_t) + pc;
      auto targetElsePc = (int64_t)(sizeof(uint32_t) + pc) + sizeof(uint32_t) * (int64_t)offset;
      getOrCreateBlock(targetElsePc);
      getOrCreateBlock(targetIfPc);
    } break;
    case eOpcode::S_CBRANCH_EXECNZ: {

      cb.pc_end = pc;

      auto targetIfPc   = sizeof(uint32_t) + pc;
      auto targetElsePc = (int64_t)(sizeof(uint32_t) + pc) + sizeof(uint32_t) * (int64_t)offset;
      getOrCreateBlock(targetElsePc);
      getOrCreateBlock(targetIfPc);
    } break;
    case eOpcode::S_BARRIER: {
    } break;
    // case eOpcode::S_SETKILL: {} break; // Does not exist
    case eOpcode::S_WAITCNT:
      break; // ignore
    // case eOpcode::S_SETHALT: {} break; // Does not exist
    case eOpcode::S_SLEEP: break; // ignore
    case eOpcode::S_SETPRIO: {
    } break;
    case eOpcode::S_SENDMSG: {
    } break;
    // case eOpcode::S_SENDMSGHALT: {} break; // Does not exist
    // case eOpcode::S_TRAP: {} break; // Does not exist
    case eOpcode::S_ICACHE_INV: {
    } break;
    case eOpcode::S_INCPERFLEVEL: {
    } break;
    case eOpcode::S_DECPERFLEVEL: {
    } break;
    case eOpcode::S_TTRACEDATA: {
    } break;
    // case eOpcode::S_CBRANCH_CDBGSYS: {} break; // Does not exist
    // case eOpcode::S_CBRANCH_CDBGUSER: {} break; // Does not exist
    // case eOpcode::S_CBRANCH_CDBGSYS_OR_USER: {} break; // Does not exist
    // case eOpcode::S_CBRANCH_CDBGSYS_AND_USER: {} break;// Does not exist
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }

  return sizeof(uint32_t);
}
} // namespace compiler::frontend