#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "frontend/ir_types.h"
#include "ir/dialects/arith/builder.h"
#include "ir/dialects/core/builder.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
InstructionKind_t handleSop2(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = SOP2(**pCode);
  auto const op   = (parser::eOpcode)(OPcodeStart_SOP2 + inst.template get<SOP2::Field::OP>());

  auto const sdst = createDst(eOperandKind((eOperandKind_t)inst.template get<SOP2::Field::SDST>()));
  auto       src0 = createSrc(eOperandKind((eOperandKind_t)inst.template get<SOP2::Field::SSRC0>()));
  auto       src1 = createSrc(eOperandKind((eOperandKind_t)inst.template get<SOP2::Field::SSRC1>()));

  if (eOperandKind(src0.kind).isLiteral()) {
    *pCode += 1;
    src0 = createSrc(ctx.create<ir::dialect::core::ConstantOp>(createDst(), ir::ConstantValue {.value_u64 = **pCode}, ir::OperandType::i32()));
  } else if (eOperandKind(src1.kind).isLiteral()) {
    *pCode += 1;
    src1 = createSrc(ctx.create<ir::dialect::core::ConstantOp>(createDst(), ir::ConstantValue {.value_u64 = **pCode}, ir::OperandType::i32()));
  }

  *pCode += 1;

  using namespace ir::dialect::arith;
  switch (op) {
    case eOpcode::S_ADD_U32: {
      auto res = ctx.create<AddIOp>(sdst, src0, src1, ir::OperandType::i32());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, src0, ir::OperandType::i32(), CmpIPredicate::ult);
    } break;
    case eOpcode::S_SUB_U32: {
      ctx.create<SubIOp>(sdst, src0, src1, ir::OperandType::i32());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src1, src0, ir::OperandType::i32(), CmpIPredicate::ugt);
    } break;
    case eOpcode::S_ADD_I32: {
      auto res = ctx.create<AddIOp>(sdst, src0, src1, ir::OperandType::i32());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, src0, ir::OperandType::i32(), CmpIPredicate::slt);
    } break;
    case eOpcode::S_SUB_I32: {
      ctx.create<SubIOp>(sdst, src0, src1, ir::OperandType::i32());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src1, src0, ir::OperandType::i32(), CmpIPredicate::sgt);
    } break;
    case eOpcode::S_ADDC_U32: {
      ctx.create<AddCarryIOp>(sdst, createDst(eOperandKind::SCC()), src0, src1, createSrc(eOperandKind::SCC()), ir::OperandType::i32());
    } break;
    case eOpcode::S_SUBB_U32: {
      ctx.create<SubBurrowIOp>(sdst, createDst(eOperandKind::SCC()), src0, src1, createSrc(eOperandKind::SCC()), ir::OperandType::i32());
    } break;
    case eOpcode::S_MIN_I32: {
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::slt);
      ctx.create<ir::dialect::core::SelectOp>(sdst, createSrc(eOperandKind::SCC()), src0, src1, ir::OperandType::i32());
    } break;
    case eOpcode::S_MIN_U32: {
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::ult);
      ctx.create<ir::dialect::core::SelectOp>(sdst, createSrc(eOperandKind::SCC()), src0, src1, ir::OperandType::i32());
    } break;
    case eOpcode::S_MAX_I32: {
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::sgt);
      ctx.create<ir::dialect::core::SelectOp>(sdst, createSrc(eOperandKind::SCC()), src0, src1, ir::OperandType::i32());
    } break;
    case eOpcode::S_MAX_U32: {
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), src0, src1, ir::OperandType::i32(), CmpIPredicate::ugt);
      ctx.create<ir::dialect::core::SelectOp>(sdst, createSrc(eOperandKind::SCC()), src0, src1, ir::OperandType::i32());
    } break;
    case eOpcode::S_CSELECT_B32: {
      ctx.create<ir::dialect::core::SelectOp>(sdst, createSrc(eOperandKind::SCC()), src0, src1, ir::OperandType::i32());
    } break;
    case eOpcode::S_CSELECT_B64: {
      ctx.create<ir::dialect::core::SelectOp>(sdst, createSrc(eOperandKind::SCC()), src0, src1, ir::OperandType::i64());
    } break;
    case eOpcode::S_AND_B32: {
      auto res = ctx.create<BitAndOp>(sdst, src0, src1, ir::OperandType::i32());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_AND_B64: {
      auto res = ctx.create<BitAndOp>(sdst, src0, src1, ir::OperandType::i64());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_OR_B32: {
      auto res = ctx.create<BitOrOp>(sdst, src0, src1, ir::OperandType::i32());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_OR_B64: {
      auto res = ctx.create<BitOrOp>(sdst, src0, src1, ir::OperandType::i64());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_XOR_B32: {
      auto res = ctx.create<BitXorOp>(sdst, src0, src1, ir::OperandType::i32());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_XOR_B64: {
      auto res = ctx.create<BitXorOp>(sdst, src0, src1, ir::OperandType::i64());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_ANDN2_B32: {
      auto in1 = ctx.create<NotOp>(createDst(), src1, ir::OperandType::i32());
      auto res = ctx.create<BitAndOp>(sdst, src0, in1, ir::OperandType::i32());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_ANDN2_B64: {
      auto in1 = ctx.create<NotOp>(createDst(), src1, ir::OperandType::i64());
      auto res = ctx.create<BitAndOp>(sdst, src0, in1, ir::OperandType::i64());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_ORN2_B32: {
      auto in1 = ctx.create<NotOp>(createDst(), src1, ir::OperandType::i32());
      auto res = ctx.create<BitOrOp>(sdst, src0, in1, ir::OperandType::i32());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_ORN2_B64: {
      auto in1 = ctx.create<NotOp>(createDst(), src1, ir::OperandType::i64());
      auto res = ctx.create<BitOrOp>(sdst, src0, in1, ir::OperandType::i64());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_NAND_B32: {
      auto res_ = ctx.create<BitAndOp>(createDst(), src0, src1, ir::OperandType::i32());
      auto res  = ctx.create<NotOp>(sdst, res_, ir::OperandType::i32());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_NAND_B64: {
      auto res_ = ctx.create<BitAndOp>(createDst(), src0, src1, ir::OperandType::i64());
      auto res  = ctx.create<NotOp>(sdst, res_, ir::OperandType::i64());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_NOR_B32: {
      auto res_ = ctx.create<BitOrOp>(createDst(), src0, src1, ir::OperandType::i32());
      auto res  = ctx.create<NotOp>(sdst, res_, ir::OperandType::i32());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_NOR_B64: {
      auto res_ = ctx.create<BitOrOp>(createDst(), src0, src1, ir::OperandType::i64());
      auto res  = ctx.create<NotOp>(sdst, res_, ir::OperandType::i64());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_XNOR_B32: {
      auto res_ = ctx.create<BitXorOp>(createDst(), src0, src1, ir::OperandType::i32());
      auto res  = ctx.create<NotOp>(sdst, res_, ir::OperandType::i32());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_XNOR_B64: {
      auto res_ = ctx.create<BitXorOp>(createDst(), src0, src1, ir::OperandType::i64());
      auto res  = ctx.create<NotOp>(sdst, res_, ir::OperandType::i64());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_LSHL_B32: {
      auto res = ctx.create<ShiftLUIOp>(sdst, src0, src1, ir::OperandType::i32());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_LSHL_B64: {
      auto res = ctx.create<ShiftLUIOp>(sdst, src0, src1, ir::OperandType::i64());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_LSHR_B32: {
      auto res = ctx.create<ShiftRUIOp>(sdst, src0, src1, ir::OperandType::i32());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_LSHR_B64: {
      auto res = ctx.create<ShiftRUIOp>(sdst, src0, src1, ir::OperandType::i64());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_ASHR_I32: {
      auto res = ctx.create<ShiftRSIOp>(sdst, src0, src1, ir::OperandType::i32());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_ASHR_I64: {
      auto res = ctx.create<ShiftRSIOp>(sdst, src0, src1, ir::OperandType::i64());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_BFM_B32: {
      auto res = ctx.create<BitFieldMaskOp>(sdst, src0, src1, ir::OperandType::i32());
    } break;
    case eOpcode::S_BFM_B64: {
      auto res = ctx.create<BitFieldMaskOp>(sdst, src0, src1, ir::OperandType::i64());
    } break;
    case eOpcode::S_MUL_I32: {
      auto res = ctx.create<MulIOp>(sdst, src0, src1, ir::OperandType::i32());
    } break;
    case eOpcode::S_BFE_U32: {
      auto res = ctx.create<BitUIExtractOp>(sdst, src0, src1, ir::OperandType::i32());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_BFE_I32: {
      auto res = ctx.create<BitSIExtractOp>(sdst, src0, src1, ir::OperandType::i32());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_BFE_U64: {
      auto res = ctx.create<BitSIExtractOp>(sdst, src0, src1, ir::OperandType::i64());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
    } break;
    case eOpcode::S_BFE_I64: {
      auto res = ctx.create<BitSIExtractOp>(sdst, src0, src1, ir::OperandType::i64());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i64(), CmpIPredicate::ne);
    } break;
    // case eOpcode::S_CBRANCH_G_FORK: { } break; // todo, make a block falltrough or handle data?
    case eOpcode::S_ABSDIFF_I32: {
      auto res_ = ctx.create<SubIOp>(sdst, src0, src1, ir::OperandType::i32());
      auto res  = ctx.create<AbsoluteOp>(sdst, res_, ir::OperandType::i32());
      ctx.create<CmpIOp>(createDst(eOperandKind::SCC()), res, createSrc(eOperandKind::createImm(0)), ir::OperandType::i32(), CmpIPredicate::ne);
    } break;
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }

  return conv(op);
}
} // namespace compiler::frontend::translate