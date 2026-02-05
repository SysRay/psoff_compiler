#include "../gfx/encoding_types.h"
#include "../gfx/operand_types.h"
#include "../parser.h"
#include "opcodes_table.h"

#include <bitset>
#include <format>
#include <stdexcept>

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
      // src0 = createSrc(ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_u64 = **pCode}, ir::OperandType::i32()));
    }
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
      // src0 = createSrc(ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_u64 = **pCode}, ir::OperandType::i32()));
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
  eOpcode op;

  uint8_t size = sizeof(uint32_t);
  if (extended) {
    auto inst = VOP3_SDST(getU64(pCode));
    op        = (eOpcode)(OPcodeStart_VOPC + inst.template get<VOP3_SDST::Field::OP>() - OpcodeOffset_VOPC_VOP3);

    auto const sdst_ = eOperandKind((eOperandKind_t)inst.template get<VOP3_SDST::Field::SDST>());
    auto const src0_ = eOperandKind((eOperandKind_t)inst.template get<VOP3_SDST::Field::SRC0>());
    auto const src1_ = eOperandKind((eOperandKind_t)inst.template get<VOP3_SDST::Field::SRC1>());

    auto const           omod   = inst.template get<VOP3_SDST::Field::OMOD>();
    std::bitset<3> const negate = inst.template get<VOP3_SDST::Field::NEG>();

    size = sizeof(uint64_t);
  } else {
    auto inst = VOPC(*pCode);
    op        = (eOpcode)(OPcodeStart_VOPC + inst.template get<VOPC::Field::OP>());
    auto sdst = eOperandKind::VCC();
    auto src0 = eOperandKind((eOperandKind_t)inst.template get<VOPC::Field::SRC0>());
    auto src1 = eOperandKind::VGPR(inst.template get<VOPC::Field::VSRC1>());

    if (src0.isLiteral() || src1.isLiteral()) {
      size = sizeof(uint64_t);
      // src0 = createSrc(ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_u64 = **pCode}, ir::OperandType::i32()));
    }
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
    // src0 = createSrc(ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_u64 = **pCode}, ir::OperandType::i32()));
  }

  return size;
}
} // namespace compiler::frontend