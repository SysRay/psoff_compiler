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
InstructionKind_t handleVopc(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode, bool extended) {
  using namespace parser;

  parser::eOpcode op;
  OpDst           sdst;
  OpSrc           src0, src1;

  if (extended) {
    auto inst = VOP3_SDST(getU64(*pCode));
    op        = (parser::eOpcode)(OPcodeStart_VOPC + inst.template get<VOP3_SDST::Field::OP>() - OpcodeOffset_VOPC_VOP3);

    auto const sdst_ = eOperandKind((eOperandKind_t)inst.template get<VOP3_SDST::Field::SDST>());
    auto const src0_ = eOperandKind((eOperandKind_t)inst.template get<VOP3_SDST::Field::SRC0>());
    auto const src1_ = eOperandKind((eOperandKind_t)inst.template get<VOP3_SDST::Field::SRC1>());

    auto const           omod   = inst.template get<VOP3_SDST::Field::OMOD>();
    std::bitset<3> const negate = inst.template get<VOP3_SDST::Field::NEG>();

    src0 = OpSrc(src0_, negate[0], false);
    src1 = OpSrc(src1_, negate[1], false);
    sdst = OpDst(sdst_, omod, false, false);
    *pCode += 1;
  } else {
    auto inst = VOPC(**pCode);
    op        = (parser::eOpcode)(OPcodeStart_VOPC + inst.template get<VOPC::Field::OP>());
    sdst      = OpDst(eOperandKind::VCC());
    src0      = OpSrc(eOperandKind((eOperandKind_t)inst.template get<VOPC::Field::SRC0>()));
    src1      = OpSrc(eOperandKind::VGPR(inst.template get<VOPC::Field::VSRC1>()));

    if (src0.kind.isLiteral() || src1.kind.isLiteral()) {
      *pCode += 1;
      builder.createInstruction(create::literalOp(**pCode));
    }
  }

  *pCode += 1;

  constexpr std::array cmpOpsF = {CmpFPredicate::AlwaysFalse, CmpFPredicate::OLT, CmpFPredicate::OEQ, CmpFPredicate::OLE,
                                  CmpFPredicate::OGT,         CmpFPredicate::ONE, CmpFPredicate::OGE, CmpFPredicate::ORD,
                                  CmpFPredicate::UNO,         CmpFPredicate::OLT, CmpFPredicate::OEQ, CmpFPredicate::OLE,
                                  CmpFPredicate::OGT,         CmpFPredicate::ONE, CmpFPredicate::OGE, CmpFPredicate::AlwaysTrue};

  constexpr std::array cmdOpsSI = {CmpIPredicate::AlwaysFalse, CmpIPredicate::slt, CmpIPredicate::eq,  CmpIPredicate::sle,
                                   CmpIPredicate::sgt,         CmpIPredicate::ne,  CmpIPredicate::sge, CmpIPredicate::AlwaysTrue};

  constexpr std::array cmdOpsUI = {CmpIPredicate::AlwaysFalse, CmpIPredicate::ult, CmpIPredicate::eq,  CmpIPredicate::ule,
                                   CmpIPredicate::ugt,         CmpIPredicate::ne,  CmpIPredicate::uge, CmpIPredicate::AlwaysTrue};

  if (op >= eOpcode::V_CMP_F_F32 && op <= eOpcode::V_CMP_T_F32) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMP_F_F32;
    builder.createInstruction(create::cmpFOp(OpDst(sdst.kind), src0, src1, ir::OperandType::f32(), cmpOpsF[opIndex]));
    builder.createInstruction(create::bitAndOp(sdst, OpSrc(sdst.kind), OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
  } else if (op >= eOpcode::V_CMPX_F_F32 && op <= eOpcode::V_CMPX_T_F32) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPX_F_F32;
    builder.createInstruction(create::cmpFOp(OpDst(sdst.kind), src0, src1, ir::OperandType::f32(), cmpOpsF[opIndex]));
    builder.createInstruction(create::bitAndOp(sdst, OpSrc(sdst.kind), OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
    builder.createInstruction(create::moveOp(OpDst(eOperandKind::EXEC()), OpSrc(sdst.kind), ir::OperandType::i1()));
  } else if (op >= eOpcode::V_CMP_F_F64 && op <= eOpcode::V_CMP_T_F64) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMP_F_F64;
    builder.createInstruction(create::cmpFOp(OpDst(sdst.kind), src0, src1, ir::OperandType::f64(), cmpOpsF[opIndex]));
    builder.createInstruction(create::bitAndOp(sdst, OpSrc(sdst.kind), OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
  } else if (op >= eOpcode::V_CMPX_F_F64 && op <= eOpcode::V_CMPX_T_F64) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPX_F_F32;
    builder.createInstruction(create::cmpFOp(OpDst(sdst.kind), src0, src1, ir::OperandType::f64(), cmpOpsF[opIndex]));
    builder.createInstruction(create::bitAndOp(sdst, OpSrc(sdst.kind), OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
    builder.createInstruction(create::moveOp(OpDst(eOperandKind::EXEC()), OpSrc(sdst.kind), ir::OperandType::i1()));
  } else if (op >= eOpcode::V_CMPS_F_F32 && op <= eOpcode::V_CMPS_T_F32) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPS_F_F32;
    builder.createInstruction(create::cmpFOp(OpDst(sdst.kind), src0, src1, ir::OperandType::f32(), cmpOpsF[opIndex]));
    builder.createInstruction(create::bitAndOp(sdst, OpSrc(sdst.kind), OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
  } else if (op >= eOpcode::V_CMPSX_F_F32 && op <= eOpcode::V_CMPSX_T_F32) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPSX_F_F32;
    builder.createInstruction(create::cmpFOp(OpDst(sdst.kind), src0, src1, ir::OperandType::f32(), cmpOpsF[opIndex]));
    builder.createInstruction(create::bitAndOp(sdst, OpSrc(sdst.kind), OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
    builder.createInstruction(create::moveOp(OpDst(eOperandKind::EXEC()), OpSrc(sdst.kind), ir::OperandType::i1()));
  } else if (op >= eOpcode::V_CMPS_F_F64 && op <= eOpcode::V_CMPS_T_F64) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPS_F_F64;
    builder.createInstruction(create::cmpFOp(OpDst(sdst.kind), src0, src1, ir::OperandType::f64(), cmpOpsF[opIndex]));
    builder.createInstruction(create::bitAndOp(sdst, OpSrc(sdst.kind), OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
  } else if (op >= eOpcode::V_CMPSX_F_F64 && op <= eOpcode::V_CMPSX_T_F64) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPSX_F_F64;
    builder.createInstruction(create::cmpFOp(OpDst(sdst.kind), src0, src1, ir::OperandType::f64(), cmpOpsF[opIndex]));
    builder.createInstruction(create::bitAndOp(sdst, OpSrc(sdst.kind), OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
    builder.createInstruction(create::moveOp(OpDst(eOperandKind::EXEC()), OpSrc(sdst.kind), ir::OperandType::i1()));
  } //
  else if (op >= eOpcode::V_CMP_F_I32 && op <= eOpcode::V_CMP_T_I32) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMP_F_I32;
    builder.createInstruction(create::cmpIOp(OpDst(sdst.kind), src0, src1, ir::OperandType::i32(), cmdOpsSI[opIndex]));
    builder.createInstruction(create::bitAndOp(sdst, OpSrc(sdst.kind), OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
  } else if (op >= eOpcode::V_CMP_CLASS_F32) { // todo
  } else if (op >= eOpcode::V_CMPX_F_I32 && op <= eOpcode::V_CMPX_T_I32) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPX_F_I32;
    builder.createInstruction(create::cmpIOp(OpDst(sdst.kind), src0, src1, ir::OperandType::i32(), cmdOpsSI[opIndex]));
    builder.createInstruction(create::bitAndOp(sdst, OpSrc(sdst.kind), OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
    builder.createInstruction(create::moveOp(OpDst(eOperandKind::EXEC()), OpSrc(sdst.kind), ir::OperandType::i1()));
  } else if (op >= eOpcode::V_CMPX_CLASS_F32) { // todo
  } else if (op >= eOpcode::V_CMP_F_I64 && op <= eOpcode::V_CMP_T_I64) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMP_F_I64;
    builder.createInstruction(create::cmpIOp(OpDst(sdst.kind), src0, src1, ir::OperandType::i32(), cmdOpsSI[opIndex]));
    builder.createInstruction(create::bitAndOp(sdst, OpSrc(sdst.kind), OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
  } else if (op >= eOpcode::V_CMP_CLASS_F64) { // todo
  } else if (op >= eOpcode::V_CMPX_F_I64 && op <= eOpcode::V_CMPX_T_I64) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPX_F_I64;
    builder.createInstruction(create::cmpIOp(OpDst(sdst.kind), src0, src1, ir::OperandType::i32(), cmdOpsSI[opIndex]));
    builder.createInstruction(create::bitAndOp(sdst, OpSrc(sdst.kind), OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
    builder.createInstruction(create::moveOp(OpDst(eOperandKind::EXEC()), OpSrc(sdst.kind), ir::OperandType::i1()));
  } else if (op >= eOpcode::V_CMPX_CLASS_F64) { // todo
  } //
  else if (op >= eOpcode::V_CMP_F_U32 && op <= eOpcode::V_CMP_T_U32) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMP_F_U32;
    builder.createInstruction(create::cmpIOp(OpDst(sdst.kind), src0, src1, ir::OperandType::i32(), cmdOpsUI[opIndex]));
    builder.createInstruction(create::bitAndOp(sdst, OpSrc(sdst.kind), OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
  } else if (op >= eOpcode::V_CMPX_F_U32 && op <= eOpcode::V_CMPX_T_U32) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPX_F_U32;
    builder.createInstruction(create::cmpIOp(OpDst(sdst.kind), src0, src1, ir::OperandType::i32(), cmdOpsUI[opIndex]));
    builder.createInstruction(create::bitAndOp(sdst, OpSrc(sdst.kind), OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
    builder.createInstruction(create::moveOp(OpDst(eOperandKind::EXEC()), OpSrc(sdst.kind), ir::OperandType::i1()));
  } else if (op >= eOpcode::V_CMP_F_U64 && op <= eOpcode::V_CMP_T_U64) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMP_F_U64;
    builder.createInstruction(create::cmpIOp(OpDst(sdst.kind), src0, src1, ir::OperandType::i32(), cmdOpsUI[opIndex]));
    builder.createInstruction(create::bitAndOp(sdst, OpSrc(sdst.kind), OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
  } else if (op >= eOpcode::V_CMPX_F_U64 && op <= eOpcode::V_CMPX_T_U64) {
    auto const opIndex = (InstructionKind_t)op - (InstructionKind_t)eOpcode::V_CMPX_F_U64;
    builder.createInstruction(create::cmpIOp(OpDst(sdst.kind), src0, src1, ir::OperandType::i32(), cmdOpsUI[opIndex]));
    builder.createInstruction(create::bitAndOp(sdst, OpSrc(sdst.kind), OpSrc(eOperandKind::EXEC()), ir::OperandType::i1()));
    builder.createInstruction(create::moveOp(OpDst(eOperandKind::EXEC()), OpSrc(sdst.kind), ir::OperandType::i1()));
  } else {
    throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op)));
  }

  return conv(op);
}
} // namespace compiler::frontend::translate