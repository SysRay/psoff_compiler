#include "../../frontend.h"
#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "instruction_builder.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
bool handleVop2(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode, bool extended) {
  using namespace parser;

  parser::eOpcode op;
  eOperandKind    vdst0;
  eOperandKind    src0, src1;

  if (extended) { // todo sdst or not
    auto inst = VOP3(getU64(*pCode));
    op        = (parser::eOpcode)inst.template get<VOP3::Field::OP>();

    vdst0 = (eOperandKind)inst.template get<VOP3::Field::VDST>();
    src0  = (eOperandKind)inst.template get<VOP3::Field::SRC0>();
    src1  = (eOperandKind)inst.template get<VOP3::Field::SRC1>();
    *pCode += 1;
  } else {
    auto inst = VOP2(**pCode);
    op        = (parser::eOpcode)inst.template get<VOP2::Field::OP>();

    vdst0 = (eOperandKind)inst.template get<VOP2::Field::VDST>();
    src0  = (eOperandKind)inst.template get<VOP2::Field::SRC0>();
    src1  = (eOperandKind)inst.template get<VOP2::Field::VSRC1>();

    if (src0 == eOperandKind::Literal || src1 == eOperandKind::Literal) {
      *pCode += 1;
      builder.createInstruction(createLiteral(**pCode));
    }
  }

  *pCode += 1;

  switch (op) {
    case eOpcode::V_CNDMASK_B32: {

    } break;
    case eOpcode::V_ADD_F32: {

    } break;
    case eOpcode::V_SUB_F32: {

    } break;
    case eOpcode::V_SUBREV_F32: {

    } break;
    case eOpcode::V_MAC_LEGACY_F32: {

    } break;
    case eOpcode::V_MUL_LEGACY_F32: {

    } break;
    case eOpcode::V_MUL_F32: {

    } break;
    case eOpcode::V_MUL_I32_I24: {

    } break;
    case eOpcode::V_MUL_HI_I32_I24: {

    } break;
    case eOpcode::V_MUL_U32_U24: {

    } break;
    case eOpcode::V_MUL_HI_U32_U24: {

    } break;
    case eOpcode::V_MIN_LEGACY_F32: {

    } break;
    case eOpcode::V_MAX_LEGACY_F32: {

    } break;
    case eOpcode::V_MIN_F32: {

    } break;
    case eOpcode::V_MAX_F32: {

    } break;
    case eOpcode::V_MIN_I32: {

    } break;
    case eOpcode::V_MAX_I32: {

    } break;
    case eOpcode::V_MIN_U32: {

    } break;
    case eOpcode::V_MAX_U32: {

    } break;
    case eOpcode::V_LSHR_B32: {

    } break;
    case eOpcode::V_LSHRREV_B32: {

    } break;
    case eOpcode::V_ASHR_I32: {

    } break;
    case eOpcode::V_ASHRREV_I32: {

    } break;
    case eOpcode::V_LSHL_B32: {

    } break;
    case eOpcode::V_LSHLREV_B32: {

    } break;
    case eOpcode::V_AND_B32: {

    } break;
    case eOpcode::V_OR_B32: {

    } break;
    case eOpcode::V_XOR_B32: {

    } break;
    case eOpcode::V_BFM_B32: {

    } break;
    case eOpcode::V_MAC_F32: {

    } break;
    case eOpcode::V_MADMK_F32: {
      builder.createInstruction(createLiteral(**pCode));
      *pCode += 1;
    } break;
    case eOpcode::V_MADAK_F32: {
      builder.createInstruction(createLiteral(**pCode));
      *pCode += 1;
    } break;
    case eOpcode::V_BCNT_U32_B32: {

    } break;
    case eOpcode::V_MBCNT_LO_U32_B32: {

    } break;
    case eOpcode::V_MBCNT_HI_U32_B32: {

    } break;
    case eOpcode::V_ADD_I32: {

    } break;
    case eOpcode::V_SUB_I32: {

    } break;
    case eOpcode::V_SUBREV_I32: {

    } break;
    case eOpcode::V_ADDC_U32: {

    } break;
    case eOpcode::V_SUBB_U32: {

    } break;
    case eOpcode::V_SUBBREV_U32: {

    } break;
    case eOpcode::V_LDEXP_F32: {

    } break;
    // case eOpcode::V_CVT_PKACCUM_U8_F32: { // does not exist

    // } break;
    case eOpcode::V_CVT_PKNORM_I16_F32: {

    } break;
    case eOpcode::V_CVT_PKNORM_U16_F32: {

    } break;
    case eOpcode::V_CVT_PKRTZ_F16_F32: {

    } break;
    case eOpcode::V_CVT_PK_U16_U32: {

    } break;
    case eOpcode::V_CVT_PK_I16_I32: {

    } break;
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }
  return true;
}
} // namespace compiler::frontend::translate