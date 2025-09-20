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
bool handleVop1(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode, bool extended) {
  using namespace parser;

  parser::eOpcode op;
  eOperandKind    vdst;
  eOperandKind    src0;

  uint8_t omod   = 0;
  bool    negate = false;

  if (extended) { // todo sdst or not
    auto inst = VOP3(getU64(*pCode));
    op        = (parser::eOpcode)(OPcodeStart_VOP1 + inst.template get<VOP3::Field::OP>() - OpcodeOffset_VOP1_VOP3);

    vdst = (eOperandKind)inst.template get<VOP3::Field::VDST>();
    src0 = (eOperandKind)inst.template get<VOP3::Field::SRC0>();

    omod   = inst.template get<VOP3::Field::OMOD>();
    negate = inst.template get<VOP3::Field::NEG>();
    *pCode += 1;
  } else {
    auto inst = VOP1(**pCode);
    op        = (parser::eOpcode)(OPcodeStart_VOP1 + inst.template get<VOP1::Field::OP>());

    vdst = (eOperandKind)inst.template get<VOP1::Field::VDST>();
    src0 = (eOperandKind)inst.template get<VOP1::Field::SRC0>();
    if (src0 == eOperandKind::Literal) {
      *pCode += 1;
      builder.createInstruction(create::literalOp(**pCode));
    }
  }

  *pCode += 1;

  switch (op) {
    case eOpcode::V_NOP: break; // ignore
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
  return true;
}
} // namespace compiler::frontend::translate