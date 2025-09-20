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
bool handleVop3(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  // todo sdst or not
  auto       inst = VOP3(getU64(*pCode));
  auto const op   = (parser::eOpcode)(OPcodeStart_VOP3 + inst.template get<VOP3::Field::OP>());

  auto const vdst = (eOperandKind)inst.template get<VOP3::Field::VDST>();
  auto const src0 = (eOperandKind)inst.template get<VOP3::Field::SRC0>();
  auto const src1 = (eOperandKind)inst.template get<VOP3::Field::SRC1>();
  auto const src2 = (eOperandKind)inst.template get<VOP3::Field::SRC2>();

  *pCode += 2;

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
  return true;
}
} // namespace compiler::frontend::translate