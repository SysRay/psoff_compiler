#include "../debug_strings.h"
#include "../instruction_builder.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
InstructionKind_t handleDs(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = DS(getU64(*pCode));
  auto const op   = (parser::eOpcode)(OPcodeStart_DS + inst.template get<DS::Field::OP>());

  create::IRBuilder ir(ctx.instructions);
  *pCode += 2;

  switch (op) {
    case eOpcode::DS_ADD_U32: {
    } break;
    case eOpcode::DS_SUB_U32: {
    } break;
    case eOpcode::DS_RSUB_U32: {
    } break;
    case eOpcode::DS_INC_U32: {
    } break;
    case eOpcode::DS_DEC_U32: {
    } break;
    case eOpcode::DS_MIN_I32: {
    } break;
    case eOpcode::DS_MAX_I32: {
    } break;
    case eOpcode::DS_MIN_U32: {
    } break;
    case eOpcode::DS_MAX_U32: {
    } break;
    case eOpcode::DS_AND_B32: {
    } break;
    case eOpcode::DS_OR_B32: {
    } break;
    case eOpcode::DS_XOR_B32: {
    } break;
    case eOpcode::DS_MSKOR_B32: {
    } break;
    case eOpcode::DS_WRITE_B32: {
    } break;
    case eOpcode::DS_WRITE2_B32: {
    } break;
    case eOpcode::DS_WRITE2ST64_B32: {
    } break;
    case eOpcode::DS_CMPST_B32: {
    } break;
    case eOpcode::DS_CMPST_F32: {
    } break;
    case eOpcode::DS_MIN_F32: {
    } break;
    case eOpcode::DS_MAX_F32: {
    } break;
    case eOpcode::DS_NOP: {
    } break;
    case eOpcode::DS_GWS_SEMA_RELEASE_ALL: {
    } break;
    case eOpcode::DS_GWS_INIT: {
    } break;
    case eOpcode::DS_GWS_SEMA_V: {
    } break;
    case eOpcode::DS_GWS_SEMA_BR: {
    } break;
    case eOpcode::DS_GWS_SEMA_P: {
    } break;
    case eOpcode::DS_GWS_BARRIER: {
    } break;
    case eOpcode::DS_WRITE_B8: {
    } break;
    case eOpcode::DS_WRITE_B16: {
    } break;
    case eOpcode::DS_ADD_RTN_U32: {
    } break;
    case eOpcode::DS_SUB_RTN_U32: {
    } break;
    case eOpcode::DS_RSUB_RTN_U32: {
    } break;
    case eOpcode::DS_INC_RTN_U32: {
    } break;
    case eOpcode::DS_DEC_RTN_U32: {
    } break;
    case eOpcode::DS_MIN_RTN_I32: {
    } break;
    case eOpcode::DS_MAX_RTN_I32: {
    } break;
    case eOpcode::DS_MIN_RTN_U32: {
    } break;
    case eOpcode::DS_MAX_RTN_U32: {
    } break;
    case eOpcode::DS_AND_RTN_B32: {
    } break;
    case eOpcode::DS_OR_RTN_B32: {
    } break;
    case eOpcode::DS_XOR_RTN_B32: {
    } break;
    case eOpcode::DS_MSKOR_RTN_B32: {
    } break;
    case eOpcode::DS_WRXCHG_RTN_B32: {
    } break;
    case eOpcode::DS_WRXCHG2_RTN_B32: {
    } break;
    case eOpcode::DS_WRXCHG2ST64_RTN_B32: {
    } break;
    case eOpcode::DS_CMPST_RTN_B32: {
    } break;
    case eOpcode::DS_CMPST_RTN_F32: {
    } break;
    case eOpcode::DS_MIN_RTN_F32: {
    } break;
    case eOpcode::DS_MAX_RTN_F32: {
    } break;
    case eOpcode::DS_WRAP_RTN_B32: {
    } break;
    case eOpcode::DS_SWIZZLE_B32: {
    } break;
    case eOpcode::DS_READ_B32: {
    } break;
    case eOpcode::DS_READ2_B32: {
    } break;
    case eOpcode::DS_READ2ST64_B32: {
    } break;
    case eOpcode::DS_READ_I8: {
    } break;
    case eOpcode::DS_READ_U8: {
    } break;
    case eOpcode::DS_READ_I16: {
    } break;
    case eOpcode::DS_READ_U16: {
    } break;
    case eOpcode::DS_CONSUME: {
    } break;
    case eOpcode::DS_APPEND: {
    } break;
    case eOpcode::DS_ORDERED_COUNT: {
    } break;
    case eOpcode::DS_ADD_U64: {
    } break;
    case eOpcode::DS_SUB_U64: {
    } break;
    case eOpcode::DS_RSUB_U64: {
    } break;
    case eOpcode::DS_INC_U64: {
    } break;
    case eOpcode::DS_DEC_U64: {
    } break;
    case eOpcode::DS_MIN_I64: {
    } break;
    case eOpcode::DS_MAX_I64: {
    } break;
    case eOpcode::DS_MIN_U64: {
    } break;
    case eOpcode::DS_MAX_U64: {
    } break;
    case eOpcode::DS_AND_B64: {
    } break;
    case eOpcode::DS_OR_B64: {
    } break;
    case eOpcode::DS_XOR_B64: {
    } break;
    case eOpcode::DS_MSKOR_B64: {
    } break;
    case eOpcode::DS_WRITE_B64: {
    } break;
    case eOpcode::DS_WRITE2_B64: {
    } break;
    case eOpcode::DS_WRITE2ST64_B64: {
    } break;
    case eOpcode::DS_CMPST_B64: {
    } break;
    case eOpcode::DS_CMPST_F64: {
    } break;
    case eOpcode::DS_MIN_F64: {
    } break;
    case eOpcode::DS_MAX_F64: {
    } break;
    case eOpcode::DS_ADD_RTN_U64: {
    } break;
    case eOpcode::DS_SUB_RTN_U64: {
    } break;
    case eOpcode::DS_RSUB_RTN_U64: {
    } break;
    case eOpcode::DS_INC_RTN_U64: {
    } break;
    case eOpcode::DS_DEC_RTN_U64: {
    } break;
    case eOpcode::DS_MIN_RTN_I64: {
    } break;
    case eOpcode::DS_MAX_RTN_I64: {
    } break;
    case eOpcode::DS_MIN_RTN_U64: {
    } break;
    case eOpcode::DS_MAX_RTN_U64: {
    } break;
    case eOpcode::DS_AND_RTN_B64: {
    } break;
    case eOpcode::DS_OR_RTN_B64: {
    } break;
    case eOpcode::DS_XOR_RTN_B64: {
    } break;
    case eOpcode::DS_MSKOR_RTN_B64: {
    } break;
    case eOpcode::DS_WRXCHG_RTN_B64: {
    } break;
    case eOpcode::DS_WRXCHG2_RTN_B64: {
    } break;
    case eOpcode::DS_WRXCHG2ST64_RTN_B64: {
    } break;
    case eOpcode::DS_CMPST_RTN_B64: {
    } break;
    case eOpcode::DS_CMPST_RTN_F64: {
    } break;
    case eOpcode::DS_MIN_RTN_F64: {
    } break;
    case eOpcode::DS_MAX_RTN_F64: {
    } break;
    case eOpcode::DS_READ_B64: {
    } break;
    case eOpcode::DS_READ2_B64: {
    } break;
    case eOpcode::DS_READ2ST64_B64: {
    } break;
    case eOpcode::DS_CONDXCHG32_RTN_B64: {
    } break;
    case eOpcode::DS_ADD_SRC2_U32: {
    } break;
    case eOpcode::DS_SUB_SRC2_U32: {
    } break;
    case eOpcode::DS_RSUB_SRC2_U32: {
    } break;
    case eOpcode::DS_INC_SRC2_U32: {
    } break;
    case eOpcode::DS_DEC_SRC2_U32: {
    } break;
    case eOpcode::DS_MIN_SRC2_I32: {
    } break;
    case eOpcode::DS_MAX_SRC2_I32: {
    } break;
    case eOpcode::DS_MIN_SRC2_U32: {
    } break;
    case eOpcode::DS_MAX_SRC2_U32: {
    } break;
    case eOpcode::DS_AND_SRC2_B32: {
    } break;
    case eOpcode::DS_OR_SRC2_B32: {
    } break;
    case eOpcode::DS_XOR_SRC2_B32: {
    } break;
    case eOpcode::DS_WRITE_SRC2_B32: {
    } break;
    case eOpcode::DS_MIN_SRC2_F32: {
    } break;
    case eOpcode::DS_MAX_SRC2_F32: {
    } break;
    case eOpcode::DS_ADD_SRC2_U64: {
    } break;
    case eOpcode::DS_SUB_SRC2_U64: {
    } break;
    case eOpcode::DS_RSUB_SRC2_U64: {
    } break;
    case eOpcode::DS_INC_SRC2_U64: {
    } break;
    case eOpcode::DS_DEC_SRC2_U64: {
    } break;
    case eOpcode::DS_MIN_SRC2_I64: {
    } break;
    case eOpcode::DS_MAX_SRC2_I64: {
    } break;
    case eOpcode::DS_MIN_SRC2_U64: {
    } break;
    case eOpcode::DS_MAX_SRC2_U64: {
    } break;
    case eOpcode::DS_AND_SRC2_B64: {
    } break;
    case eOpcode::DS_OR_SRC2_B64: {
    } break;
    case eOpcode::DS_XOR_SRC2_B64: {
    } break;
    case eOpcode::DS_WRITE_SRC2_B64: {
    } break;
    case eOpcode::DS_MIN_SRC2_F64: {
    } break;
    case eOpcode::DS_MAX_SRC2_F64: {
    } break;
    case eOpcode::DS_WRITE_B96: {
    } break;
    case eOpcode::DS_WRITE_B128: {
    } break;
    case eOpcode::DS_CONDXCHG32_RTN_B128: {
    } break;
    case eOpcode::DS_READ_B96: {
    } break;
    case eOpcode::DS_READ_B128: {
    } break;
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }

  return conv(op);
}
} // namespace compiler::frontend::translate