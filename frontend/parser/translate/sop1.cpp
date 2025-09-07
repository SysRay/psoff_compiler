#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
ir::InstCore handleSop1(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = SOP1(**pCode);
  auto const op   = (parser::eOpcode)inst.template get<SOP1::Field::OP>();

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
    case eOpcode::S_WQM_B64: break;
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
    // case eOpcode::S_SETPC_B64: break; // in branch
    //  case eOpcode::S_SWAPPC_B64: break; // in branch
    // case eOpcode::S_RFE_B64: break; // Does not exist
    case eOpcode::S_AND_SAVEEXEC_B64:
    case eOpcode::S_OR_SAVEEXEC_B64:
    case eOpcode::S_XOR_SAVEEXEC_B64:
    case eOpcode::S_ANDN2_SAVEEXEC_B64:
    case eOpcode::S_ORN2_SAVEEXEC_B64:
    case eOpcode::S_NAND_SAVEEXEC_B64:
    case eOpcode::S_NOR_SAVEEXEC_B64:
    case eOpcode::S_XNOR_SAVEEXEC_B64: {

    } break;
    // case eOpcode::S_QUADMASK_B32: return; // todo ? might be same as wqm
    // case eOpcode::S_QUADMASK_B64: return; // todo ? might be same as wqm
    case eOpcode::S_MOVRELS_B32: break;
    case eOpcode::S_MOVRELS_B64: break;
    case eOpcode::S_MOVRELD_B32: break;
    case eOpcode::S_MOVRELD_B64: break;
    // case eOpcode::S_CBRANCH_JOIN: break; // in branch
    //  case eOpcode::S_MOV_REGRD_B32: break; // Does not exist
    case eOpcode::S_ABS_I32: {

    } break;
    case eOpcode::S_MOV_FED_B32: break; // Does not exist
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }

  return {};
}
} // namespace compiler::frontend::translate