#include "../debug_strings.h"
#include "../instruction_builder.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
InstructionKind_t handleMubuf(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = MUBUF(getU64(*pCode));
  auto const op   = (parser::eOpcode)(OPcodeStart_MUBUF + inst.template get<MUBUF::Field::OP>());

  create::IRBuilder ir(ctx.instructions);
  *pCode += 2;

  switch (op) {
    case eOpcode::BUFFER_LOAD_FORMAT_X: {
    } break;
    case eOpcode::BUFFER_LOAD_FORMAT_XY: {
    } break;
    case eOpcode::BUFFER_LOAD_FORMAT_XYZ: {
    } break;
    case eOpcode::BUFFER_LOAD_FORMAT_XYZW: {
    } break;
    case eOpcode::BUFFER_STORE_FORMAT_X: {
    } break;
    case eOpcode::BUFFER_STORE_FORMAT_XY: {
    } break;
    case eOpcode::BUFFER_STORE_FORMAT_XYZ: {
    } break;
    case eOpcode::BUFFER_STORE_FORMAT_XYZW: {
    } break;
    case eOpcode::BUFFER_LOAD_UBYTE: {
    } break;
    case eOpcode::BUFFER_LOAD_SBYTE: {
    } break;
    case eOpcode::BUFFER_LOAD_USHORT: {
    } break;
    case eOpcode::BUFFER_LOAD_SSHORT: {
    } break;
    case eOpcode::BUFFER_LOAD_DWORD: {
    } break;
    case eOpcode::BUFFER_LOAD_DWORDX2: {
    } break;
    case eOpcode::BUFFER_LOAD_DWORDX4: {
    } break;
    case eOpcode::BUFFER_LOAD_DWORDX3: {
    } break;
    case eOpcode::BUFFER_STORE_BYTE: {
    } break;
    case eOpcode::BUFFER_STORE_SHORT: {
    } break;
    case eOpcode::BUFFER_STORE_DWORD: {
    } break;
    case eOpcode::BUFFER_STORE_DWORDX2: {
    } break;
    case eOpcode::BUFFER_STORE_DWORDX4: {
    } break;
    case eOpcode::BUFFER_STORE_DWORDX3: {
    } break;
    case eOpcode::BUFFER_ATOMIC_SWAP: {
    } break;
    case eOpcode::BUFFER_ATOMIC_CMPSWAP: {
    } break;
    case eOpcode::BUFFER_ATOMIC_ADD: {
    } break;
    case eOpcode::BUFFER_ATOMIC_SUB: {
    } break;
    case eOpcode::BUFFER_ATOMIC_SMIN: {
    } break;
    case eOpcode::BUFFER_ATOMIC_UMIN: {
    } break;
    case eOpcode::BUFFER_ATOMIC_SMAX: {
    } break;
    case eOpcode::BUFFER_ATOMIC_UMAX: {
    } break;
    case eOpcode::BUFFER_ATOMIC_AND: {
    } break;
    case eOpcode::BUFFER_ATOMIC_OR: {
    } break;
    case eOpcode::BUFFER_ATOMIC_XOR: {
    } break;
    case eOpcode::BUFFER_ATOMIC_INC: {
    } break;
    case eOpcode::BUFFER_ATOMIC_DEC: {
    } break;
    case eOpcode::BUFFER_ATOMIC_FCMPSWAP: {
    } break;
    case eOpcode::BUFFER_ATOMIC_FMIN: {
    } break;
    case eOpcode::BUFFER_ATOMIC_FMAX: {
    } break;
    case eOpcode::BUFFER_ATOMIC_SWAP_X2: {
    } break;
    case eOpcode::BUFFER_ATOMIC_CMPSWAP_X2: {
    } break;
    case eOpcode::BUFFER_ATOMIC_ADD_X2: {
    } break;
    case eOpcode::BUFFER_ATOMIC_SUB_X2: {
    } break;
    case eOpcode::BUFFER_ATOMIC_SMIN_X2: {
    } break;
    case eOpcode::BUFFER_ATOMIC_UMIN_X2: {
    } break;
    case eOpcode::BUFFER_ATOMIC_SMAX_X2: {
    } break;
    case eOpcode::BUFFER_ATOMIC_UMAX_X2: {
    } break;
    case eOpcode::BUFFER_ATOMIC_AND_X2: {
    } break;
    case eOpcode::BUFFER_ATOMIC_OR_X2: {
    } break;
    case eOpcode::BUFFER_ATOMIC_XOR_X2: {
    } break;
    case eOpcode::BUFFER_ATOMIC_INC_X2: {
    } break;
    case eOpcode::BUFFER_ATOMIC_DEC_X2: {
    } break;
    case eOpcode::BUFFER_ATOMIC_FCMPSWAP_X2: {
    } break;
    case eOpcode::BUFFER_ATOMIC_FMIN_X2: {
    } break;
    case eOpcode::BUFFER_ATOMIC_FMAX_X2: {
    } break;
    case eOpcode::BUFFER_WBINVL1_SC: {
    } break;
    case eOpcode::BUFFER_WBINVL1: {
    } break;
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }
  return conv(op);
}
} // namespace compiler::frontend::translate