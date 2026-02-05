#include "../debug_strings.h"
#include "../gfx/encoding_types.h"
#include "../gfx/operand_types.h"
#include "../parser.h"
#include "opcodes_table.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend {

uint8_t Parser::handleMimg(CodeBlock& cb, pc_t pc, uint32_t const* pCode) {
  auto       inst = MIMG(*pCode);
  auto const op   = (eOpcode)(OPcodeStart_MIMG + inst.template get<MIMG::Field::OP>());

  switch (op) {
    case eOpcode::IMAGE_LOAD: {
    } break;
    case eOpcode::IMAGE_LOAD_MIP: {
    } break;
    case eOpcode::IMAGE_LOAD_PCK: {
    } break;
    case eOpcode::IMAGE_LOAD_PCK_SGN: {
    } break;
    case eOpcode::IMAGE_LOAD_MIP_PCK: {
    } break;
    case eOpcode::IMAGE_LOAD_MIP_PCK_SGN: {
    } break;
    case eOpcode::IMAGE_STORE: {
    } break;
    case eOpcode::IMAGE_STORE_MIP: {
    } break;
    case eOpcode::IMAGE_STORE_PCK: {
    } break;
    case eOpcode::IMAGE_STORE_MIP_PCK: {
    } break;
    case eOpcode::IMAGE_GET_RESINFO: {
    } break;
    case eOpcode::IMAGE_ATOMIC_SWAP: {
    } break;
    case eOpcode::IMAGE_ATOMIC_CMPSWAP: {
    } break;
    case eOpcode::IMAGE_ATOMIC_ADD: {
    } break;
    case eOpcode::IMAGE_ATOMIC_SUB: {
    } break;
    case eOpcode::IMAGE_ATOMIC_SMIN: {
    } break;
    case eOpcode::IMAGE_ATOMIC_UMIN: {
    } break;
    case eOpcode::IMAGE_ATOMIC_SMAX: {
    } break;
    case eOpcode::IMAGE_ATOMIC_UMAX: {
    } break;
    case eOpcode::IMAGE_ATOMIC_AND: {
    } break;
    case eOpcode::IMAGE_ATOMIC_OR: {
    } break;
    case eOpcode::IMAGE_ATOMIC_XOR: {
    } break;
    case eOpcode::IMAGE_ATOMIC_INC: {
    } break;
    case eOpcode::IMAGE_ATOMIC_DEC: {
    } break;
    case eOpcode::IMAGE_ATOMIC_FCMPSWAP: {
    } break;
    case eOpcode::IMAGE_ATOMIC_FMIN: {
    } break;
    case eOpcode::IMAGE_ATOMIC_FMAX: {
    } break;
    case eOpcode::IMAGE_SAMPLE: {
    } break;
    case eOpcode::IMAGE_SAMPLE_CL: {
    } break;
    case eOpcode::IMAGE_SAMPLE_D: {
    } break;
    case eOpcode::IMAGE_SAMPLE_D_CL: {
    } break;
    case eOpcode::IMAGE_SAMPLE_L: {
    } break;
    case eOpcode::IMAGE_SAMPLE_B: {
    } break;
    case eOpcode::IMAGE_SAMPLE_B_CL: {
    } break;
    case eOpcode::IMAGE_SAMPLE_LZ: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C_CL: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C_D: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C_D_CL: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C_L: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C_B: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C_B_CL: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C_LZ: {
    } break;
    case eOpcode::IMAGE_SAMPLE_O: {
    } break;
    case eOpcode::IMAGE_SAMPLE_CL_O: {
    } break;
    case eOpcode::IMAGE_SAMPLE_D_O: {
    } break;
    case eOpcode::IMAGE_SAMPLE_D_CL_O: {
    } break;
    case eOpcode::IMAGE_SAMPLE_L_O: {
    } break;
    case eOpcode::IMAGE_SAMPLE_B_O: {
    } break;
    case eOpcode::IMAGE_SAMPLE_B_CL_O: {
    } break;
    case eOpcode::IMAGE_SAMPLE_LZ_O: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C_O: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C_CL_O: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C_D_O: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C_D_CL_O: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C_L_O: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C_B_O: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C_B_CL_O: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C_LZ_O: {
    } break;
    case eOpcode::IMAGE_GATHER4: {
    } break;
    case eOpcode::IMAGE_GATHER4_CL: {
    } break;
    case eOpcode::IMAGE_GATHER4_L: {
    } break;
    case eOpcode::IMAGE_GATHER4_B: {
    } break;
    case eOpcode::IMAGE_GATHER4_B_CL: {
    } break;
    case eOpcode::IMAGE_GATHER4_LZ: {
    } break;
    case eOpcode::IMAGE_GATHER4_C: {
    } break;
    case eOpcode::IMAGE_GATHER4_C_CL: {
    } break;
    case eOpcode::IMAGE_GATHER4_C_L: {
    } break;
    case eOpcode::IMAGE_GATHER4_C_B: {
    } break;
    case eOpcode::IMAGE_GATHER4_C_B_CL: {
    } break;
    case eOpcode::IMAGE_GATHER4_C_LZ: {
    } break;
    case eOpcode::IMAGE_GATHER4_O: {
    } break;
    case eOpcode::IMAGE_GATHER4_CL_O: {
    } break;
    case eOpcode::IMAGE_GATHER4_L_O: {
    } break;
    case eOpcode::IMAGE_GATHER4_B_O: {
    } break;
    case eOpcode::IMAGE_GATHER4_B_CL_O: {
    } break;
    case eOpcode::IMAGE_GATHER4_LZ_O: {
    } break;
    case eOpcode::IMAGE_GATHER4_C_O: {
    } break;
    case eOpcode::IMAGE_GATHER4_C_CL_O: {
    } break;
    case eOpcode::IMAGE_GATHER4_C_L_O: {
    } break;
    case eOpcode::IMAGE_GATHER4_C_B_O: {
    } break;
    case eOpcode::IMAGE_GATHER4_C_B_CL_O: {
    } break;
    case eOpcode::IMAGE_GATHER4_C_LZ_O: {
    } break;
    case eOpcode::IMAGE_GET_LOD: {
    } break;
    case eOpcode::IMAGE_SAMPLE_CD: {
    } break;
    case eOpcode::IMAGE_SAMPLE_CD_CL: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C_CD: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C_CD_CL: {
    } break;
    case eOpcode::IMAGE_SAMPLE_CD_O: {
    } break;
    case eOpcode::IMAGE_SAMPLE_CD_CL_O: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C_CD_O: {
    } break;
    case eOpcode::IMAGE_SAMPLE_C_CD_CL_O: {
    } break;
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }

  return sizeof(uint64_t);
}

} // namespace compiler::frontend