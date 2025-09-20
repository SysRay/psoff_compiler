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
bool handleSmrd(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = SMRD(**pCode);
  auto const op   = (parser::eOpcode)(OPcodeStart_SMRD + inst.template get<SMRD::Field::OP>());

  auto const sdst      = (eOperandKind)inst.template get<SMRD::Field::SDST>();
  auto const sBase     = (eOperandKind)inst.template get<SMRD::Field::SBASE>();
  auto const offsetImm = inst.template get<SMRD::Field::OFFSET>();
  auto const sOffset   = (eOperandKind)offsetImm; // either imm or op
  auto const isImm     = (bool)inst.template get<SMRD::Field::IMM>();

  if (!isImm && sOffset == eOperandKind::Literal) {
    *pCode += 1;
    builder.createInstruction(create::literalOp(**pCode));
  }

  *pCode += 1;

  switch (op) {
    case eOpcode::S_LOAD_DWORD: {
    } break;
    case eOpcode::S_LOAD_DWORDX2: {
    } break;
    case eOpcode::S_LOAD_DWORDX4: {
    } break;
    case eOpcode::S_LOAD_DWORDX8: {
    } break;
    case eOpcode::S_LOAD_DWORDX16: {
    } break;
    case eOpcode::S_BUFFER_LOAD_DWORD: {
    } break;
    case eOpcode::S_BUFFER_LOAD_DWORDX2: {
    } break;
    case eOpcode::S_BUFFER_LOAD_DWORDX4: {
    } break;
    case eOpcode::S_BUFFER_LOAD_DWORDX8: {
    } break;
    case eOpcode::S_BUFFER_LOAD_DWORDX16: {
    } break;
    // case eOpcode::S_DCACHE_INV_VOL: {} break; // Does not exist
    case eOpcode::S_MEMTIME: {
    } break;
    case eOpcode::S_DCACHE_INV: {
    } break;
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }

  return true;
}
} // namespace compiler::frontend::translate