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
bool handleSopp(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = SOPP(**pCode);
  auto const op   = (parser::eOpcode)inst.template get<SOPP::Field::OP>();

  auto const offset = (int16_t)inst.template get<SOPP::Field::SIMM16>();

  *pCode += 1;

  switch (op) {
    case eOpcode::S_NOP: {
    } break;
    case eOpcode::S_ENDPGM: {
    } break;
    case eOpcode::S_BRANCH: {
    } break;
    case eOpcode::S_CBRANCH_SCC0: {
    } break;
    case eOpcode::S_CBRANCH_SCC1: {
    } break;
    case eOpcode::S_CBRANCH_VCCZ: {
    } break;
    case eOpcode::S_CBRANCH_VCCNZ: {
    } break;
    case eOpcode::S_CBRANCH_EXECZ: {
    } break;
    case eOpcode::S_CBRANCH_EXECNZ: {
    } break;
    case eOpcode::S_BARRIER: {
    } break;
    case eOpcode::S_SETKILL: {
    } break;
    case eOpcode::S_WAITCNT: {
    } break;
    case eOpcode::S_SETHALT: {
    } break;
    case eOpcode::S_SLEEP: {
    } break;
    case eOpcode::S_SETPRIO: {
    } break;
    case eOpcode::S_SENDMSG: {
    } break;
    // case eOpcode::S_SENDMSGHALT: {} break; // Does not exist
    // case eOpcode::S_TRAP: {} break; // Does not exist
    case eOpcode::S_ICACHE_INV: {
    } break;
    case eOpcode::S_INCPERFLEVEL: {
    } break;
    case eOpcode::S_DECPERFLEVEL: {
    } break;
    case eOpcode::S_TTRACEDATA: {
    } break;
    // case eOpcode::S_CBRANCH_CDBGSYS: {} break; // Does not exist
    // case eOpcode::S_CBRANCH_CDBGUSER: {} break; // Does not exist
    // case eOpcode::S_CBRANCH_CDBGSYS_OR_USER: {} break; // Does not exist
    // case eOpcode::S_CBRANCH_CDBGSYS_AND_USER: {} break;// Does not exist
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }
  return true;
}
} // namespace compiler::frontend::translate