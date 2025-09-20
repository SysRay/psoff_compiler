#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
bool handleMtbuf(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = MTBUF(getU64(*pCode));
  auto const op   = (parser::eOpcode)(OPcodeStart_MTBUF + inst.template get<MTBUF::Field::OP>());

  *pCode += 2;

  switch (op) {
    case eOpcode::TBUFFER_LOAD_FORMAT_X: {
    } break;
    case eOpcode::TBUFFER_LOAD_FORMAT_XY: {
    } break;
    case eOpcode::TBUFFER_LOAD_FORMAT_XYZ: {
    } break;
    case eOpcode::TBUFFER_LOAD_FORMAT_XYZW: {
    } break;
    case eOpcode::TBUFFER_STORE_FORMAT_X: {
    } break;
    case eOpcode::TBUFFER_STORE_FORMAT_XY: {
    } break;
    case eOpcode::TBUFFER_STORE_FORMAT_XYZ: {
    } break;
    case eOpcode::TBUFFER_STORE_FORMAT_XYZW: {
    } break;

    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }
  return true;
}
} // namespace compiler::frontend::translate