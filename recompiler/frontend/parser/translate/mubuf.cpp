#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
bool handleMubuf(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  auto       inst = MUBUF(getU64(*pCode));
  auto const op   = (parser::eOpcode)inst.template get<MUBUF::Field::OP>();

  *pCode += 2;
  return true;
}
} // namespace compiler::frontend::translate