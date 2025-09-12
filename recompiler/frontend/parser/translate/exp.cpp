#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
bool handleExp(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  auto inst = EXP(getU64(*pCode));

  *pCode += 2;
  return true;
}
} // namespace compiler::frontend::translate