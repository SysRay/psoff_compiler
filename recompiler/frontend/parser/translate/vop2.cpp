#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
ir::InstCore handleVop2(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode, bool extended) {
  return {};
}
} // namespace compiler::frontend::translate