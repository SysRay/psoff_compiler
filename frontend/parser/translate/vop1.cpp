#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "encodings.h"
#include "frontend/shader_input.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
ir::InstCore handleVop1(ShaderInput const& ctx, parser::pc_t pc, parser::code_p_t* pCode, bool extended) {
  return {};
}
} // namespace compiler::frontend::translate