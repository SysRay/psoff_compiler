#pragma once

#include "ir/config.h"
#include "parser_types.h"

namespace compiler::frontend::parser {

InstructionKind_t parseInstruction(Context& ctx, pc_t, code_p_t*);

} // namespace compiler::frontend::parser