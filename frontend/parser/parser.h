#pragma once

#include "ir/ir.h"

namespace compiler::frontend {
struct ShaderInput;
}

namespace compiler::frontend::parser {
using pc_t      = uint64_t;
using code_t    = uint32_t;
using codeE_t   = uint64_t;
using code_p_t  = code_t const*;
using codeE_p_t = codeE_t const*;

ir::InstCore parseInstruction(ShaderInput const&, pc_t, code_p_t*);

} // namespace compiler::frontend::parser