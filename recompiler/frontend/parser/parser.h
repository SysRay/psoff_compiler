#pragma once

#include "ir/ir.h"

namespace compiler {
class Builder;
}

namespace compiler::frontend::parser {
using pc_t      = uint64_t;
using code_t    = uint32_t;
using codeE_t   = uint64_t;
using code_p_t  = code_t const*;
using codeE_p_t = codeE_t const*;

bool parseInstruction(Builder& builder, pc_t, code_p_t*);

} // namespace compiler::frontend::parser