#pragma once

#include <cstdint>

namespace compiler {
class Builder;

namespace ir {
class InstructionManager;
}
} // namespace compiler

namespace compiler::frontend::parser {
using pc_t      = uint64_t;
using code_t    = uint32_t;
using codeE_t   = uint64_t;
using code_p_t  = code_t const*;
using codeE_p_t = codeE_t const*;

struct Context {
  Builder&                builder;
  ir::InstructionManager& instructions;

  Context(Builder& builder, ir::InstructionManager& instructions): builder(builder), instructions(instructions) {}
};

} // namespace compiler::frontend::parser