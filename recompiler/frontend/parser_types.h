#pragma once

#include <cstdint>
#include <utility>

namespace compiler {
class Builder;

namespace ir {
class IROperations;
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
  ir::IROperations& instructions;

  Context(Builder& builder, ir::IROperations& instructions): builder(builder), instructions(instructions) {}

  template <typename Op, typename... Args>
  inline auto create(Args&&... args) {
    return Op::create(instructions, std::forward<Args>(args)...);
  }
};

} // namespace compiler::frontend::parser