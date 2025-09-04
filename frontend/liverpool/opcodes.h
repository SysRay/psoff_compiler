#pragma once

#include "../types.h"

namespace compiler::frontend::liverpool {

enum class Opcode : uint16_t {
#define X(name, ...) name,
#include "opcodes.inc"
#undef X
  Count
};

namespace internal {
static constexpr OpInfo kOpTable[] = {
#define xstr(s) str(s)
#define str(s)  #s

#define X(name, operands, result, src_types, flags) {xstr(name), operands, result, src_types, flags},
    OPCODE_LIST
#undef X

#undef str
#undef xstr
};

#undef IN

} // namespace internal

inline constexpr const OpInfo& getOpInfo(Opcode op) {
  return internal::kOpTable[(std::underlying_type<Opcode>::type)op];
}
} // namespace compiler::frontend::liverpool