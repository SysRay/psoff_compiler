#pragma once

#include "ir.h"

#include <string_view>
#include <tuple>

namespace compiler::ir {
// // Opcode definition
#define INSTRUCTION_LIST                                                                                                                                       \
  X(MoveOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32)))                                                                                                   \
  X(AddF32Op, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32), OP(f32)))                                                                                        \
  X(AddI32Op, kALU, kNone, DST_OPS(OP(i32), OP(i1)), SRC_OPS(OP(i32), OP(i32), OP(i1)))                                                                        \
  X_NO_OPS(ReturnOp, kFlowControl, kHasSideEffects)

// // Create table etc
enum class eInstKind : InstructionKind_t {
#define X(name, ...)        name,
#define X_NO_OPS(name, ...) name,
  INSTRUCTION_LIST
#undef X
#undef X_NO_OPS
};

constexpr inline InstructionKind_t conv(eInstKind&& code) {
  return (InstructionKind_t)code;
}

namespace debug {
std::string_view getInstrKindStr(eInstKind code);
}

namespace internal {
#define OP(name)                                                                                                                                               \
  Operand {                                                                                                                                                    \
    .type = OperandType::name()                                                                                                                                \
  }

#define DST_OPS(...) __VA_ARGS__
#define SRC_OPS(...) __VA_ARGS__

#define X(name, instGroup, instFlags, dstOps, srcOps)                                                                                                          \
  ir::InstCore {.kind  = conv(eInstKind::name),                                                                                                                \
                .group = eInstructionGroup::instGroup,                                                                                                         \
                .flags = Flags<eInstructionFlags>(instFlags),                                                                                                  \
                .operands {.numDst   = std::tuple_size<decltype(std::make_tuple(dstOps))>::value,                                                              \
                           .numSrc   = std::tuple_size<decltype(std::make_tuple(srcOps))>::value,                                                              \
                           .operands = {dstOps, srcOps}}},

#define X_NO_OPS(name, instGroup, instFlags)                                                                                                                   \
  ir::InstCore {.kind  = conv(eInstKind::name),                                                                                                                \
                .group = eInstructionGroup::instGroup,                                                                                                         \
                .flags = Flags<eInstructionFlags>(instFlags),                                                                                                  \
                .operands {.numDst = 0, .numSrc = 0}},

static constexpr InstCore kOpTable[] = {INSTRUCTION_LIST};

#undef SRC_OPS
#undef DST_OPS
#undef OP
#undef X
#undef X_NO_OPS
} // namespace internal

inline constexpr const ir::InstCore& getInfo(eInstKind instr) {
  return internal::kOpTable[(std::underlying_type<eInstKind>::type)instr];
}
} // namespace compiler::ir