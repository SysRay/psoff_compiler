#pragma once

#include "ir.h"

#include <string_view>
#include <tuple>

namespace compiler::ir {
// // Opcode definition
#define INSTRUCTION_LIST                                                                                                                                       \
  X(MoveOp, kUnknown, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32)))                                                                                               \
  X(SelectOp, kUnknown, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i1), OP(i32), OP(i32)))                                                                            \
  X(BitReverseOp, kBIT, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32)))                                                                                             \
  X(BitCountOp, kBIT, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32)))                                                                                               \
  X(BitAndOp, kBIT, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                        \
  X(BitOrOp, kBIT, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                         \
  X(BitXorOp, kBIT, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                        \
  X(BitFieldMaskOp, kBIT, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                  \
  X(FindILsbOp, kBIT, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32)))                                                                                               \
  X(FindUMsbOp, kBIT, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32)))                                                                                               \
  X(FindSMsbOp, kBIT, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32)))                                                                                               \
  X(SignExtendOp, kBIT, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32)))                                                                                             \
  X(BitsetOp, kBIT, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32), OP(i1)))                                                                                \
  X(BitCmpOp, kBIT, kNone, DST_OPS(OP(i1)), SRC_OPS(OP(i32), OP(i32)))                                                                                         \
  X(BitFieldInsertOp, kBIT, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32), OP(i32)))                                                                       \
  X(BitUIExtractOp, kBIT, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32), OP(i32)))                                                                         \
  X(BitSIExtractOp, kBIT, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32), OP(i32)))                                                                         \
  X(BitUIExtractCompactOp, kBIT, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                           \
  X(BitSIExtractCompactOp, kBIT, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                           \
  X(ShiftLUIOp, kBIT, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                      \
  X(ShiftRUIOp, kBIT, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                      \
  X(ShiftRSIOp, kBIT, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                      \
  X(CmpIOp, kALU, kNone, DST_OPS(OP(i1)), SRC_OPS(OP(i32), OP(i32)))                                                                                           \
  X(AddF32Op, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32), OP(f32)))                                                                                        \
  X(MulIOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                          \
  X(AddIOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                          \
  X(AddCarryIOp, kALU, kNone, DST_OPS(OP(i32), OP(i1)), SRC_OPS(OP(i32), OP(i32), OP(i1)))                                                                     \
  X(SubIOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                          \
  X(SubBurrowIOp, kALU, kNone, DST_OPS(OP(i32), OP(i1)), SRC_OPS(OP(i32), OP(i32), OP(i1)))                                                                    \
  X_NO_OPS(ReturnOp, kFlowControl, kHasSideEffects)                                                                                                            \
  X_NO_DST(JumpAbsOp, kFlowControl, kHasSideEffects, SRC_OPS(OP(i64)))                                                                                         \
  X_NO_SRC(ConstantOp, kConstant, kConstant, DST_OPS(OP(f32)))

// // Create table etc
enum class eInstKind : InstructionKind_t {
#define X(name, ...)        name,
#define X_NO_OPS(name, ...) name,
#define X_NO_SRC(name, ...) name,
#define X_NO_DST(name, ...) name,
  INSTRUCTION_LIST
#undef X
#undef X_NO_OPS
#undef X_NO_SRC
#undef X_NO_DST
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
  ir::InstCore {.kind        = conv(eInstKind::name),                                                                                                          \
                .group       = eInstructionGroup::instGroup,                                                                                                   \
                .flags       = Flags<eInstructionFlags>(eInstructionFlags::instFlags),                                                                         \
                .numDst      = std::tuple_size<decltype(std::make_tuple(dstOps))>::value,                                                                      \
                .numSrc      = std::tuple_size<decltype(std::make_tuple(srcOps))>::value,                                                                      \
                .dstOperands = {dstOps},                                                                                                                       \
                .srcOperands = {srcOps}},

#define X_NO_OPS(name, instGroup, instFlags)                                                                                                                   \
  ir::InstCore {.kind   = conv(eInstKind::name),                                                                                                               \
                .group  = eInstructionGroup::instGroup,                                                                                                        \
                .flags  = Flags<eInstructionFlags>(eInstructionFlags::instFlags),                                                                              \
                .numDst = 0,                                                                                                                                   \
                .numSrc = 0,                                                                                                                                   \
                .dstOperands {},                                                                                                                               \
                .srcOperands = {}},

#define X_NO_SRC(name, instGroup, instFlags, dstOps)                                                                                                           \
  ir::InstCore {.kind        = conv(eInstKind::name),                                                                                                          \
                .group       = eInstructionGroup::instGroup,                                                                                                   \
                .flags       = Flags<eInstructionFlags>(eInstructionFlags::instFlags),                                                                         \
                .numDst      = std::tuple_size<decltype(std::make_tuple(dstOps))>::value,                                                                      \
                .numSrc      = 0,                                                                                                                              \
                .dstOperands = {dstOps}},

#define X_NO_DST(name, instGroup, instFlags, srcOps)                                                                                                           \
  ir::InstCore {.kind        = conv(eInstKind::name),                                                                                                          \
                .group       = eInstructionGroup::instGroup,                                                                                                   \
                .flags       = Flags<eInstructionFlags>(eInstructionFlags::instFlags),                                                                         \
                .numDst      = 0,                                                                                                                              \
                .numSrc      = std::tuple_size<decltype(std::make_tuple(srcOps))>::value,                                                                      \
                .srcOperands = {srcOps}},
static constexpr InstCore kOpTable[] = {INSTRUCTION_LIST};

#undef SRC_OPS
#undef DST_OPS
#undef OP
#undef X
#undef X_NO_OPS
#undef X_NO_SRC
#undef X_NO_DST
} // namespace internal

inline constexpr const ir::InstCore& getInfo(eInstKind instr) {
  return internal::kOpTable[(std::underlying_type<eInstKind>::type)instr];
}
} // namespace compiler::ir