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
  X(Max3UIOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32), OP(i32)))                                                                               \
  X(MaxUIOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                         \
  X(Max3SIOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32), OP(i32)))                                                                               \
  X(MaxSIOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                         \
  X(Max3FOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32), OP(f32), OP(f32)))                                                                                \
  X(MaxFOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32), OP(f32)))                                                                                          \
  X(MaxNOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32), OP(f32)))                                                                                          \
  X(MedUIOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                         \
  X(MedSIOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                         \
  X(MedFOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                          \
  X(MinUIOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                         \
  X(Min3UIOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32), OP(i32)))                                                                               \
  X(MinSIOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                         \
  X(Min3SIOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32), OP(i32)))                                                                               \
  X(MinFOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32), OP(f32)))                                                                                          \
  X(Min3FOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32), OP(f32), OP(f32)))                                                                                \
  X(MinNOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32), OP(f32)))                                                                                          \
  X(CmpIOp, kALU, kNone, DST_OPS(OP(i1)), SRC_OPS(OP(i32), OP(i32)))                                                                                           \
  X(CmpFOp, kALU, kNone, DST_OPS(OP(i1)), SRC_OPS(OP(f32), OP(f32)))                                                                                           \
  X(AddFOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32), OP(f32)))                                                                                          \
  X(SubFOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32), OP(f32)))                                                                                          \
  X(MulFOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32), OP(f32)))                                                                                          \
  X(FmaFOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32), OP(f32), OP(f32)))                                                                                 \
  X(FmaIOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32), OP(i32)))                                                                                 \
  X(MulIOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                          \
  X(MulIExtendedOp, kALU, kNone, DST_OPS(OP(i32), OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                         \
  X(AddIOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                          \
  X(AddCarryIOp, kALU, kNone, DST_OPS(OP(i32), OP(i1)), SRC_OPS(OP(i32), OP(i32), OP(i1)))                                                                     \
  X(SubIOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(i32), OP(i32)))                                                                                          \
  X(SubBurrowIOp, kALU, kNone, DST_OPS(OP(i32), OP(i1)), SRC_OPS(OP(i32), OP(i32), OP(i1)))                                                                    \
  X(FPToSIOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(f32)))                                                                                                 \
  X(SIToFPOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(i32)))                                                                                                 \
  X(FPToUIOp, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(f32)))                                                                                                 \
  X(UIToFPOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(i32)))                                                                                                 \
  X(SI4ToFloat, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(i32)))                                                                                               \
  X(TruncFOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f64)))                                                                                                 \
  X(ExtFOp, kALU, kNone, DST_OPS(OP(f64)), SRC_OPS(OP(f32)))                                                                                                   \
  X(TruncOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32)))                                                                                                  \
  X(CeilOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32)))                                                                                                   \
  X(RoundEvenOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32)))                                                                                              \
  X(FractOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32)))                                                                                                  \
  X(FloorOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32)))                                                                                                  \
  X(Exp2Op, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32)))                                                                                                   \
  X(Log2Op, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32)))                                                                                                   \
  X(RcpOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32)))                                                                                                    \
  X(RsqrtOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32)))                                                                                                  \
  X(SqrtOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32)))                                                                                                   \
  X(SinOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32)))                                                                                                    \
  X(CosOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32)))                                                                                                    \
  X(LdexpOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32), OP(i32)))                                                                                         \
  X(ClampFMinMaxOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32)))                                                                                           \
  X(ClampFZeroOp, kALU, kNone, DST_OPS(OP(f32)), SRC_OPS(OP(f32)))                                                                                             \
  X(FrexpOp, kALU, kNone, DST_OPS(OP(f32), OP(i32)), SRC_OPS(OP(f32)))                                                                                         \
  X(PackHalf2x16Op, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(f32), OP(f32)))                                                                                  \
  X(UnpackHalf2x16, kALU, kNone, DST_OPS(OP(f32), OP(f32)), SRC_OPS(OP(i32)))                                                                                  \
  X(PackSnorm2x16Op, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(f32), OP(f32)))                                                                                 \
  X(PackUnorm2x16Op, kALU, kNone, DST_OPS(OP(i32)), SRC_OPS(OP(f32), OP(f32)))                                                                                 \
  X_NO_OPS(ReturnOp, kFlowControl, kHasSideEffects)                                                                                                            \
  X_NO_DST(DiscardOp, kFlowControl, kHasSideEffects, SRC_OPS(OP(i1)))                                                                                          \
  X_NO_OPS(BarrierOp, kFlowControl, kHasSideEffects)                                                                                                           \
  X_NO_DST(JumpAbsOp, kFlowControl, kHasSideEffects, SRC_OPS(OP(i64)))                                                                                         \
  X_NO_DST(CondJumpAbsOp, kFlowControl, kHasSideEffects, SRC_OPS(OP(i1), OP(i64)))                                                                             \
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