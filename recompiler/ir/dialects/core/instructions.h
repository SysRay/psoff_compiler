#pragma once

#include "../types.h"
#include "ir/ir.h"

#include <array>
#include <string_view>
#include <tuple>

namespace compiler::ir::dialect::core {
enum class eInstKind {
  MoveOp,
  SelectOp,
  YieldOp,
  ReturnOp,
  DiscardOp,
  BarrierOp,
  JumpAbsOp,
  CondJumpAbsOp,
  ConstantOp,
};

constexpr inline InstructionKind_t conv(eInstKind code) {
  return (InstructionKind_t)code;
}

constexpr inline eInstKind conv(InstructionKind_t code) {
  return (eInstKind)code;
}

namespace internal {
using namespace compiler::ir::dialect::internal;

#define __OPS(...) __VA_ARGS__

#define __INST(kind, flags, dstOps, srcOps)                                                                                                                    \
  makeInstDef(eDialect::kCore, conv(eInstKind::kind), {eInstructionFlags::flags}, std::array {dstOps}, std::array {srcOps}, #kind)

#define __INST_NO_OPS(kind, flags)                                                                                                                             \
  makeInstDef(eDialect::kCore, conv(eInstKind::kind), {eInstructionFlags::flags}, std::array<OutputOperand, 0> {}, std::array<InputOperand, 0> {}, #kind)

#define __INST_NO_DST(kind, flags, srcOps)                                                                                                                     \
  makeInstDef(eDialect::kCore, conv(eInstKind::kind), {eInstructionFlags::flags}, std::array<OutputOperand, 0> {}, std::array {srcOps}, #kind)

#define __INST_NO_SRC(kind, flags, dstOps)                                                                                                                     \
  makeInstDef(eDialect::kCore, conv(eInstKind::kind), {eInstructionFlags::flags}, std::array {dstOps}, std::array<InputOperand, 0> {}, #kind)

static constexpr std::array kInstTable = {
    __INST(MoveOp, kNone, __OPS(o::i32), __OPS(i::i32)),  __INST(SelectOp, kNone, __OPS(o::i32), __OPS(i::i1, i::i32, i::i32)),
    __INST_NO_DST(YieldOp, kTerminator, __OPS(i::i64)),   __INST_NO_OPS(ReturnOp, kTerminator),
    __INST_NO_DST(DiscardOp, kTerminator, __OPS(i::i1)),  __INST_NO_OPS(BarrierOp, kBarrier),
    __INST_NO_DST(JumpAbsOp, kTerminator, __OPS(i::i64)), __INST_NO_DST(CondJumpAbsOp, kTerminator, __OPS(i::i1, i::i64)),
    __INST_NO_SRC(ConstantOp, kConstant, __OPS(o::i64)),
};

#undef __INST
#undef __INST_NO_OPS
#undef __INST_NO_DST
#undef __INST_NO_SRC

// Check if enum and array are in sync
template <std::size_t... I>
constexpr bool checkAll(std::index_sequence<I...>) {
  return ((static_cast<InstructionKind_t>(static_cast<eInstKind>(I)) == kInstTable[I].core.kind) && ...);
}

static_assert(checkAll(std::make_index_sequence<kInstTable.size()> {}), "eInstKind values must match kInstructionTable kinds");
} // namespace internal

inline constexpr const ir::InstCore& getInfo(eInstKind instr) {
  return internal::kInstTable[(std::underlying_type<eInstKind>::type)instr].core;
}

inline constexpr const std::string_view getInstrKindStr(eInstKind instr) {
  return internal::kInstTable[(std::underlying_type<eInstKind>::type)instr].name;
}

} // namespace compiler::ir::dialect::core