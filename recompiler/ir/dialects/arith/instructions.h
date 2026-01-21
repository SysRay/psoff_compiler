#pragma once

#include "../types.h"
#include "ir/ir.h"

#include <array>
#include <string_view>
#include <tuple>

namespace compiler::ir::dialect::arith {
enum class eInstKind {
  BitReverseOp,
  BitCountOp,
  BitAndOp,
  BitOrOp,
  BitXorOp,
  BitFieldMaskOp,
  FindILsbOp,
  FindUMsbOp,
  FindSMsbOp,
  SignExtendOp,
  BitsetOp,
  BitCmpOp,
  BitFieldInsertOp,
  BitUIExtractOp,
  BitSIExtractOp,
  BitUIExtractCompactOp,
  BitSIExtractCompactOp,
  ShiftLUIOp,
  ShiftRUIOp,
  ShiftRSIOp,
  Max3UIOp,
  MaxUIOp,
  Max3SIOp,
  MaxSIOp,
  Max3FOp,
  MaxFOp,
  MaxNOp,
  MedUIOp,
  MedSIOp,
  MedFOp,
  MinUIOp,
  Min3UIOp,
  MinSIOp,
  Min3SIOp,
  MinFOp,
  Min3FOp,
  MinNOp,
  CmpIOp,
  CmpFOp,
  AddFOp,
  SubFOp,
  MulFOp,
  FmaFOp,
  FmaIOp,
  MulIOp,
  MulIExtendedOp,
  AddIOp,
  AddCarryIOp,
  SubIOp,
  SubBurrowIOp,
  FPToSIOp,
  SIToFPOp,
  FPToUIOp,
  UIToFPOp,
  SI4ToFloat,
  TruncFOp,
  ExtFOp,
  TruncOp,
  CeilOp,
  RoundEvenOp,
  FractOp,
  FloorOp,
  Exp2Op,
  Log2Op,
  RcpOp,
  RsqrtOp,
  SqrtOp,
  SinOp,
  CosOp,
  LdexpOp,
  ClampFMinMaxOp,
  ClampFZeroOp,
  FrexpOp,
  PackHalf2x16Op,
  UnpackHalf2x16,
  PackSnorm2x16Op,
  PackUnorm2x16Op,
  NegateFOp,
  AbsoluteOp,
  NotOp,
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
  makeInstDef(eDialect::kArith, conv(eInstKind::kind), {eInstructionFlags::flags}, std::array {dstOps}, std::array {srcOps}, #kind)

#define __INST_NO_OPS(kind, flags)                                                                                                                             \
  makeInstDef(eDialect::kArith, conv(eInstKind::kind), {eInstructionFlags::flags}, std::array<OutputOperand, 0> {}, std::array<InputOperand, 0> {}, #kind)

#define __INST_NO_DST(kind, flags, srcOps)                                                                                                                     \
  makeInstDef(eDialect::kArith, conv(eInstKind::kind), {eInstructionFlags::flags}, std::array<OutputOperand, 0> {}, std::array {srcOps}, #kind)

#define __INST_NO_SRC(kind, flags, dstOps)                                                                                                                     \
  makeInstDef(eDialect::kArith, conv(eInstKind::kind), {eInstructionFlags::flags}, std::array {dstOps}, std::array<InputOperand, 0> {}, #kind)

static constexpr std::array kInstTable = {
    __INST(BitReverseOp, kNone, __OPS(o::i32), __OPS(i::i32)),
    __INST(BitCountOp, kNone, __OPS(o::i32), __OPS(i::i32)),
    __INST(BitAndOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32)),
    __INST(BitOrOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32)),
    __INST(BitXorOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32)),
    __INST(BitFieldMaskOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32)),
    __INST(FindILsbOp, kNone, __OPS(o::i32), __OPS(i::i32)),
    __INST(FindUMsbOp, kNone, __OPS(o::i32), __OPS(i::i32)),
    __INST(FindSMsbOp, kNone, __OPS(o::i32), __OPS(i::i32)),
    __INST(SignExtendOp, kNone, __OPS(o::i32), __OPS(i::i32)),
    __INST(BitsetOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32, i::i1)),
    __INST(BitCmpOp, kNone, __OPS(o::i1), __OPS(i::i32, i::i32)),
    __INST(BitFieldInsertOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32, i::i32)),
    __INST(BitUIExtractOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32, i::i32)),
    __INST(BitSIExtractOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32, i::i32)),
    __INST(BitUIExtractCompactOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32)),
    __INST(BitSIExtractCompactOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32)),
    __INST(ShiftLUIOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32)),
    __INST(ShiftRUIOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32)),
    __INST(ShiftRSIOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32)),
    __INST(Max3UIOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32, i::i32)),
    __INST(MaxUIOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32)),
    __INST(Max3SIOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32, i::i32)),
    __INST(MaxSIOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32)),
    __INST(Max3FOp, kNone, __OPS(o::f32), __OPS(i::f32, i::f32, i::f32)),
    __INST(MaxFOp, kNone, __OPS(o::f32), __OPS(i::f32, i::f32)),
    __INST(MaxNOp, kNone, __OPS(o::f32), __OPS(i::f32, i::f32)),
    __INST(MedUIOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32)),
    __INST(MedSIOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32)),
    __INST(MedFOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32)),
    __INST(MinUIOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32)),
    __INST(Min3UIOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32, i::i32)),
    __INST(MinSIOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32)),
    __INST(Min3SIOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32, i::i32)),
    __INST(MinFOp, kNone, __OPS(o::f32), __OPS(i::f32, i::f32)),
    __INST(Min3FOp, kNone, __OPS(o::f32), __OPS(i::f32, i::f32, i::f32)),
    __INST(MinNOp, kNone, __OPS(o::f32), __OPS(i::f32, i::f32)),
    __INST(CmpIOp, kNone, __OPS(o::i1), __OPS(i::i32, i::i32)),
    __INST(CmpFOp, kNone, __OPS(o::i1), __OPS(i::f32, i::f32)),
    __INST(AddFOp, kNone, __OPS(o::f32), __OPS(i::f32, i::f32)),
    __INST(SubFOp, kNone, __OPS(o::f32), __OPS(i::f32, i::f32)),
    __INST(MulFOp, kNone, __OPS(o::f32), __OPS(i::f32, i::f32)),
    __INST(FmaFOp, kNone, __OPS(o::f32), __OPS(i::f32, i::f32, i::f32)),
    __INST(FmaIOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32, i::i32)),
    __INST(MulIOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32)),
    __INST(MulIExtendedOp, kNone, __OPS(o::i32, o::i32), __OPS(i::i32, i::i32)),
    __INST(AddIOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32)),
    __INST(AddCarryIOp, kNone, __OPS(o::i32, o::i1), __OPS(i::i32, i::i32, i::i1)),
    __INST(SubIOp, kNone, __OPS(o::i32), __OPS(i::i32, i::i32)),
    __INST(SubBurrowIOp, kNone, __OPS(o::i32, o::i1), __OPS(i::i32, i::i32, i::i1)),
    __INST(FPToSIOp, kNone, __OPS(o::i32), __OPS(i::f32)),
    __INST(SIToFPOp, kNone, __OPS(o::f32), __OPS(i::i32)),
    __INST(FPToUIOp, kNone, __OPS(o::i32), __OPS(i::f32)),
    __INST(UIToFPOp, kNone, __OPS(o::f32), __OPS(i::i32)),
    __INST(SI4ToFloat, kNone, __OPS(o::f32), __OPS(i::i32)),
    __INST(TruncFOp, kNone, __OPS(o::f32), __OPS(i::f64)),
    __INST(ExtFOp, kNone, __OPS(o::f64), __OPS(i::f32)),
    __INST(TruncOp, kNone, __OPS(o::f32), __OPS(i::f32)),
    __INST(CeilOp, kNone, __OPS(o::f32), __OPS(i::f32)),
    __INST(RoundEvenOp, kNone, __OPS(o::f32), __OPS(i::f32)),
    __INST(FractOp, kNone, __OPS(o::f32), __OPS(i::f32)),
    __INST(FloorOp, kNone, __OPS(o::f32), __OPS(i::f32)),
    __INST(Exp2Op, kNone, __OPS(o::f32), __OPS(i::f32)),
    __INST(Log2Op, kNone, __OPS(o::f32), __OPS(i::f32)),
    __INST(RcpOp, kNone, __OPS(o::f32), __OPS(i::f32)),
    __INST(RsqrtOp, kNone, __OPS(o::f32), __OPS(i::f32)),
    __INST(SqrtOp, kNone, __OPS(o::f32), __OPS(i::f32)),
    __INST(SinOp, kNone, __OPS(o::f32), __OPS(i::f32)),
    __INST(CosOp, kNone, __OPS(o::f32), __OPS(i::f32)),
    __INST(LdexpOp, kNone, __OPS(o::f32), __OPS(i::f32, i::i32)),
    __INST(ClampFMinMaxOp, kNone, __OPS(o::f32), __OPS(i::f32)),
    __INST(ClampFZeroOp, kNone, __OPS(o::f32), __OPS(i::f32)),
    __INST(FrexpOp, kNone, __OPS(o::f32, o::i32), __OPS(i::f32)),
    __INST(PackHalf2x16Op, kNone, __OPS(o::i32), __OPS(i::f32, i::f32)),
    __INST(UnpackHalf2x16, kNone, __OPS(o::f32, o::f32), __OPS(i::i32)),
    __INST(PackSnorm2x16Op, kNone, __OPS(o::i32), __OPS(i::f32, i::f32)),
    __INST(PackUnorm2x16Op, kNone, __OPS(o::i32), __OPS(i::f32, i::f32)),
    __INST(NegateFOp, kNone, __OPS(o::f32), __OPS(i::f32)),
    __INST(AbsoluteOp, kNone, __OPS(o::f32), __OPS(i::f32)),
    __INST(NotOp, kNone, __OPS(o::i32), __OPS(i::i32)),
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

} // namespace compiler::ir::dialect::arith