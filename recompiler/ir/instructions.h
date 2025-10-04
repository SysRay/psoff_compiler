#pragma once

#include "ir.h"

#include <string_view>
#include <tuple>

namespace compiler::ir {
enum class eInstKind {
  MoveOp                = 0,
  SelectOp              = 1,
  BitReverseOp          = 2,
  BitCountOp            = 3,
  BitAndOp              = 4,
  BitOrOp               = 5,
  BitXorOp              = 6,
  BitFieldMaskOp        = 7,
  FindILsbOp            = 8,
  FindUMsbOp            = 9,
  FindSMsbOp            = 10,
  SignExtendOp          = 11,
  BitsetOp              = 12,
  BitCmpOp              = 13,
  BitFieldInsertOp      = 14,
  BitUIExtractOp        = 15,
  BitSIExtractOp        = 16,
  BitUIExtractCompactOp = 17,
  BitSIExtractCompactOp = 18,
  ShiftLUIOp            = 19,
  ShiftRUIOp            = 20,
  ShiftRSIOp            = 21,
  Max3UIOp              = 22,
  MaxUIOp               = 23,
  Max3SIOp              = 24,
  MaxSIOp               = 25,
  Max3FOp               = 26,
  MaxFOp                = 27,
  MaxNOp                = 28,
  MedUIOp               = 29,
  MedSIOp               = 30,
  MedFOp                = 31,
  MinUIOp               = 32,
  Min3UIOp              = 33,
  MinSIOp               = 34,
  Min3SIOp              = 35,
  MinFOp                = 36,
  Min3FOp               = 37,
  MinNOp                = 38,
  CmpIOp                = 39,
  CmpFOp                = 40,
  AddFOp                = 41,
  SubFOp                = 42,
  MulFOp                = 43,
  FmaFOp                = 44,
  FmaIOp                = 45,
  MulIOp                = 46,
  MulIExtendedOp        = 47,
  AddIOp                = 48,
  AddCarryIOp           = 49,
  SubIOp                = 50,
  SubBurrowIOp          = 51,
  FPToSIOp              = 52,
  SIToFPOp              = 53,
  FPToUIOp              = 54,
  UIToFPOp              = 55,
  SI4ToFloat            = 56,
  TruncFOp              = 57,
  ExtFOp                = 58,
  TruncOp               = 59,
  CeilOp                = 60,
  RoundEvenOp           = 61,
  FractOp               = 62,
  FloorOp               = 63,
  Exp2Op                = 64,
  Log2Op                = 65,
  RcpOp                 = 66,
  RsqrtOp               = 67,
  SqrtOp                = 68,
  SinOp                 = 69,
  CosOp                 = 70,
  LdexpOp               = 71,
  ClampFMinMaxOp        = 72,
  ClampFZeroOp          = 73,
  FrexpOp               = 74,
  PackHalf2x16Op        = 75,
  UnpackHalf2x16        = 76,
  PackSnorm2x16Op       = 77,
  PackUnorm2x16Op       = 78,
  ReturnOp              = 79,
  DiscardOp             = 80,
  BarrierOp             = 81,
  JumpAbsOp             = 82,
  CondJumpAbsOp         = 83,
  ConstantOp            = 84,
};

constexpr inline InstructionKind_t conv(eInstKind code) {
  return (InstructionKind_t)code;
}

constexpr inline eInstKind conv(InstructionKind_t code) {
  return (eInstKind)code;
}

namespace internal {
struct InstDef {
  InstCore         core;
  std::string_view name;
};

constexpr auto makeInstDef(eInstKind kind, eInstructionGroup group, util::Flags<ir::eInstructionFlags> flags, auto&& dstOps, auto&& srcOps,
                           std::string_view name) {
  // Helper to convert container to std::array
  auto toArray = []<typename Container>(const auto& container) -> Container {
    Container result {};
    size_t    i = 0;
    for (const auto& item: container) {
      result[i++] = item;
    }
    return result;
  };

  return InstDef {
    .core =
        ir::InstCore {
            .kind        = conv(kind),
            .group       = group,
            .flags       = flags,
            .numDst      = (uint8_t)dstOps.size(),
            .numSrc      = (uint8_t)srcOps.size(),
            .dstOperands = toArray.template operator()<decltype(ir::InstCore::dstOperands)>(dstOps),
            .srcOperands = toArray.template operator()<decltype(ir::InstCore::srcOperands)>(srcOps),
        },
    .name = name
  };
}

namespace compiler::ir::ops {
constexpr Operand i1 {.type = OperandType::i1()};
constexpr Operand i32 {.type = OperandType::i32()};
constexpr Operand i64 {.type = OperandType::i32()};
constexpr Operand f32 {.type = OperandType::f32()};
constexpr Operand f64 {.type = OperandType::f64()};
// ...
} // namespace compiler::ir::ops

#define __OPS(...) __VA_ARGS__
#define __INST(kind, group, flags, dstOps, srcOps)                                                                                                             \
  makeInstDef(eInstKind::kind, eInstructionGroup::group, {eInstructionFlags::flags}, std::array {dstOps}, std::array {srcOps}, #kind)

#define __INST_NO_OPS(kind, group, flags)                                                                                                                      \
  makeInstDef(eInstKind::kind, eInstructionGroup::group, {eInstructionFlags::flags}, std::array<Operand, 0> {}, std::array<Operand, 0> {}, #kind)

#define __INST_NO_DST(kind, group, flags, srcOps)                                                                                                              \
  makeInstDef(eInstKind::kind, eInstructionGroup::group, {eInstructionFlags::flags}, std::array<Operand, 0> {}, std::array {srcOps}, #kind)

#define __INST_NO_SRC(kind, group, flags, dstOps)                                                                                                              \
  makeInstDef(eInstKind::kind, eInstructionGroup::group, {eInstructionFlags::flags}, std::array {dstOps}, std::array<Operand, 0> {}, #kind)

using namespace compiler::ir::ops;
static constexpr std::array kInstTable = {
    __INST(MoveOp, kUnknown, kNone, __OPS(i32), __OPS(i32)),
    __INST(SelectOp, kUnknown, kNone, __OPS(i32), __OPS(i1, i32, i32)),
    __INST(BitReverseOp, kBIT, kNone, __OPS(i32), __OPS(i32)),
    __INST(BitCountOp, kBIT, kNone, __OPS(i32), __OPS(i32)),
    __INST(BitAndOp, kBIT, kNone, __OPS(i32), __OPS(i32, i32)),
    __INST(BitOrOp, kBIT, kNone, __OPS(i32), __OPS(i32, i32)),
    __INST(BitXorOp, kBIT, kNone, __OPS(i32), __OPS(i32, i32)),
    __INST(BitFieldMaskOp, kBIT, kNone, __OPS(i32), __OPS(i32, i32)),
    __INST(FindILsbOp, kBIT, kNone, __OPS(i32), __OPS(i32)),
    __INST(FindUMsbOp, kBIT, kNone, __OPS(i32), __OPS(i32)),
    __INST(FindSMsbOp, kBIT, kNone, __OPS(i32), __OPS(i32)),
    __INST(SignExtendOp, kBIT, kNone, __OPS(i32), __OPS(i32)),
    __INST(BitsetOp, kBIT, kNone, __OPS(i32), __OPS(i32, i32, i1)),
    __INST(BitCmpOp, kBIT, kNone, __OPS(i1), __OPS(i32, i32)),
    __INST(BitFieldInsertOp, kBIT, kNone, __OPS(i32), __OPS(i32, i32, i32)),
    __INST(BitUIExtractOp, kBIT, kNone, __OPS(i32), __OPS(i32, i32, i32)),
    __INST(BitSIExtractOp, kBIT, kNone, __OPS(i32), __OPS(i32, i32, i32)),
    __INST(BitUIExtractCompactOp, kBIT, kNone, __OPS(i32), __OPS(i32, i32)),
    __INST(BitSIExtractCompactOp, kBIT, kNone, __OPS(i32), __OPS(i32, i32)),
    __INST(ShiftLUIOp, kBIT, kNone, __OPS(i32), __OPS(i32, i32)),
    __INST(ShiftRUIOp, kBIT, kNone, __OPS(i32), __OPS(i32, i32)),
    __INST(ShiftRSIOp, kBIT, kNone, __OPS(i32), __OPS(i32, i32)),
    __INST(Max3UIOp, kALU, kNone, __OPS(i32), __OPS(i32, i32, i32)),
    __INST(MaxUIOp, kALU, kNone, __OPS(i32), __OPS(i32, i32)),
    __INST(Max3SIOp, kALU, kNone, __OPS(i32), __OPS(i32, i32, i32)),
    __INST(MaxSIOp, kALU, kNone, __OPS(i32), __OPS(i32, i32)),
    __INST(Max3FOp, kALU, kNone, __OPS(f32), __OPS(f32, f32, f32)),
    __INST(MaxFOp, kALU, kNone, __OPS(f32), __OPS(f32, f32)),
    __INST(MaxNOp, kALU, kNone, __OPS(f32), __OPS(f32, f32)),
    __INST(MedUIOp, kALU, kNone, __OPS(i32), __OPS(i32, i32)),
    __INST(MedSIOp, kALU, kNone, __OPS(i32), __OPS(i32, i32)),
    __INST(MedFOp, kALU, kNone, __OPS(i32), __OPS(i32, i32)),
    __INST(MinUIOp, kALU, kNone, __OPS(i32), __OPS(i32, i32)),
    __INST(Min3UIOp, kALU, kNone, __OPS(i32), __OPS(i32, i32, i32)),
    __INST(MinSIOp, kALU, kNone, __OPS(i32), __OPS(i32, i32)),
    __INST(Min3SIOp, kALU, kNone, __OPS(i32), __OPS(i32, i32, i32)),
    __INST(MinFOp, kALU, kNone, __OPS(f32), __OPS(f32, f32)),
    __INST(Min3FOp, kALU, kNone, __OPS(f32), __OPS(f32, f32, f32)),
    __INST(MinNOp, kALU, kNone, __OPS(f32), __OPS(f32, f32)),
    __INST(CmpIOp, kALU, kNone, __OPS(i1), __OPS(i32, i32)),
    __INST(CmpFOp, kALU, kNone, __OPS(i1), __OPS(f32, f32)),
    __INST(AddFOp, kALU, kNone, __OPS(f32), __OPS(f32, f32)),
    __INST(SubFOp, kALU, kNone, __OPS(f32), __OPS(f32, f32)),
    __INST(MulFOp, kALU, kNone, __OPS(f32), __OPS(f32, f32)),
    __INST(FmaFOp, kALU, kNone, __OPS(f32), __OPS(f32, f32, f32)),
    __INST(FmaIOp, kALU, kNone, __OPS(i32), __OPS(i32, i32, i32)),
    __INST(MulIOp, kALU, kNone, __OPS(i32), __OPS(i32, i32)),
    __INST(MulIExtendedOp, kALU, kNone, __OPS(i32, i32), __OPS(i32, i32)),
    __INST(AddIOp, kALU, kNone, __OPS(i32), __OPS(i32, i32)),
    __INST(AddCarryIOp, kALU, kNone, __OPS(i32, i1), __OPS(i32, i32, i1)),
    __INST(SubIOp, kALU, kNone, __OPS(i32), __OPS(i32, i32)),
    __INST(SubBurrowIOp, kALU, kNone, __OPS(i32, i1), __OPS(i32, i32, i1)),
    __INST(FPToSIOp, kALU, kNone, __OPS(i32), __OPS(f32)),
    __INST(SIToFPOp, kALU, kNone, __OPS(f32), __OPS(i32)),
    __INST(FPToUIOp, kALU, kNone, __OPS(i32), __OPS(f32)),
    __INST(UIToFPOp, kALU, kNone, __OPS(f32), __OPS(i32)),
    __INST(SI4ToFloat, kALU, kNone, __OPS(f32), __OPS(i32)),
    __INST(TruncFOp, kALU, kNone, __OPS(f32), __OPS(f64)),
    __INST(ExtFOp, kALU, kNone, __OPS(f64), __OPS(f32)),
    __INST(TruncOp, kALU, kNone, __OPS(f32), __OPS(f32)),
    __INST(CeilOp, kALU, kNone, __OPS(f32), __OPS(f32)),
    __INST(RoundEvenOp, kALU, kNone, __OPS(f32), __OPS(f32)),
    __INST(FractOp, kALU, kNone, __OPS(f32), __OPS(f32)),
    __INST(FloorOp, kALU, kNone, __OPS(f32), __OPS(f32)),
    __INST(Exp2Op, kALU, kNone, __OPS(f32), __OPS(f32)),
    __INST(Log2Op, kALU, kNone, __OPS(f32), __OPS(f32)),
    __INST(RcpOp, kALU, kNone, __OPS(f32), __OPS(f32)),
    __INST(RsqrtOp, kALU, kNone, __OPS(f32), __OPS(f32)),
    __INST(SqrtOp, kALU, kNone, __OPS(f32), __OPS(f32)),
    __INST(SinOp, kALU, kNone, __OPS(f32), __OPS(f32)),
    __INST(CosOp, kALU, kNone, __OPS(f32), __OPS(f32)),
    __INST(LdexpOp, kALU, kNone, __OPS(f32), __OPS(f32, i32)),
    __INST(ClampFMinMaxOp, kALU, kNone, __OPS(f32), __OPS(f32)),
    __INST(ClampFZeroOp, kALU, kNone, __OPS(f32), __OPS(f32)),
    __INST(FrexpOp, kALU, kNone, __OPS(f32, i32), __OPS(f32)),
    __INST(PackHalf2x16Op, kALU, kNone, __OPS(i32), __OPS(f32, f32)),
    __INST(UnpackHalf2x16, kALU, kNone, __OPS(f32, f32), __OPS(i32)),
    __INST(PackSnorm2x16Op, kALU, kNone, __OPS(i32), __OPS(f32, f32)),
    __INST(PackUnorm2x16Op, kALU, kNone, __OPS(i32), __OPS(f32, f32)),
    __INST_NO_OPS(ReturnOp, kFlowControl, kHasSideEffects),
    __INST_NO_DST(DiscardOp, kFlowControl, kHasSideEffects, __OPS(i1)),
    __INST_NO_OPS(BarrierOp, kBarrier, kHasSideEffects),
    __INST_NO_DST(JumpAbsOp, kFlowControl, kHasSideEffects, __OPS(i64)),
    __INST_NO_DST(CondJumpAbsOp, kFlowControl, kHasSideEffects, __OPS(i1, i64)),
    __INST_NO_SRC(ConstantOp, kConstant, kConstant, __OPS(f32)),
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

} // namespace compiler::ir