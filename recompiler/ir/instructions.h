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
  InstCore                     core;
  std::array<OutputOperand, 5> dstOperands; // adjust if needed
  std::array<InputOperand, 5>  srcOperands; // adjust if needed
  std::string_view             name;
};

constexpr auto makeInstDef(eInstKind kind, util::Flags<ir::eInstructionFlags> flags, auto&& dstOps, auto&& srcOps, std::string_view name) {
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
            .kind   = conv(kind),
            .flags  = flags,
            .numDst = (uint8_t)dstOps.size(),
            .numSrc = (uint8_t)srcOps.size(),
        },
    .dstOperands = toArray.template operator()<decltype(InstDef::dstOperands)>(dstOps),
    .srcOperands = toArray.template operator()<decltype(InstDef::srcOperands)>(srcOps),
    .name        = name
  };
}

namespace compiler::ir::o {
constexpr OutputOperand i1 {.type = OperandType::i1()};
constexpr OutputOperand i32 {.type = OperandType::i32()};
constexpr OutputOperand i64 {.type = OperandType::i64()};
constexpr OutputOperand f32 {.type = OperandType::f32()};
constexpr OutputOperand f64 {.type = OperandType::f64()};
// ...
} // namespace compiler::ir::o

namespace compiler::ir::i {
constexpr InputOperand i1 {.type = OperandType::i1()};
constexpr InputOperand i32 {.type = OperandType::i32()};
constexpr InputOperand i64 {.type = OperandType::i64()};
constexpr InputOperand f32 {.type = OperandType::f32()};
constexpr InputOperand f64 {.type = OperandType::f64()};
// ...
} // namespace compiler::ir::i

#define __OPS(...)                          __VA_ARGS__
#define __INST(kind, flags, dstOps, srcOps) makeInstDef(eInstKind::kind, {eInstructionFlags::flags}, std::array {dstOps}, std::array {srcOps}, #kind)

#define __INST_NO_OPS(kind, flags)                                                                                                                             \
  makeInstDef(eInstKind::kind, {eInstructionFlags::flags}, std::array<OutputOperand, 0> {}, std::array<InputOperand, 0> {}, #kind)

#define __INST_NO_DST(kind, flags, srcOps) makeInstDef(eInstKind::kind, {eInstructionFlags::flags}, std::array<OutputOperand, 0> {}, std::array {srcOps}, #kind)

#define __INST_NO_SRC(kind, flags, dstOps) makeInstDef(eInstKind::kind, {eInstructionFlags::flags}, std::array {dstOps}, std::array<InputOperand, 0> {}, #kind)

using namespace compiler::ir;
static constexpr std::array kInstTable = {
    __INST(MoveOp, kNone, __OPS(o::i32), __OPS(i::i32)),
    __INST(SelectOp, kNone, __OPS(o::i32), __OPS(i::i1, i::i32, i::i32)),
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
    __INST_NO_OPS(ReturnOp, kTerminator),
    __INST_NO_DST(DiscardOp, kTerminator, __OPS(i::i1)),
    __INST_NO_OPS(BarrierOp, kBarrier),
    __INST_NO_DST(JumpAbsOp, kTerminator, __OPS(i::i64)),
    __INST_NO_DST(CondJumpAbsOp, kTerminator, __OPS(i::i1, i::i64)),
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

} // namespace compiler::ir