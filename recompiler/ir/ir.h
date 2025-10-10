#pragma once

#include "include/flags.h"
#include "types.h"

#include <array>
#include <type_traits>

namespace compiler::ir {

struct Operand {
  OperandKind_t  kind  = 0;
  OperandFlags_t flags = 0;

  OperandType type = OperandType::i32();
};

struct InstConstant {
  OperandType type = OperandType::i32();
  uint64_t    value;
};

struct alignas(64) InstCore {
  InstructionKind_t                  kind  = -1;
  eInstructionGroup                  group = eInstructionGroup::kUnknown;
  util::Flags<ir::eInstructionFlags> flags;

  InstructionUserData_t userData = 0;

  struct {
    uint8_t numDst : 4;
    uint8_t numSrc : 4;
  };

  std::array<Operand, config::kMaxDstOps> dstOperands;

  union {
    std::array<Operand, config::kMaxSrcOps> srcOperands;
    InstConstant                            srcConstant;
  };

  inline bool isValid() const { return group != eInstructionGroup::kUnknown; }

  inline bool isConstant() const { return flags.is_set(ir::eInstructionFlags::kConstant); }
};

static_assert(sizeof(InstCore) <= 64); ///< cache lines
static_assert(config::kMaxOps <= 15);  ///< only 4 bits

// // Handle enum bits to underlying conversion
template <typename Enum>
struct Flags {
  static_assert(std::is_enum_v<Enum>, "Flags<T> requires an enum type");

  using underlying_t = std::underlying_type_t<Enum>;
  underlying_t value {};

  constexpr Flags(): value(0) {}

  constexpr Flags(Enum flag): value(static_cast<underlying_t>(flag)) {}

  constexpr Flags(underlying_t raw): value(raw) {}

  constexpr operator underlying_t() const { return value; }

  constexpr bool has(Enum flag) const { return (value & static_cast<underlying_t>(flag)) != 0; }
};

template <typename Enum>
constexpr Flags<Enum> operator|(Enum lhs, Enum rhs) {
  using U = std::underlying_type_t<Enum>;
  return Flags<Enum>(static_cast<U>(lhs) | static_cast<U>(rhs));
}

template <typename Enum>
constexpr Flags<Enum> operator|(Flags<Enum> lhs, Enum rhs) {
  return Flags<Enum>(lhs.value | static_cast<std::underlying_type_t<Enum>>(rhs));
}

template <typename Enum>
constexpr Flags<Enum> operator|(Enum lhs, Flags<Enum> rhs) {
  return Flags<Enum>(static_cast<std::underlying_type_t<Enum>>(lhs) | rhs.value);
}
} // namespace compiler::ir