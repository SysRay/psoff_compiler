#pragma once

#include "include/flags.h"
#include "types.h"
#include "utility/util.h"

#include <array>
#include <deque>
#include <memory_resource>
#include <type_traits>
#include <vector>

namespace compiler::ir {

struct InputOperand {
  OperandKind_t  kind  = -1;
  OperandFlags_t flags = 0;

  SsaId_t ssaId = {};

  OperandType type = OperandType::i32();

  bool isSSA() const { return ssaId.isValid(); }
};

struct OutputOperand {
  OperandKind_t  kind  = -1;
  OperandFlags_t flags = 0;

  struct {
    SsaId_t  ssaValue    = {};
    uint32_t resultIndex = 0;
  } ssa = {};

  OperandType type = OperandType::i32();

  inline bool hasKind() const { return kind >= 0; }
};

struct alignas(16) InstCore {
  InstructionKind_t kind    = -1;
  eDialect          dialect = eDialect::kCore;

  util::Flags<ir::eInstructionFlags> flags;

  InstructionUserData_t userData = 0;

  uint8_t numDst;
  uint8_t numSrc;

  OutputOperandId_t dstStartId; ///< index into operand table

  union {
    InputOperandId_t srcStartId; ///< index into operand table

    ConstantId_t constantId;
  };

  inline bool isValid() const { return kind != -1; }

  inline bool isConstant() const { return flags.is_set(eInstructionFlags::kConstant); }

  inline bool isTerminator() const { return flags.is_set(eInstructionFlags::kTerminator); }

  inline auto getSrcStart() const { return srcStartId; }

  inline auto getConstantId() const { return constantId; }

  inline auto getOutputId(uint8_t n) const { return OutputOperandId_t(dstStartId + n); }

  inline auto getInputId(uint8_t n) const { return InputOperandId_t(srcStartId + n); }
};

static_assert(sizeof(InstCore) <= 16); ///< cache lines

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

struct ConstantValue {
  union {
    int64_t  value_i64;
    uint64_t value_u64;
    double   value_f64;
  };
};

} // namespace compiler::ir