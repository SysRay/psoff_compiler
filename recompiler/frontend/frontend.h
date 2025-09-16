#pragma once

#include "ir/config.h"

#include <array>
#include <string>
#include <utility>

namespace compiler::frontend {
enum class eRegClass : OperandFlags_t {
  SGPR,
  VGPR,
  Compare,
  Exec,
  Constant,
  Lds,
};

enum class eOperandKind : OperandKind_t {
  SGPR  = 0,
  VccLo = 106,
  VccHi,
  M0     = 124,
  ExecLo = 126,
  ExecHi,
  ConstZero,
  ConstUInt  = 129, //< 1..64
  ConstSInt  = 193, ///< -1..-64
  ConstFloat = 240, ///< 0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 4.0f, -4.0f
  VccZ       = 251,
  ExecZ      = 252,
  Scc        = 253,
  LdsDirect  = 254,
  Literal    = 255,
  VGPR,
};

constexpr inline eOperandKind getOperandKind(OperandKind_t kind) {
  return (eOperandKind)kind;
}

constexpr inline OperandKind_t getOperandKind(eOperandKind kind) {
  return (OperandKind_t)kind;
}

struct OperandFlagsSrc {
  constexpr OperandFlagsSrc(OperandFlags_t flags): raw(flags) {}

  constexpr OperandFlagsSrc(eRegClass regClass, bool negate, bool abs): bits {.regClass = regClass, .negate = negate, .abs = abs} {}

  constexpr auto getRegClass() const { return (eRegClass)bits.regClass; }

  constexpr auto getNegate() const { return bits.negate; }

  constexpr auto getAbsolute() const { return bits.abs; }

  operator OperandFlags_t() const { return raw; }

  private:
  union {
    struct {
      eRegClass regClass : 3;
      bool      negate   : 1;
      bool      abs      : 1;
    } bits;

    OperandFlags_t raw;
  };
};

struct OperandFlagsDst {
  constexpr OperandFlagsDst(OperandFlags_t flags): raw(flags) {}

  constexpr OperandFlagsDst(eRegClass regClass, uint8_t omod, bool clamp, bool negate)
      : bits {.regClass = regClass, .omod = omod, .clamp = clamp, .negate = negate} {}

  constexpr auto getRegClass() const { return (eRegClass)bits.regClass; }

  constexpr auto getClamp() const { return bits.clamp; }

  constexpr auto getMultiply() const {
    constexpr float omodTable[] = {1.0f, 2.0f, 4.0f, 0.5f};
    return omodTable[bits.omod];
  }

  constexpr auto hasMultiply() const { return bits.omod != 0; }

  constexpr auto getNegate() const { return bits.negate; }

  operator OperandFlags_t() const { return raw; }

  private:
  union {
    struct {
      eRegClass regClass : 3;
      uint8_t   omod     : 2;
      bool      clamp    : 1;
      bool      negate   : 1;
    } bits;

    OperandFlags_t raw;
  };
};
} // namespace compiler::frontend