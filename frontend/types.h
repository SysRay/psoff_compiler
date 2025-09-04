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
  Const,
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
  ConstInt     = 129,
  ConstUInt    = 193,
  ConstFloat   = 240,
  VccZ         = 251,
  ExecZ        = 252,
  Scc          = 253,
  LdsDirect    = 254,
  LiteralConst = 255,
  VGPR,
};

eOperandKind getOperandKind(OperandKind_t kind) {
  return (eOperandKind)kind;
}

struct OperandFlagsSrc {
  constexpr OperandFlagsSrc(OperandFlags_t flags): raw(flags) {}

  constexpr auto getRegClass() const { return (eRegClass)bits.regClass; }

  constexpr auto getNegate() const { return bits.negate; }

  constexpr auto getAbsolute() const { return bits.abs; }

  private:
  union {
    struct {
      OperandFlags_t regClass : 3;
      bool           negate   : 1;
      bool           abs      : 1;
    } bits;

    OperandFlags_t raw;
  };
};

struct OperandFlagsDst {
  constexpr OperandFlagsDst(OperandFlags_t flags): raw(flags) {}

  constexpr auto getRegClass() const { return (eRegClass)bits.regClass; }

  constexpr auto getClamp() const { return bits.clamp; }

  constexpr auto getMultiply() const {
    constexpr float omodTable[] = {1.0f, 2.0f, 4.0f, 0.5f};
    return omodTable[bits.omod];
  }

  constexpr auto hasMultiply() const {
    return bits.omod != 0;
  }

  private:
  union {
    struct {
      OperandFlags_t regClass : 3;
      OperandFlags_t omod     : 2;
      bool           clamp    : 1;
    } bits;

    OperandFlags_t raw;
  };
};
} // namespace compiler::frontend