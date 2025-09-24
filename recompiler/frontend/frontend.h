#pragma once

#include "ir/config.h"

#include <array>
#include <assert.h>
#include <string>
#include <utility>

namespace compiler::frontend {
struct eOperandKind {
  enum class eBase : OperandKind_t {
    SGPR  = 0,
    VccLo = 106,
    VccHi,
    CustomTemp0Lo,
    CustomTemp0Hi,
    CustomTemp1Lo,
    CustomTemp1Hi,
    M0          = 124,
    CustomVskip = 125,
    ExecLo      = 126,
    ExecHi,
    ConstZero,
    ConstUInt       = 129, //< 1..64
    ConstSInt       = 193, ///< -1..-64
    ConstFloat_0_5  = 240,
    ConstFloat_n0_5 = 241,
    ConstFloat_1_0  = 242,
    ConstFloat_n1_0 = 243,
    ConstFloat_2_0  = 244,
    ConstFloat_n2_0 = 245,
    ConstFloat_4_0  = 246,
    ConstFloat_n4_0 = 247,
    INV_2PI         = 248,
    SDWA            = 249,
    DPP             = 250,
    VccZ            = 251,
    ExecZ           = 252,
    Scc             = 253,
    LdsDirect       = 254,
    Literal         = 255,
    VGPR,
  };
  enum class eKind : uint8_t { Register, ConstantF, ConstantI };

  struct __OperandTypeData {
    union {
      struct _struct {
        eBase base : 10;
        eKind kind : 3;
        bool  b64  : 1;
      } bits;

      OperandKind_t raw;
    };
  };

  // ---- construction -------------------------------------------------------

  static constexpr eOperandKind create(OperandKind_t b) {
    return eOperandKind(__OperandTypeData {.bits = {.base = (eBase)b, .kind = getKind(b), .b64 = is64bit(b)}});
  }

  static constexpr eOperandKind createImm(uint8_t value) {
    assert(value <= 64);
    return create((OperandKind_t)((OperandKind_t)eBase::ConstZero + value));
  }

  // ---- queries ------------------------------------------------------------

  constexpr OperandKind_t raw() const noexcept { return _v.raw; }

  constexpr eKind kind() const noexcept { return _v.bits.kind; }

  constexpr eBase base() const noexcept { return _v.bits.base; }

  constexpr bool is64bit() const noexcept { return _v.bits.b64; }

  constexpr bool isLiteral() const noexcept { return _v.bits.base == eBase::Literal; }

  constexpr bool isConstF() const noexcept { return _v.bits.kind == eKind::ConstantF; }

  constexpr bool isConstI() const noexcept { return _v.bits.kind == eKind::ConstantI; }

  // ---- pre-made  ---------------------------------------------------
  static constexpr eOperandKind Literal() { return create((OperandKind_t)eBase::Literal); }

  static constexpr eOperandKind SGPR(uint16_t index) { return create((OperandKind_t)eBase::SGPR + index); }

  static constexpr eOperandKind VGPR(uint16_t index) { return create((OperandKind_t)eBase::VGPR + index); }

  static constexpr eOperandKind EXEC() { return create((OperandKind_t)eBase::ExecLo); }

  static constexpr eOperandKind VCC() { return create((OperandKind_t)eBase::VccLo); }

  static constexpr eOperandKind SCC() { return create((OperandKind_t)eBase::Scc); }

  static constexpr eOperandKind VSKIP() { return create((OperandKind_t)eBase::CustomVskip); }

  static constexpr eOperandKind Temp0() { return create((OperandKind_t)eBase::CustomTemp0Lo); }

  static constexpr eOperandKind Temp1() { return create((OperandKind_t)eBase::CustomTemp1Lo); }

  private:
  static constexpr eKind getKind(OperandKind_t b) {
    if (b >= (OperandKind_t)eBase::ConstZero && b < (OperandKind_t)eBase::ConstFloat_0_5) return eKind::ConstantI;
    if (b >= (OperandKind_t)eBase::ConstFloat_0_5 && b <= (OperandKind_t)eBase::ConstFloat_n4_0) return eKind::ConstantF;
    return eKind::Register;
  }

  static constexpr bool is64bit(OperandKind_t b) {
    if (b == (OperandKind_t)eBase::ExecLo || b == (OperandKind_t)eBase::VccLo || b == (OperandKind_t)eBase::CustomTemp0Lo ||
        b == (OperandKind_t)eBase::CustomTemp1Lo)
      return true;
    return false;
  }

  __OperandTypeData _v;

  explicit constexpr eOperandKind(__OperandTypeData&& type): _v(type) {}
};

constexpr inline OperandKind_t getOperandKind(eOperandKind kind) {
  return kind.raw();
}

struct OperandFlagsSrc {
  constexpr OperandFlagsSrc(OperandFlags_t flags): raw(flags) {}

  constexpr OperandFlagsSrc(bool negate, bool abs): bits {.negate = negate, .abs = abs} {}

  constexpr auto getNegate() const { return bits.negate; }

  constexpr auto getAbsolute() const { return bits.abs; }

  operator OperandFlags_t() const { return raw; }

  private:
  union {
    struct {
      bool negate : 1;
      bool abs    : 1;
    } bits;

    OperandFlags_t raw;
  };
};

struct OperandFlagsDst {
  constexpr OperandFlagsDst(OperandFlags_t flags): raw(flags) {}

  constexpr OperandFlagsDst(uint8_t omod, bool clamp, bool negate): bits {.omod = omod, .clamp = clamp, .negate = negate} {}

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
      uint8_t omod   : 2;
      bool    clamp  : 1;
      bool    negate : 1;
    } bits;

    OperandFlags_t raw;
  };
};
} // namespace compiler::frontend