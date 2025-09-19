#pragma once

#include "config.h"

#include <limits>
#include <utility>

namespace compiler::ir {

enum class eInstructionFlags : InstructionFlags_t {
  kNone           = 0,
  kHasSideEffects = 1u << 0, ///< cannot be removed / reordered
  kWritesEXEC     = 1u << 1, ///< Writes to exec
  kVirtual        = 1u << 2, ///< Instructions that run only with Exec set
  kBarrier        = 1u << 3, ///< e.g. exec barrier, waitcnt
  kConstant       = 1u << 4, ///< use src constant
};

enum class eInstructionGroup : InstructionGroup_t {
  kUnknown,
  kALU,
  kBIT,
  kConstant,
  kFlowControl,
  kBarrier,
};

struct OperandType {
  enum class eBase : uint8_t {
    // integers
    i1,
    i8,
    i16,
    i32,
    i64,
    // floats
    f16,
    f32,
    f64
  };

  enum class eKind : uint8_t { Scalar = 0, Vector = 1, Array = 2 };

  struct __OperandTypeData {
    union {
      struct _struct {
        OperandType_t base  : 4;
        OperandType_t kind  : 3;
        OperandType_t lanes : 9;
      } bits;

      OperandType_t raw;
    };
  };

  // ---- construction -------------------------------------------------------

  static constexpr OperandType scalar(eBase b) {
    return OperandType(__OperandTypeData {.bits = {.base = (uint8_t)b, .kind = (uint8_t)eKind::Scalar, .lanes = 0}});
  }

  static constexpr OperandType vector(eBase b, std::uint8_t lanes) {
    return OperandType(__OperandTypeData {.bits = {.base = (uint8_t)b, .kind = (uint8_t)eKind::Vector, .lanes = lanes}});
  }

  static constexpr OperandType array(eBase b, std::uint8_t lanes) {
    return OperandType(__OperandTypeData {.bits = {.base = (uint8_t)b, .kind = (uint8_t)eKind::Array, .lanes = lanes}});
  }

  // ---- queries ------------------------------------------------------------

  constexpr eKind kind() const noexcept { return static_cast<eKind>(_v.bits.kind); }

  constexpr eBase base() const noexcept { return static_cast<eBase>(_v.bits.base); }

  constexpr bool is_scalar() const noexcept { return kind() == eKind::Scalar; }

  constexpr bool is_vector() const noexcept { return kind() == eKind::Vector; }

  constexpr bool is_array() const noexcept { return kind() == eKind::Array; }

  constexpr bool is_signed() const noexcept { return 0; }

  constexpr bool is_float() const noexcept { return is_float_base((eBase)_v.bits.base); }

  // eBase element size in bytes
  constexpr std::uint8_t base_size_bytes() const noexcept { return base_size((eBase)_v.bits.base); }

  // Vector columns; scalar -> 0; vector -> lanes>=2; matrix -> columns
  constexpr std::uint16_t cols() const noexcept { return _v.bits.lanes; }

  // Total byte size if known at compile time (matrices = rows*cols*base)
  // Returns 0 for opaque, and for pointers returns pointer-size-agnostic 0 in IR.
  constexpr std::uint32_t byte_size() const noexcept {
    const std::uint32_t elem = base_size_bytes();
    const std::uint32_t c    = cols();
    if (c) return elem * c;
    return elem;
  }

  constexpr OperandType_t packed() const noexcept { return _v.raw; }

  // ---- pre-made scalars ---------------------------------------------------
  static constexpr OperandType i1() { return scalar(eBase::i1); }

  static constexpr OperandType i8() { return scalar(eBase::i8); }

  static constexpr OperandType i16() { return scalar(eBase::i16); }

  static constexpr OperandType i32() { return scalar(eBase::i32); }

  static constexpr OperandType i64() { return scalar(eBase::i64); }

  static constexpr OperandType f16() { return scalar(eBase::f16); }

  static constexpr OperandType f32() { return scalar(eBase::f32); }

  static constexpr OperandType f64() { return scalar(eBase::f64); }

  private:
  __OperandTypeData _v;

  static constexpr std::uint8_t base_size(eBase b) {
    switch (b) {
      case eBase::i1: return 1;
      case eBase::i8: return 1;
      case eBase::i16: return 2;
      case eBase::i32:
      case eBase::f32: return 4;
      case eBase::i64:
      case eBase::f64: return 8;
      case eBase::f16: return 2;
    }
    return 0;
  }

  static constexpr bool is_float_base(eBase b) {
    switch (b) {
      case eBase::f16:
      case eBase::f32:
      case eBase::f64: return true;
      default: return false;
    }
  }

  explicit constexpr OperandType(__OperandTypeData&& type): _v(type) {}
};

// Hash support
struct OperandTypeHash {
  auto operator()(const OperandType& t) const noexcept { return std::hash<OperandType_t> {}(t.packed()); }
};

} // namespace compiler::ir
