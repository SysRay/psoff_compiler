#pragma once
#include <stdint.h>

namespace util {

template <class T, size_t Offset, size_t Width>
struct Bitfield {
  static constexpr size_t offset = Offset;
  static constexpr size_t width  = Width;
  static constexpr T      mask   = ((1ull << Width) - 1) << Offset;

  static constexpr auto extract(T value) noexcept { return (value & mask) >> Offset; }

  static constexpr auto insert(T value, auto field) noexcept { return (value & ~mask) | ((field << Offset) & mask); }
};
} // namespace util

// ---------------- Macro Helper ----------------
#define DEFINE_BITFIELD_STRUCT(StructName, StorageType, FIELDS_MACRO)                                                                                          \
  struct StructName {                                                                                                                                          \
    using storage_t = StorageType;                                                                                                                             \
    enum class Field { FIELDS_MACRO(ENUM_ENTRY) };                                                                                                             \
    template <Field F>                                                                                                                                         \
    struct desc;                                                                                                                                               \
    FIELDS_MACRO(DESC_ENTRY)                                                                                                                                   \
    storage_t bits {};                                                                                                                                         \
    constexpr StructName() = default;                                                                                                                          \
    constexpr explicit StructName(storage_t raw): bits(raw) {}                                                                                                 \
    template <Field F>                                                                                                                                         \
    constexpr storage_t get() const noexcept {                                                                                                                 \
      using bf = typename desc<F>::type;                                                                                                                       \
      return bf::extract(bits);                                                                                                                                \
    }                                                                                                                                                          \
    template <Field F>                                                                                                                                         \
    constexpr void set(storage_t value) noexcept {                                                                                                             \
      using bf = typename desc<F>::type;                                                                                                                       \
      bits     = bf::insert(bits, value);                                                                                                                      \
    }                                                                                                                                                          \
  };

// Helper macros for ENUM entries and DESC specializations
#define ENUM_ENTRY(name, offset, width) name,
#define DESC_ENTRY(name, offset, width)                                                                                                                        \
  template <>                                                                                                                                                  \
  struct desc<Field::name> {                                                                                                                                   \
    using type = util::Bitfield<storage_t, offset, width>;                                                                                                     \
  };

// // ---------------- Example Fields ----------------
// #define SOP1_FIELDS(X) \
//   X(SRC0, 0, 9) \ X(DST, 9, 7)

// // ---------------- Define the Struct ----------------
// DEFINE_BITFIELD_STRUCT(SOP1, uint32_t, SOP1_FIELDS)

// // ---------------- Use the Struct ----------------
// SOP1 sop{};
// sop.set<SOP1::Field::SRC0>(0x123);
// sop.set<SOP1::Field::DST>(0x3F);
