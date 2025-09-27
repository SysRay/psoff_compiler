#pragma once

#include <initializer_list>
#include <type_traits>

namespace compiler::util {
template <typename T>
class Flags {
  static_assert(std::is_enum_v<T>, "Template parameter must be an enum class");
  using Underlying_t = std::underlying_type_t<T>;

  Underlying_t flags = 0;

  public:
  constexpr Flags() = default;

  constexpr explicit Flags(T flag) { set(flag); }

  constexpr Flags(std::initializer_list<T> flags) {
    for (T flag: flags) {
      set(flag);
    }
  }

  constexpr void set(T flag) noexcept { flags |= static_cast<Underlying_t>(flag); }

  constexpr void clear(T flag) noexcept { flags &= ~static_cast<Underlying_t>(flag); }

  [[nodiscard]] constexpr bool is_set(T flag) const noexcept { return (flags & static_cast<Underlying_t>(flag)) != 0; }

  constexpr void reset() noexcept { flags = 0; }

  [[nodiscard]] constexpr Flags operator|(T flag) const noexcept {
    Flags result = *this;
    result.set(flag);
    return result;
  }

  constexpr Flags& operator|=(T flag) noexcept {
    set(flag);
    return *this;
  }

  constexpr Flags& operator=(T flag) noexcept {
    flags = (Underlying_t)flag;
    return *this;
  }

  [[nodiscard]] constexpr Flags operator&(T flag) const noexcept {
    Flags result;
    if (is_set(flag)) result.set(flag);
    return result;
  }

  constexpr Flags& operator&=(T flag) noexcept {
    if (!is_set(flag)) set(flag);
    return *this;
  }

  [[nodiscard]] constexpr bool operator==(const Flags& other) const noexcept { return flags == other.flags; }

  [[nodiscard]] constexpr bool operator!=(const Flags& other) const noexcept { return !(*this == other); }

  [[nodiscard]] auto value() const { return flags; }
};
} // namespace compiler::util