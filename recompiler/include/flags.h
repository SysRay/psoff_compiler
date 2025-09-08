#pragma once

#include <initializer_list>
#include <type_traits>

namespace util {
template <typename Enum>
class Flags {
  static_assert(std::is_enum_v<Enum>, "Template parameter must be an enum class");
  using Underlying = std::underlying_type_t<Enum>;

  Underlying flags = 0;

  public:
  constexpr Flags() = default;

  constexpr explicit Flags(Enum flag) { set(flag); }

  constexpr Flags(std::initializer_list<Enum> flags) {
    for (Enum flag: flags) {
      set(flag);
    }
  }

  constexpr void set(Enum flag) noexcept { flags |= static_cast<Underlying>(flag); }

  constexpr void clear(Enum flag) noexcept { flags &= ~static_cast<Underlying>(flag); }

  [[nodiscard]] constexpr bool is_set(Enum flag) const noexcept { return (flags & static_cast<Underlying>(flag)) != 0; }

  constexpr void reset() noexcept { flags = 0; }

  [[nodiscard]] constexpr Flags operator|(Enum flag) const noexcept {
    Flags result = *this;
    result.set(flag);
    return result;
  }

  constexpr Flags& operator|=(Enum flag) noexcept {
    set(flag);
    return *this;
  }

  [[nodiscard]] constexpr Flags operator&(Enum flag) const noexcept {
    Flags result;
    if (is_set(flag)) result.set(flag);
    return result;
  }

  constexpr Flags& operator&=(Enum flag) noexcept {
    if (!is_set(flag)) set(flag);
    return *this;
  }

  [[nodiscard]] constexpr bool operator==(const Flags& other) const noexcept { return flags == other.flags; }

  [[nodiscard]] constexpr bool operator!=(const Flags& other) const noexcept { return !(*this == other); }

  [[nodiscard]] auto value() const { return flags; }
};
} // namespace util