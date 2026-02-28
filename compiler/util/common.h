#pragma once
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <stdint.h>

#ifdef _MSC_VER
#define PACK(__Declaration__) __pragma(pack(push, 1)) __Declaration__ __pragma(pack(pop))
#else
#define PACK(__Declaration__) __Declaration__ __attribute__((__packed__))
#endif

// Concept to check if container supports push_back
template <typename C, typename T>
concept HasPushBack = requires(C& c, T value) { c.push_back(value); };

// Concept to check if container supports insert
template <typename C, typename T>
concept HasInsert = requires(C& c, T value) { c.insert(value); };

template <typename Container, typename T>
auto addUnique(Container& container, const T& value) -> typename Container::iterator {
  // Check if value already exists
  auto it = std::find(container.begin(), container.end(), value);

  if (it != container.end()) return it; // Return iterator to existing element

  if constexpr (HasPushBack<Container, T>) {
    container.push_back(value);
    return std::prev(container.end()); // Return iterator to newly added element
  } else if constexpr (HasInsert<Container, T>) {
    auto [insert_it, inserted] = container.insert(value);
    return insert_it;
  } else {
    static_assert(HasPushBack<Container, T> || HasInsert<Container, T>, "Container must support push_back or insert");
  }
}

template <typename Container, typename T>
bool contains(const Container& container, const T& value) {
  return std::find(container.begin(), container.end(), value) != container.end();
}

namespace compiler {
template <typename Tag, typename T = uint32_t>
struct id_t {
  using underlying_t = T;

  static inline constexpr id_t NO_VALUE() { return id_t(std::numeric_limits<underlying_t>::max()); };

  underlying_t value = NO_VALUE().value;

  constexpr id_t() = default;

  constexpr explicit id_t(underlying_t v): value(v) {}

  constexpr operator underlying_t() const { return value; }

  constexpr bool operator==(id_t const&) const = default;

  constexpr bool isValid() const { return value != NO_VALUE().value; }
};

#define CLASS_NO_COPY(name)                                                                                                                                    \
  name(const name&)            = delete;                                                                                                                       \
  name& operator=(const name&) = delete;

#define CLASS_NO_MOVE(name)                                                                                                                                    \
  name(name&&) noexcept            = delete;                                                                                                                   \
  name& operator=(name&&) noexcept = delete
} // namespace compiler