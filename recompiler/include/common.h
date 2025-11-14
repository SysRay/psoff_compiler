#pragma once
#include <algorithm>
#include <concepts>

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