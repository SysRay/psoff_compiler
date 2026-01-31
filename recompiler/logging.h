#pragma once
#include <cstdarg>
#include <cstdio>
#include <format>
#include <string_view>
#include <type_traits>

namespace compiler {
auto inline width(size_t depth) {
  return std::string(depth * 2, ' ');
}

enum class eLOG_TYPE : uint8_t { DEBUG, INFO, ERROR };
void __LOG_IMPL(eLOG_TYPE, std::string_view);

template <typename... Args>
void LOG(eLOG_TYPE type, std::string_view fmt, Args&&... args) {
  if constexpr (sizeof...(Args) == 0) {
    __LOG_IMPL(type, fmt);
  } else {
    __LOG_IMPL(type, std::vformat(fmt, std::make_format_args(args...)));
  }
}
} // namespace compiler