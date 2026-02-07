#pragma once

#include <stdint.h>
#include <vector>

namespace compiler {
using ShaderDump_t = std::vector<uint8_t>;

enum class ShaderBuildFlags : uint16_t {
  ISDUMP   = (1 << 0),
  ISNEO    = (1 << 1),
  ISDEBUG  = (1 << 2),
  WITHDUMP = (1 << 3),
};

} // namespace compiler