#pragma once

#include <stdint.h>
#include <vector>

namespace compiler::frontend::analysis {
using pcmapping_t = std::pmr::vector<std::pair<uint64_t, uint32_t>>; ///< pc_t, instruction index
using regionid_t  = uint32_t;

} // namespace compiler::frontend::analysis