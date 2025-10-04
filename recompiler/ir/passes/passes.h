#pragma once

#include <memory_resource>

namespace compiler {
class Builder;

namespace ir::passes {

using pcmapping_t = std::pmr::vector<std::pair<uint64_t, uint32_t>>;

bool createRegions(Builder& builder, pcmapping_t const& mapping);
} // namespace ir::passes
} // namespace compiler