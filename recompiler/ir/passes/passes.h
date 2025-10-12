#pragma once
#include "ir/ir.h"

#include <memory_resource>
#include <optional>

namespace compiler {
class Builder;

namespace ir {
class RegionBuilder;

namespace passes {

using pcmapping_t = std::pmr::vector<std::pair<uint64_t, uint32_t>>;
using regionid_t  = uint32_t;

bool createRegions(Builder& builder, pcmapping_t const& mapping);

std::optional<InstConstant> evaluate(Builder& builder, RegionBuilder& regions, uint32_t index, Operand const& reg);
} // namespace passes
} // namespace ir
} // namespace compiler