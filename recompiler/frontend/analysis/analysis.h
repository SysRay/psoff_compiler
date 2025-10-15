#pragma once
#include "ir/ir.h"

#include <memory_resource>
#include <optional>

namespace compiler {
class Builder;

namespace frontend {

namespace analysis {
class RegionBuilder;

using pcmapping_t = std::pmr::vector<std::pair<uint64_t, uint32_t>>;
using regionid_t  = uint32_t;

bool createRegions(Builder& builder, pcmapping_t const& mapping);

std::optional<ir::InstConstant> evaluate(Builder& builder, RegionBuilder& regions, uint32_t index, ir::Operand const& reg);
} // namespace analysis
} // namespace frontend
} // namespace compiler