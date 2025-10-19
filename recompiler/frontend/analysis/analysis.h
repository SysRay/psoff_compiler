#pragma once
#include "ir/ir.h"
#include "types.h"

#include <memory_resource>
#include <optional>
#include <span>

namespace compiler::frontend::analysis {

bool createRegions(std::pmr::polymorphic_allocator<> allocator, std::span<ir::InstCore> instructions, pcmapping_t const& mapping);

class RegionBuilder;
std::optional<ir::InstConstant> evaluate(std::pmr::polymorphic_allocator<> allocator, std::span<ir::InstCore> instructions, RegionBuilder& regions,
                                         uint32_t index, ir::Operand const& reg);
} // namespace compiler::frontend::analysis