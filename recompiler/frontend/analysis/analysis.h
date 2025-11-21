#pragma once
#include "ir/ir.h"
#include "types.h"

#include <memory_resource>
#include <optional>
#include <span>

namespace compiler::frontend::analysis {

bool createRegions(std::pmr::polymorphic_allocator<> allocator, ir::InstructionManager const& instructions, pcmapping_t const& mapping);

class RegionBuilder;

// std::optional<ir::ConstantValue> evaluate(std::pmr::polymorphic_allocator<> allocator, std::span<ir::InstCore> instructions, RegionBuilder& regions,
//                                          size_t index, ir::Operand const& reg);

} // namespace compiler::frontend::analysis