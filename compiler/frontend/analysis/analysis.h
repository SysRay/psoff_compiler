#pragma once

#include "types.h"

#include <memory_resource>
#include <optional>
#include <span>

namespace compiler {
class Builder;
}

namespace mlir::func {
class FuncOp;
}

namespace compiler::frontend::analysis {
class RegionBuilder;
bool createRegions(Builder& builder, mlir::func::FuncOp& func, RegionBuilder const& regions, std::span<pcmapping_t> const& mapping);

// std::optional<ir::ConstantValue> evaluate(std::pmr::polymorphic_allocator<> allocator, std::span<ir::InstCore> instructions, RegionBuilder& regions,
//                                          size_t index, ir::Operand const& reg);

} // namespace compiler::frontend::analysis