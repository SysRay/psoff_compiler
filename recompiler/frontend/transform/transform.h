#pragma once

#include "ir/cfg/types.h"

#include <memory_resource>
#include <optional>
#include <span>

namespace compiler::frontend::analysis {
class RegionGraph;
}

namespace compiler::frontend::transform {
/**
 * @brief Create a structured cfg from regions
 * Note: From "Perfect Reconstructability of control flow"
 * @param allocPool
 * @param tempPool
 * @param regions
 */
ir::cfg::ControlFlow transformRegions(std::pmr::polymorphic_allocator<> allocPool, std::pmr::memory_resource* tempPool, analysis::RegionGraph& regionGraph);
} // namespace compiler::frontend::transform