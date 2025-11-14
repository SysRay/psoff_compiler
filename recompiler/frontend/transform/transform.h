#pragma once

#include "include/checkpoint_resource_fwd.h"
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
void reconstructSCF(util::checkpoint_resource& checkpoint_resource, analysis::RegionGraph& regionGraph);
} // namespace compiler::frontend::transform