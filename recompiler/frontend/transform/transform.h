#pragma once

#include "cfg/types.h"
#include "include/checkpoint_resource_fwd.h"

#include <memory_resource>
#include <optional>
#include <span>

namespace compiler::cfg {
class ControlFlow;
}

namespace compiler::frontend::analysis {
class RegionBuilder;
}

namespace compiler::frontend::transform {
/**
 * @brief Creates a CFG out from code regions
 *
 * @param allocator Used in control flow
 * @param rg
 */
cfg::ControlFlow transformRg2Cfg(std::pmr::polymorphic_allocator<> allocator, analysis::RegionBuilder const& rg);
} // namespace compiler::frontend::transform