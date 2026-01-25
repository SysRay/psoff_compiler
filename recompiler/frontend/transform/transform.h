#pragma once

#include "include/checkpoint_resource_fwd.h"
#include "ir/config.h"
#include "ir/rvsdg.h"

#include <memory_resource>
#include <optional>
#include <span>

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
ir::rvsdg::Builder transformRg2Cfg(std::pmr::polymorphic_allocator<> allocator, analysis::RegionBuilder const& rg,
                                   std::span<compiler::InstructionId_t> instructions);
} // namespace compiler::frontend::transform