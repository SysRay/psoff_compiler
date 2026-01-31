#pragma once

#include "include/checkpoint_resource_fwd.h"
#include "ir/cfg.h"
#include "ir/config.h"

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
bool transformRg2Cfg(std::pmr::polymorphic_allocator<> allocator, analysis::RegionBuilder const& rg, std::span<compiler::InstructionId_t> instructions,
                     ir::ControlFlow& cfg);
} // namespace compiler::frontend::transform