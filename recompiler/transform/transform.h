#pragma once

#include "include/checkpoint_resource_fwd.h"

namespace compiler::ir::rvsdg {
class IRBlocks;
}

namespace compiler::transform {
void restructureCfg(util::checkpoint_resource& checkpoint_resource, ir::rvsdg::IRBlocks& builder);
} // namespace compiler::transform