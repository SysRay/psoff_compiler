#pragma once

#include "include/checkpoint_resource_fwd.h"

namespace compiler::cfg {
class ControlFlow;
}

namespace compiler::transform {
void restructureCfg(util::checkpoint_resource& checkpoint_resource, cfg::ControlFlow& cfg);
} // namespace compiler::transform