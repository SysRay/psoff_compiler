#pragma once

#include "include/checkpoint_resource_fwd.h"

namespace compiler::ir::rvsdg {
class Builder;
}

namespace compiler::transform {
void restructureCfg(util::checkpoint_resource& checkpoint_resource, ir::rvsdg::Builder& builder);
} // namespace compiler::transform