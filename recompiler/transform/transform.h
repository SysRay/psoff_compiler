#pragma once

#include "include/checkpoint_resource_fwd.h"

namespace compiler::ir {

class ControlFlow;

namespace rvsdg {
class IRBlocks;
}
} // namespace compiler::ir

namespace compiler::transform {
void createRVSDG(util::checkpoint_resource& checkpoint_resource, ir::ControlFlow& cfg);
} // namespace compiler::transform