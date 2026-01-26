#pragma once

#include "ir.h"

#include <string>

namespace compiler::ir {
class IROperations;
class ControlFlow;

namespace rvsdg {
class IRBlocks;
}
} // namespace compiler::ir

namespace compiler::ir::debug {
// todo needs IROperations
void getDebug(std::ostream& os, IROperations const& im, InstCore const& op);

void dump(std::ostream& os, const ControlFlow& cfg);
void dump(std::ostream& os, const rvsdg::IRBlocks& blocks);
} // namespace compiler::ir::debug