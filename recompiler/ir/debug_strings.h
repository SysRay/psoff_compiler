#pragma once

#include "ir.h"

#include <string_view>

namespace compiler::cfg {
class ControlFlow;

}

namespace compiler::ir::debug {
// todo needs instructionmanager
void getDebug(std::ostream& os, InstructionManager const& im, InstCore const& op);
} // namespace compiler::ir::debug