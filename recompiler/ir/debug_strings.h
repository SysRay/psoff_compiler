#pragma once

#include "ir.h"

#include <string_view>

namespace compiler::ir::cfg {
class ControlFlow;

}

namespace compiler::ir::debug {
void getDebug(std::ostream& os, InstCore const& op);

void dump(std::ostream& os, const compiler::ir::cfg::ControlFlow& g);
} // namespace compiler::ir::debug