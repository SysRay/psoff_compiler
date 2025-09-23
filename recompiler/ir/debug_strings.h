#pragma once

#include "ir.h"

#include <string_view>

namespace compiler::ir::debug {
void getDebug(std::ostream& os, InstCore const& op);
}