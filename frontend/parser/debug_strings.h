#pragma once
#include "opcodes_table.h"

#include <string_view>

namespace compiler::frontend::debug {
std::string_view getDebug(parser::eOpcode op);
}