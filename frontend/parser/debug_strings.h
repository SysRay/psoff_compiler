#pragma once
#include "opcodes_table.h"

#include <string_view>

namespace compiler::frontend::parser {
std::string_view getOpcodeString(eOpcode op);
}