#pragma once

#include "ir/instructions.h"

namespace compiler::frontend::translate {
ir::InstCore createLiteral(uint32_t);
}