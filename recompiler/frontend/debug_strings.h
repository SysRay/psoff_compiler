#pragma once

#include <string_view>

namespace compiler::ir {
struct Operand;
struct OperandType;
} // namespace compiler::ir

namespace compiler::frontend {

namespace debug {

void printOperandDst(std::ostream& os, ir::Operand const& op);
void printOperandSrc(std::ostream& os, ir::Operand const& op);
void printType(std::ostream& os, ir::OperandType type);
} // namespace debug
} // namespace compiler::frontend