#pragma once

#include <string_view>

namespace compiler::ir {
struct InputOperand;
struct OutputOperand;
struct OperandType;
} // namespace compiler::ir

namespace compiler::frontend {

namespace debug {

void printOperandDst(std::ostream& os, ir::OutputOperand const& op);
void printOperandSrc(std::ostream& os, ir::InputOperand const& op);
void printType(std::ostream& os, ir::OperandType type);
} // namespace debug
} // namespace compiler::frontend