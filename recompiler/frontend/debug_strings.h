#pragma once

#include "shader_types.h"

#include <string_view>

namespace compiler::ir {
struct Operand;
struct OperandType;
} // namespace compiler::ir

namespace compiler::frontend {

namespace debug {
void dumpResource(ResourceVBuffer const&);
void dumpResource(ResourceTBuffer const&);
void dumpResource(ResourceSampler const&);

std::string_view getDebug(eSwizzle swizzle);

void printOperandDst(std::ostream& os, ir::Operand const& op);
void printOperandSrc(std::ostream& os, ir::Operand const& op);
void printType(std::ostream& os, ir::OperandType type);
} // namespace debug
} // namespace compiler::frontend