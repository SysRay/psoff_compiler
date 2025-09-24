#pragma once

#include "shader_types.h"

#include <string_view>

namespace compiler::frontend {

struct OperandFlagsDst;
struct OperandFlagsSrc;

namespace debug {
void dumpResource(ResourceVBuffer const&);
void dumpResource(ResourceTBuffer const&);
void dumpResource(ResourceSampler const&);

std::string_view getDebug(eSwizzle swizzle);

void printOperand(std::ostream& os, OperandFlagsDst const& op);
void printOperand(std::ostream& os, OperandFlagsSrc const& op);
} // namespace debug
} // namespace compiler::frontend