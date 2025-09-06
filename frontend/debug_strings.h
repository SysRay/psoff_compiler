#pragma once

#include "shader_types.h"

#include <string_view>

namespace compiler::frontend {

namespace debug {
void dumpResource(ResourceVBuffer const&);
void dumpResource(ResourceTBuffer const&);
void dumpResource(ResourceSampler const&);

std::string_view getDebug(eSwizzle swizzle);
} // namespace debug
} // namespace compiler::frontend