#pragma once
#include "encoding/opcodes_table.h"
#include "shader_types.h"

#include <string_view>

namespace compiler::frontend::debug {
std::string_view getDebug(eOpcode op);
std::string_view getDebug(eSwizzle swizzle);

void dumpResource(std::ostream& os, ResourceVBuffer const&);
void dumpResource(std::ostream& os, ResourceTBuffer const&);
void dumpResource(std::ostream& os, ResourceSampler const&);

} // namespace compiler::frontend::debug