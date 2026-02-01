#pragma once
#include "../types.h"

#include <span>
namespace compiler {
class Builder;
}

namespace compiler::frontend::transform {
bool transformRg2Cfg(Builder& builder, std::span<pcmapping_t> const& mapping);
} // namespace compiler::frontend::transform