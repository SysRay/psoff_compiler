#pragma once
#include "../types.h"

#include <span>

namespace compiler {
class Builder;
}

namespace mlir::func {
class FuncOp;
}

namespace compiler::frontend::transform {
bool transformRg2Cfg(Builder& builder, mlir::func::FuncOp& func, std::span<pcmapping_t> const& mapping);
} // namespace compiler::frontend::transform