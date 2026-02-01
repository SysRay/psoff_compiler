#pragma once
#include <mlir/IR/Operation.h>
#include <stdint.h>
#include <vector>

namespace compiler::frontend {
using pcmapping_t = std::pair<uint64_t, mlir::Operation*>; ///< pc_t, instruction index
}