#pragma once

#include "instructions.h"
#include <span>

namespace compiler::ir::dialect::core {
class IRBuilder {
  ir::InstructionManager& _ir;
  bool                    _isVirtual;

  public:
  IRBuilder(ir::InstructionManager& manager, bool isVirtual = false): _ir(manager), _isVirtual(isVirtual) {}

  IRResult yieldOp(std::span<ir::OutputOperand> results);
};

} // namespace compiler::ir::dialect::core