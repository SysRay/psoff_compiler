#include "ir.h"

namespace compiler::ir {
InstructionId_t InstructionManager::createInstruction(InstCore const& instr) {
  auto& item = _instruction.emplace_back(instr);

  item.dstStartId = static_cast<uint32_t>(_operands.size());
  _operands.resize(_operands.size() + instr.numDst);

  item.srcStartId = static_cast<uint32_t>(_operands.size());
  _operands.resize(_operands.size() + instr.numSrc);

  return static_cast<InstructionId_t>(_instruction.size() - 1);
}

} // namespace compiler::ir