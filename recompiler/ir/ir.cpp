#include "ir.h"

namespace compiler::ir {
InstructionId_t InstructionManager::createInstruction(InstCore const& instr, bool isVirtual) {
  auto& item = _instructions.emplace_back(instr);
  if (isVirtual) item.flags |= ir::eInstructionFlags::kVirtual;

  item.dstStartId = static_cast<uint32_t>(_operands.size());
  _operands.resize(_operands.size() + instr.numDst);

  item.srcStartId = static_cast<uint32_t>(_operands.size());
  _operands.resize(_operands.size() + instr.numSrc);

  return static_cast<InstructionId_t>(_instructions.size() - 1);
}

} // namespace compiler::ir