#include "ir.h"

namespace compiler::ir {
InstructionId_t InstructionManager::createInstruction(InstCore const& instr, bool isVirtual) {
  auto const instrId = static_cast<InstructionId_t>(_instructions.size());
  auto&      item    = _instructions.emplace_back(instr);
  if (isVirtual) item.flags |= ir::eInstructionFlags::kVirtual;

  if (instr.numDst > 0) {
    item.dstStartId = OutputOperandId_t(_outputs.size());
    _outputs.resize(_outputs.size() + instr.numDst);

    auto const ssaStart = _ssa.size();
    _ssa.resize(_ssa.size() + instr.numDst);

    for (uint32_t n = 0; n < instr.numDst; ++n) {
      auto&      out = _outputs[item.dstStartId + n];
      auto const id  = OutputOperandId_t(item.dstStartId + n);

      out.ssa.ssaValue       = SsaId_t(ssaStart + n);
      _ssa[ssaStart + n].def = id;
    }
  }

  item.srcStartId = InputOperandId_t(_inputs.size());
  _inputs.resize(_inputs.size() + instr.numSrc);

  return instrId;
}

InputOperandId_t InstructionManager::createInput(ir::OperandType type) {
  auto  id   = InputOperandId_t(_inputs.size());
  auto& item = _inputs.emplace_back();
  item.type  = type;
  return id;
}

OutputOperandId_t InstructionManager::createOutput(ir::OperandType type) {
  auto  id   = OutputOperandId_t(_outputs.size());
  auto& item = _outputs.emplace_back();
  item.type  = type;

  item.ssa.ssaValue = createSSA(id);
  return id;
}

SsaId_t InstructionManager::createSSA(OutputOperandId_t def) {
  auto const id   = SsaId_t(_ssa.size());
  auto       item = _ssa.emplace_back();

  item.def = def;
  return id;
}
} // namespace compiler::ir