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
      auto& out              = getDst(instrId, n);
      out.ssa.ssaValue       = SsaId_t(ssaStart + n);
      _ssa[ssaStart + n].def = out.id;
    }
  }

  item.srcStartId = InputOperandId_t(_inputs.size());
  _inputs.resize(_inputs.size() + instr.numSrc);

  return instrId;
}

} // namespace compiler::ir