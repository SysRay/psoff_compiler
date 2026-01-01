#include "builder.h"

namespace compiler::ir::dialect::core {

IRResult IRBuilder::yieldOp(std::span<ir::OutputOperand> results) {
  auto info   = getInfo(eInstKind::YieldOp);
  info.numSrc = results.size();

  auto id = _ir.createInstruction(info);

  for (size_t n = 0; n < results.size(); ++n) {
    auto&       src    = _ir.getSrc(id, n);
    auto const& result = results[n];
    src.kind           = result.kind;
    src.flags          = result.flags;
    src.ssaId          = result.ssa.ssaValue;
    src.type           = result.type;
  }

  return IRResult(_ir, id);
}

} // namespace compiler::ir::dialect::core