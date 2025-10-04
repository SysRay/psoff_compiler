#include "../instructions.h"
#include "builder.h"
#include "passes.h"
#include "util/region_graph.h"

#include <algorithm>
#include <map>
#include <optional>

namespace compiler::ir::passes {

static std::optional<uint64_t> evaluate(Builder& builder, RegionBuilder& regions, size_t index) {
  auto const& instructions = builder.getInstructions();
  for (int32_t n = index; n >= 0; ++n) {
    auto const& inst = instructions[index];
    // todo const pass
  }
  return std::nullopt;
}

bool createRegions(Builder& builder, pcmapping_t const& mapping) {
  auto const& instructions = builder.getInstructions();

  RegionBuilder regions(instructions.size(), builder.getTempBuffer());

  // Collect Labels first
  for (size_t n = 0; n < instructions.size(); ++n) {
    auto const& inst = instructions[n];
    if (inst.group != eInstructionGroup::kFlowControl) continue;

    switch (conv(inst.kind)) {
      case eInstKind::ReturnOp: {
        regions.addReturn(n);
      } break;
      case eInstKind::DiscardOp: {
        // todo needed?
      } break;
      case eInstKind::JumpAbsOp: {
        auto const targetPc = evaluate(builder, regions, n);
        if (!targetPc) return false;

        auto const targetIt = std::upper_bound(mapping.begin(), mapping.end(), *targetPc, [](uint64_t val, auto const& b) { return val < b.first; });

        regions.addJump(n, std::distance(mapping.begin(), targetIt));
      } break;
      case eInstKind::CondJumpAbsOp: {
        auto const targetPc = evaluate(builder, regions, n);
        if (!targetPc) return false;

        auto const targetIt = std::upper_bound(mapping.begin(), mapping.end(), *targetPc, [](uint64_t val, auto const& b) { return val < b.first; });
        regions.addCondJump(n, std::distance(mapping.begin(), targetIt));
      } break;

      default: break;
    }
  }
  return true;
}
} // namespace compiler::ir::passes