#include "../debug_strings.h"
#include "../instructions.h"
#include "builder.h"
#include "frontend/frontend.h" // todo move to global compare
#include "passes.h"
#include "util/cfg_builder.h"

#include <algorithm>
#include <iostream>
#include <optional>
#include <set>
#include <span>

namespace compiler::ir::passes {
bool createRegions(Builder& builder, pcmapping_t const& mapping) {
  auto const&                         instructions = builder.getInstructions();
  std::pmr::monotonic_buffer_resource checkpoint(&builder.getTempBuffer());

  RegionBuilder regions(instructions.size(), checkpoint);

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
        auto const targetPc = evaluate(builder, regions, n, inst.srcOperands[0]);
        if (!targetPc) return false;

        auto const targetIt = std::lower_bound(mapping.begin(), mapping.end(), targetPc->value_u64, [](auto const& b, uint64_t val) { return b.first < val; });
        regions.addJump(n, targetIt->second);
      } break;
      case eInstKind::CondJumpAbsOp: {
        auto const targetPc = evaluate(builder, regions, n, inst.srcOperands[1]);
        if (!targetPc) return false;

        auto const targetIt = std::lower_bound(mapping.begin(), mapping.end(), targetPc->value_u64, [](auto const& b, uint64_t val) { return b.first < val; });
        regions.addCondJump(n, targetIt->second);
      } break;

      default: break;
    }
  }

  regions.for_each([&](uint32_t start, uint32_t end, void* region) {
    regions.dump(std::cout, region);
    for (auto n = start; n < end; ++n) {
      auto it = std::find_if(mapping.begin(), mapping.end(), [n](auto const& item) { return item.second == n; });
      if (it == mapping.end())
        std::cout << "\t";
      else
        std::cout << std::hex << it->first;
      std::cout << '\t' << std::dec << n << "| ";
      ir::debug::getDebug(std::cout, instructions[n]);
    }
  });

  // // transform to hierarchical structured graph
  // ref: "Perfect Reconstructability of Control Flow from Demand Dependence Graphs"
  auto rootNode = transformStructuredCFG(builder.getBuffer(), builder.getTempBuffer(), regions);
  dump(std::cout, &rootNode);

  return true;
}
} // namespace compiler::ir::passes