#include "../debug_strings.h"
#include "../instructions.h"
#include "builder.h"
#include "frontend/frontend.h" // todo move to global compare
#include "passes.h"
#include "util/region_graph.h"

#include <algorithm>
#include <iostream>
#include <optional>
#include <set>
#include <span>

namespace compiler::ir::passes {

class Evaluate {
  public:
  Evaluate(Builder& builder, RegionBuilder& regions): _builder(builder), _regions(regions), _visited(&builder.getTempBuffer()) {}

  ~Evaluate() {}

  std::optional<InstConstant> check(uint32_t index, Operand const& reg) {
    auto const& instructions = _builder.getInstructions();
    auto const& instr        = instructions[index];

    //  std::cout << "get const for ";
    // ir::debug::getDebug(std::cout, instructions[index]);
    //  std::cout << "\n";

    auto const [start, end] = _regions.findRegion(index);
    auto const res          = findInstruction(reg, index, start);
    // if (res) std::cout << "result: 0x" << std::hex << res->value_u64 << "\n";
    return res;
  }

  private:
  std::optional<InstConstant> check(uint32_t index, regionid_t currentBlock);
  std::optional<InstConstant> findInstruction(Operand const& reg, uint32_t index, regionid_t currentBlock);

  std::optional<InstConstant> evaluate(ir::InstCore const& instr, std::span<InstConstant> inputs) {
    // std::cout << "<- evaluate ";
    // ir::debug::getDebug(std::cout, instr);
    // std::cout << "\n";

    // todo move this to instructions
    // todo actual evaluate
    return inputs[0];
  }

  std::pmr::set<regionid_t> _visited;
  Builder&                  _builder;
  RegionBuilder&            _regions;
};

std::optional<InstConstant> Evaluate::check(uint32_t index, regionid_t region) {
  auto const& instructions = _builder.getInstructions();
  auto const& instr        = instructions[index];

  // std::cout << "\tcheck ";
  // ir::debug::getDebug(std::cout, instr);
  // std::cout << "\n";

  if (instr.isConstant()) {
    return instr.srcConstant;
  }

  // Check all source operands
  std::array<InstConstant, config::kMaxSrcOps> inputs;
  for (uint8_t s = 0; s < instr.numSrc; s++) {
    auto const res = findInstruction(instr.srcOperands[s], index, region);
    if (!res) return std::nullopt;
    inputs[s] = *res;
  }

  return evaluate(instr, inputs);
}

std::optional<InstConstant> Evaluate::findInstruction(Operand const& reg, uint32_t index, regionid_t region) {
  auto const& instructions = _builder.getInstructions();

  for (int64_t i = index - 1; i >= region; i--) {
    auto const& instr = instructions[i];

    for (uint8_t d = 0; d < instr.numDst; d++) {
      // todo handle multiple regs (64bit, arrays)
      // move to frontend compare
      if (frontend::eOperandKind::import(instr.dstOperands[d].kind).base() == frontend::eOperandKind::import(reg.kind).base()) {
        return check(i, region);
      }
    }
  }

  // todo check other predecessors
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
        auto const targetPc = Evaluate(builder, regions).check(n, inst.srcOperands[0]);
        if (!targetPc) return false;

        auto const targetIt = std::lower_bound(mapping.begin(), mapping.end(), targetPc->value_u64, [](auto const& b, uint64_t val) { return b.first < val; });
        regions.addJump(n, targetIt->second);
      } break;
      case eInstKind::CondJumpAbsOp: {
        auto const targetPc = Evaluate(builder, regions).check(n, inst.srcOperands[1]);
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
  // Structurizer graphBuilder(regions);
  // auto         root = graphBuilder.build();
  // graphBuilder.dump(std::cout, root);

  return true;
}
} // namespace compiler::ir::passes