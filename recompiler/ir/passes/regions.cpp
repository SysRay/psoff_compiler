#include "../debug_strings.h"
#include "../instructions.h"
#include "builder.h"
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

    std::cout << "get const for ";
    ir::debug::getDebug(std::cout, instructions[index]);
    std::cout << "\n";

    auto const [start, end] = _regions.findRegion(index);
    return findInstruction(reg, index, start);
  }

  private:
  std::optional<InstConstant> check(uint32_t index, regionid_t currentBlock);
  std::optional<InstConstant> findInstruction(Operand const& reg, uint32_t index, regionid_t currentBlock);

  std::optional<InstConstant> evaluate(ir::InstCore const& instr, std::span<InstConstant> inputs) {
    std::cout << "<- evaluate ";
    ir::debug::getDebug(std::cout, instr);
    std::cout << "\n";

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

  std::cout << "\tcheck ";
  ir::debug::getDebug(std::cout, instr);
  std::cout << "\n";

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
      if (instr.dstOperands[d].kind == reg.kind) {
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

        auto const targetIt = std::upper_bound(mapping.begin(), mapping.end(), targetPc->value, [](uint64_t val, auto const& b) { return val < b.first; });
        regions.addJump(n, std::distance(mapping.begin(), targetIt));
      } break;
      case eInstKind::CondJumpAbsOp: {
        auto const targetPc = Evaluate(builder, regions).check(n, inst.srcOperands[1]);
        if (!targetPc) return false;

        auto const targetIt = std::upper_bound(mapping.begin(), mapping.end(), targetPc->value, [](uint64_t val, auto const& b) { return val < b.first; });
        regions.addCondJump(n, std::distance(mapping.begin(), targetIt));
      } break;

      default: break;
    }
  }
  return true;
}
} // namespace compiler::ir::passes