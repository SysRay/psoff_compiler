#include "evaluate.h"

#include "analysis.h"
#include "builder.h"
#include "frontend/ir_types.h"

#include <set>
#include <span>
#include <stack>
#include <unordered_map>
#include <vector>

namespace compiler::frontend::analysis {

std::optional<ir::ConstantValue> Evaluate::check(size_t index, ir::Operand const& reg) {
  auto const& instr = _instructions[index];

  //  std::cout << "get const for ";
  // ir::debug::getDebug(std::cout, instructions[index]);
  //  std::cout << "\n";

  auto const regionId = _regions.getRegionIndex((region_t)index);
  auto const res      = findInstruction(reg, index, regionId);
  // if (res) std::cout << "result: 0x" << std::hex << res->value_u64 << "\n";
  return res;
}

std::optional<ir::ConstantValue> Evaluate::check(size_t index, regionid_t regionId) {
  auto const& instr = _instructions[index];

  // std::cout << "\tcheck ";
  // ir::debug::getDebug(std::cout, instr);
  // std::cout << "\n";

  // todo
  // if (instr.isConstant()) {
  //   return instr.srcConstant;
  // }

  // // Check all source operands
  // std::array<ir::ConstantValue, config::kMaxSrcOps> inputs;
  // for (uint8_t s = 0; s < instr.numSrc; s++) {
  //   auto const res = findInstruction(instr.srcOperands[s], index, regionId);
  //   if (!res) return std::nullopt;
  //   inputs[s] = *res;
  // }

  // return evaluate(instr, inputs);
  return std::nullopt;
}

std::optional<ir::ConstantValue> Evaluate::findInstruction(ir::Operand const& reg, size_t index, regionid_t regionId) {
  auto const [start, end] = _regions.getRegion(regionId);

  // todo

  // auto const kind = frontend::eOperandKind::import(reg.kind);
  // for (int64_t i = index - 1; i >= start; i--) {
  //   auto const& instr = _instructions[i];

  //   for (uint8_t d = 0; d < instr.numDst; d++) {
  //     // todo handle multiple regs (64bit, arrays)
  //     // move to frontend compare
  //     if (frontend::eOperandKind::import(instr.dstOperands[d].kind).base() == kind.base()) {
  //       return check(i, regionId);
  //     }
  //   }
  // }

  // todo check other predecessors
  return std::nullopt;
}

std::optional<ir::ConstantValue> evaluate(std::pmr::polymorphic_allocator<> allocator, std::span<ir::InstCore> instructions, RegionBuilder& regions,
                                          size_t index, ir::Operand const& reg) {
  return Evaluate(allocator.resource(), instructions, regions).check(index, reg);
}
} // namespace compiler::frontend::analysis