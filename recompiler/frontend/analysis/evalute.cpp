#include "builder.h"
#include "cfg_builder.h"
#include "frontend/ir_types.h"
#include "analysis.h"

#include <set>
#include <span>
#include <stack>
#include <unordered_map>
#include <vector>

namespace compiler::frontend::analysis {

class Evaluate {
  public:
  Evaluate(Builder& builder, RegionBuilder& regions)
      : _checkpoint(&builder.getTempBuffer()), _builder(builder), _regions(regions), _visited(&_checkpoint), _instructions(builder.getInstructions()) {}

  ~Evaluate() {}

  std::optional<ir::InstConstant> check(uint32_t index, ir::Operand const& reg) {
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
  std::optional<ir::InstConstant> check(uint32_t index, regionid_t currentBlock);
  std::optional<ir::InstConstant> findInstruction(ir::Operand const& reg, uint32_t index, regionid_t currentBlock);

  std::optional<ir::InstConstant> evaluate(ir::InstCore const& instr, std::span<ir::InstConstant> inputs) {
    // std::cout << "<- evaluate ";
    // ir::debug::getDebug(std::cout, instr);
    // std::cout << "\n";

    // todo move this to instructions
    // todo actual evaluate
    return inputs[0];
  }

  std::pmr::monotonic_buffer_resource _checkpoint;

  Builder&       _builder;
  RegionBuilder& _regions;

  std::pmr::vector<ir::InstCore> const& _instructions;
  std::pmr::set<regionid_t>             _visited;
};

std::optional<ir::InstConstant> Evaluate::check(uint32_t index, regionid_t region) {
  auto const& instr = _instructions[index];

  // std::cout << "\tcheck ";
  // ir::debug::getDebug(std::cout, instr);
  // std::cout << "\n";

  if (instr.isConstant()) {
    return instr.srcConstant;
  }

  // Check all source operands
  std::array<ir::InstConstant, config::kMaxSrcOps> inputs;
  for (uint8_t s = 0; s < instr.numSrc; s++) {
    auto const res = findInstruction(instr.srcOperands[s], index, region);
    if (!res) return std::nullopt;
    inputs[s] = *res;
  }

  return evaluate(instr, inputs);
}

std::optional<ir::InstConstant> Evaluate::findInstruction(ir::Operand const& reg, uint32_t index, regionid_t region) {
  auto const kind = frontend::eOperandKind::import(reg.kind);
  for (int64_t i = index - 1; i >= region; i--) {
    auto const& instr = _instructions[i];

    for (uint8_t d = 0; d < instr.numDst; d++) {
      // todo handle multiple regs (64bit, arrays)
      // move to frontend compare
      if (frontend::eOperandKind::import(instr.dstOperands[d].kind).base() == kind.base()) {
        return check(i, region);
      }
    }
  }

  // todo check other predecessors
  return std::nullopt;
}

std::optional<ir::InstConstant> evaluate(Builder& builder, RegionBuilder& regions, uint32_t index, ir::Operand const& reg) {
  return Evaluate(builder, regions).check(index, reg);
}
} // namespace compiler::frontent::analysis