#pragma once

#include "ir/cfg/types.h"
#include "ir/ir.h"
#include "regions.h"
#include "types.h"

#include <optional>
#include <set>
#include <vector>

namespace compiler::frontend::analysis {
class Evaluate {
  public:
  Evaluate(std::pmr::memory_resource* pool, std::span<ir::InstCore> instructions, RegionBuilder& regions)
      : _checkpoint(pool), _regions(regions), _visited(&_checkpoint), _instructions(instructions) {}

  ~Evaluate() {}

  std::optional<ir::InstConstant> check(uint32_t index, ir::Operand const& reg);

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

  RegionBuilder& _regions;

  std::span<ir::InstCore>   _instructions;
  std::pmr::set<regionid_t> _visited;
};

} // namespace compiler::frontend::analysis