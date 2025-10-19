#include "regions.h"

#include "analysis.h"
#include "builder.h"
#include "frontend/ir_types.h"
#include "ir/debug_strings.h"
#include "ir/instructions.h"

#include <algorithm>
#include <iostream>
#include <optional>
#include <set>
#include <span>

namespace compiler::frontend::analysis {
void RegionBuilder::addReturn(regionid_t from) {
  auto itfrom     = splitRegion(from + 1);
  auto before     = std::prev(itfrom);
  before->hasJump = true;
}

void RegionBuilder::addJump(regionid_t from, regionid_t to) {
  auto itfrom      = splitRegion(from + 1);
  auto before      = std::prev(itfrom);
  before->trueSucc = to;
  before->hasJump  = true;
  auto itto        = splitRegion(to);
}

void RegionBuilder::addCondJump(regionid_t from, regionid_t to) {
  auto itfrom      = splitRegion(from + 1);
  auto before      = std::prev(itfrom);
  before->trueSucc = to;
  auto itto        = splitRegion(to);
}

fixed_containers::FixedVector<int32_t, 2> RegionBuilder::getSuccessorsIdx(uint32_t region_idx) const {
  fixed_containers::FixedVector<int32_t, 2> result;

  const auto& r = _regions[region_idx];
  if (r.hasFalseSucc()) { // Fallthrough
    if (1 + region_idx < _regions.size() && _regions[1 + region_idx].start == r.end) {
      result.emplace_back(1 + region_idx);
    }
  }
  if (r.hasTrueSucc()) {
    result.emplace_back(getRegionIndex(r.trueSucc));
  }
  return result;
}

std::pair<regionid_t, uint32_t> RegionBuilder::findRegion(uint32_t index) const {
  auto it = std::upper_bound(_regions.begin(), _regions.end(), index, [](uint32_t val, const Region& reg) { return val < reg.start; });
  if (it != _regions.begin()) --it;

  if (it->end < index) return {0, 0}; // Sanity check
  return {it->start, it->end};
}

std::pair<uint32_t, uint32_t> RegionBuilder::getRegion(uint32_t index) const {
  if (index < 0) return {-1, -1};
  auto const& r = _regions[index];
  return {r.start, r.end};
}

regionid_t RegionBuilder::getRegionIndex(uint32_t pos) const {
  auto it = std::upper_bound(_regions.begin(), _regions.end(), pos, [](uint32_t val, const Region& reg) { return val < reg.start; });
  if (it == _regions.begin()) return 0;
  --it;
  return std::distance(_regions.begin(), it);
}

RegionBuilder::regionsit_t RegionBuilder::splitRegion(uint32_t pos) {
  auto it = std::upper_bound(_regions.begin(), _regions.end(), pos, [](uint32_t val, const Region& reg) { return val < reg.start; });
  if (it != _regions.begin()) --it;
  if (pos <= it->start) {
    return it;
  }
  if (pos >= it->end) {
    return ++it;
  }
  // printf("split range @%u [%u,%u) to [%u,%u) [%u,%u)\n", pos, it->start, it->end, it->start, pos, pos, it->end);

  Region after = *it;
  after.start  = pos;
  after.end    = it->end;

  *it = Region(it->start, pos);
  return _regions.insert(1 + it, after);
}

void RegionBuilder::dump(std::ostream& os, void* region) const {
  auto const& r = *(Region const*)region;
  os << "Region [" << std::dec << r.start << ", " << r.end << ")\n";

  os << "  Succ: ";
  visitSuccessors(r.start, [&os](regionid_t item) { os << "[" << item << "]"; });

  os << "\n  Pred: ";
  visitPredecessors(r.start, [&os](regionid_t item) { os << "[" << item << "]"; });
  os << "\n";

  if (r.hasTrueSucc()) os << "  True -> [" << r.trueSucc << "]\n";
  if (r.hasFalseSucc()) os << "  False -> [" << r.end << "]\n";
  os << "  hasJump: " << r.hasJump << "\n";
  os << "-------------------------\n";
}

using namespace compiler::ir;

bool createRegions(std::pmr::polymorphic_allocator<> allocator, std::span<ir::InstCore> instructions, pcmapping_t const& mapping) {
  RegionBuilder regions(instructions.size(), allocator);

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
        auto const targetPc = evaluate(allocator, instructions, regions, n, inst.srcOperands[0]);
        if (!targetPc) return false;

        auto const targetIt = std::lower_bound(mapping.begin(), mapping.end(), targetPc->value_u64, [](auto const& b, uint64_t val) { return b.first < val; });
        regions.addJump(n, targetIt->second);
      } break;
      case eInstKind::CondJumpAbsOp: {
        auto const targetPc = evaluate(allocator, instructions, regions, n, inst.srcOperands[1]);
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
  // auto rootNode = transformStructuredCFG(&builder.getBuffer(), &checkpoint, regions);
  // dump(std::cout, &rootNode);
  // dump(std::cout, &rootNode, instructions.data());

  return true;
}
} // namespace compiler::frontend::analysis