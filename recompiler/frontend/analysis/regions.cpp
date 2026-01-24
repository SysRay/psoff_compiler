#include "regions.h"

#include "analysis.h"
#include "builder.h"
#include "frontend/ir_types.h"
#include "include/checkpoint_resource.h"
#include "include/common.h"
#include "ir/debug_strings.h"

#include <algorithm>
#include <iostream>
#include <optional>
#include <set>
#include <span>
#include <stack>
#include <unordered_set>

namespace compiler::frontend::analysis {

void RegionBuilder::addReturn(region_t from) {
  _jumpInfo.push_back({from, 0, false, true});
  _splitPoints.push_back(from + 1);
}

void RegionBuilder::addJump(region_t from, region_t to) {
  _jumpInfo.push_back({from, to, false, false});
  _splitPoints.push_back(from + 1);
  _splitPoints.push_back(to);
}

void RegionBuilder::addCondJump(region_t from, region_t to) {
  _jumpInfo.push_back({from, to, true, false});
  _splitPoints.push_back(from + 1);
  _splitPoints.push_back(to);
}

void RegionBuilder::buildRegionsFromSplits() {
  if (_splitPoints.empty()) {
    // No splits, single region
    _regions.emplace_back(0, _endPosition);
    return;
  }

  // Sort and deduplicate split points
  std::sort(_splitPoints.begin(), _splitPoints.end());
  auto last = std::unique(_splitPoints.begin(), _splitPoints.end());
  _splitPoints.erase(last, _splitPoints.end());

  // Reserve exact space needed
  size_t numRegions = _splitPoints.size() + 1;
  _regions.reserve(numRegions);

  // Build regions in single pass
  region_t start = 0;
  for (region_t split: _splitPoints) {
    if (split > start && split <= _endPosition) {
      _regions.emplace_back(start, split);
      start = split;
    }
  }

  // Final region
  if (start < _endPosition) {
    _regions.emplace_back(start, _endPosition);
  }
}

void RegionBuilder::applyJumpInfo() {
  for (const auto& jump: _jumpInfo) {
    // Find the region containing 'from'
    regionid_t fromIdx = getRegionIndex(jump.from);

    if (fromIdx >= _regions.size()) continue;

    auto& region = _regions[fromIdx];

    if (jump.isReturn) {
      region.hasJump = true;
    } else if (jump.isConditional) {
      region.trueSucc = jump.to;
      // Conditional jumps can fall through, so no hasJump = true
    } else {
      // Unconditional jump
      region.trueSucc = jump.to;
      region.hasJump  = true;
    }
  }
}

void RegionBuilder::finalize() {
  if (_finalized) return;

  buildRegionsFromSplits();
  applyJumpInfo();

  _finalized = true;

  // Clear temporary data to free memory
  _splitPoints.clear();
  _splitPoints.shrink_to_fit();
  _jumpInfo.clear();
  _jumpInfo.shrink_to_fit();
}

fixed_containers::FixedVector<regionid_t, 2> RegionBuilder::getSuccessorsIdx(regionid_t id) const {
  fixed_containers::FixedVector<regionid_t, 2> result;

  const auto& r = _regions[id];
  if (r.hasFalseSucc()) { // Fallthrough
    if (1 + id < _regions.size() && _regions[1 + id].start == r.end) {
      result.emplace_back(1 + id);
    }
  }
  if (r.hasTrueSucc()) {
    result.emplace_back(getRegionIndex(r.trueSucc));
  }
  return result;
}

std::pair<region_t, region_t> RegionBuilder::findRegion(region_t from) const {
  auto it = std::upper_bound(_regions.begin(), _regions.end(), from, [](region_t val, const Region& reg) { return val < reg.start; });

  if (it != _regions.begin()) --it;

  if (it->end <= from) return std::make_pair(0, 0); // Sanity check
  return std::make_pair(it->start, it->end);
}

std::pair<region_t, region_t> RegionBuilder::getRegion(regionid_t id) const {
  assert(id != NO_REGION);
  auto const& r = _regions[id];
  return std::make_pair(r.start, r.end);
}

regionid_t RegionBuilder::getRegionIndex(region_t pos) const {
  auto it = std::upper_bound(_regions.begin(), _regions.end(), pos, [](region_t val, const Region& reg) { return val < reg.start; });

  if (it == _regions.begin()) return regionid_t(0);
  --it;
  return regionid_t(std::distance(_regions.begin(), it));
}

void RegionBuilder::dump(std::ostream& os, void* region) const {
  auto const& r = *(Region const*)region;
  os << "Region [" << std::dec << r.start << ", " << r.end << ")\n";

  os << "  Succ: ";
  visitSuccessors(r.start, [&os](region_t item) { os << "[" << item << "]"; });

  os << "\n  Pred: ";
  visitPredecessors(r.start, [&os](region_t item) { os << "[" << item << "]"; });
  os << "\n";

  if (r.hasTrueSucc()) os << "  True -> [" << r.trueSucc << "]\n";
  if (r.hasFalseSucc()) os << "  False -> [" << r.end << "]\n";
  os << "  hasJump: " << r.hasJump << "\n";
  os << "-------------------------\n";
}

using namespace compiler::ir;

bool createRegions(std::pmr::polymorphic_allocator<> allocator, ir::InstructionManager const& manager, pcmapping_t const& mapping) {
  RegionBuilder regions(manager.instructionCount(), allocator);

  // 1: Collect all jumps (no region splitting yet)
  for (size_t n = 0; n < manager.instructionCount(); ++n) {
    auto const& inst = manager.get()[n];
    if (inst.flags.is_set(eInstructionFlags::kTerminator)) continue;

    // todo
    // switch (conv(inst.kind)) {
    //   case eInstKind::ReturnOp: {
    //     regions.addReturn(n);
    //   } break;

    //   case eInstKind::DiscardOp: {
    //     // todo needed?
    //   } break;

    //   case eInstKind::JumpAbsOp: {
    //     // todo
    //     // auto const targetPc = evaluate(allocator, instructions, regions, n, inst.srcOperands[0]);
    //     // if (!targetPc) return false;

    //     // auto const targetIt = std::lower_bound(mapping.begin(), mapping.end(), targetPc->value_u64, [](auto const& b, uint64_t val) { return b.first <
    //     val;
    //     // }); regions.addJump(n, targetIt->second);
    //   } break;

    //   case eInstKind::CondJumpAbsOp: {
    //     // todo
    //     // auto const targetPc = evaluate(allocator, instructions, regions, n, inst.srcOperands[1]);
    //     // if (!targetPc) return false;

    //     // auto const targetIt = std::lower_bound(mapping.begin(), mapping.end(), targetPc->value_u64, [](auto const& b, uint64_t val) { return b.first <
    //     val;
    //     // }); regions.addCondJump(n, targetIt->second);
    //   } break;

    //   default: break;
    // }
  }

  // 2: Build regions in single pass
  regions.finalize();

  // Output/debug
  // regions.for_each([&](uint32_t start, uint32_t end, void* region) {
  //   regions.dump(std::cout, region);
  //   for (auto n = start; n < end; ++n) {
  //     auto it = std::find_if(mapping.begin(), mapping.end(), [n](auto const& item) { return item.second == n; });
  //     if (it == mapping.end())
  //       std::cout << "\t";
  //     else
  //       std::cout << std::hex << it->first;
  //     std::cout << '\t' << std::dec << n << "| ";
  //     ir::debug::getDebug(std::cout, manager.getInstr(n));
  //   }
  // });

  return true;
}

} // namespace compiler::frontend::analysis