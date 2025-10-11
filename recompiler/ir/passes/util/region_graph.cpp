#include "../../debug_strings.h"
#include "structured_graph.h"

#include <algorithm>
#include <ostream>

namespace compiler::ir {

void RegionBuilder::addReturn(regionid_t from) {
  auto itfrom     = splitRegion(from + 1);
  itfrom->hasJump = true;
}

void RegionBuilder::addJump(regionid_t from, regionid_t to) {
  auto itfrom      = splitRegion(from + 1);
  auto before      = std::prev(itfrom);
  before->trueSucc = to;
  before->hasJump  = true;
  printf("asd %u\n", before->start);
  auto itto = splitRegion(to);
}

void RegionBuilder::addCondJump(regionid_t from, regionid_t to) {
  auto itfrom      = splitRegion(from + 1);
  auto before      = std::prev(itfrom);
  before->trueSucc = to;
  auto itto        = splitRegion(to);
}

std::vector<regionid_t> RegionBuilder::getSuccessors(regionid_t start) const {
  std::vector<regionid_t> out;
  const auto&             r = _regions[getRegionIndex(start)];

  // Fallthrough
  if (!r.hasJump) {
    size_t idx = getRegionIndex(start);
    if (idx + 1 < _regions.size() && _regions[idx + 1].start == r.end) out.push_back(_regions[idx + 1].start);
  }
  if (r.hasTrueSucc()) out.push_back(r.trueSucc);
  if (r.hasFalseSucc()) out.push_back(r.falseSucc);

  return out;
}

std::vector<regionid_t> RegionBuilder::getPredecessors(regionid_t start) const {
  std::vector<regionid_t> out;
  for (const auto& r: _regions) {
    for (auto succ: getSuccessors(r.start)) {
      if (succ == start) out.push_back(r.start);
    }
  }
  return out;
}

std::pair<regionid_t, uint32_t> RegionBuilder::findRegion(uint32_t index) const {
  auto it = std::upper_bound(_regions.begin(), _regions.end(), index, [](uint32_t val, const Region& reg) { return val < reg.start; });
  if (it != _regions.begin()) --it;

  if (it->end < index) return {0, 0}; // Sanity check
  return {it->start, it->end};
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
  if (pos >= it->end || pos <= it->start) return it;

  printf("split range @%u [%u,%u) to [%u,%u) [%u,%u)\n", pos, it->start, it->end, it->start, pos, pos, it->end);

  Region after = *it;
  after.start  = pos;
  after.end    = it->end;

  *it = Region(it->start, pos);
  return _regions.insert(1 + it, after);
}

void RegionBuilder::dump(std::ostream& os, void* region) const {
  auto const& r = *(Region const*)region;
  os << "Region [" << std::dec << r.start << ", " << r.end << ")\n";
  auto succ = getSuccessors(r.start);
  auto pred = getPredecessors(r.start);

  os << "  Succ: ";
  for (auto s: succ)
    os << "[" << s << "]";
  os << "\n  Pred: ";
  for (auto p: pred)
    os << "[" << p << "]";
  os << "\n";

  if (r.hasTrueSucc()) os << "  True -> [" << r.trueSucc << "]\n";
  if (r.hasFalseSucc()) os << "  False -> [" << r.falseSucc << "]\n";
  os << "  hasJump: " << r.hasJump << "\n";
  os << "-------------------------\n";
}
} // namespace compiler::ir