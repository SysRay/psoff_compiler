#include "region_graph.h"

#include <algorithm>
#include <ostream>

namespace compiler::ir {

void RegionBuilder::addReturn(uint32_t from) {
  auto [beforeStart, afterStart] = splitRegionAround(from + 1);

  auto& before   = regions[getRegionIndex(beforeStart)];
  before.hasJump = true;
}

void RegionBuilder::addJump(uint32_t from, uint32_t to) {
  auto [beforeStart, afterStart] = splitRegionAround(from + 1);
  uint32_t toStart               = splitRegionAt(to);

  auto& before    = regions[getRegionIndex(beforeStart)];
  before.trueSucc = toStart;
  before.hasJump  = true;
}

void RegionBuilder::addCondJump(uint32_t from, uint32_t to) {
  auto [beforeStart, afterStart] = splitRegionAround(from + 1);
  uint32_t toStart               = splitRegionAt(to);

  auto& before     = regions[getRegionIndex(beforeStart)];
  before.trueSucc  = toStart;
  before.falseSucc = afterStart;
  before.hasJump   = true;
}

std::vector<uint32_t> RegionBuilder::getSuccessors(uint32_t start) const {
  std::vector<uint32_t> out;
  const auto&           r = regions[getRegionIndex(start)];

  if (r.hasTrueSucc()) out.push_back(r.trueSucc);
  if (r.hasFalseSucc()) out.push_back(r.falseSucc);

  // Fallthrough
  if (!r.hasJump) {
    size_t idx = getRegionIndex(start);
    if (idx + 1 < regions.size() && regions[idx + 1].start == r.end) out.push_back(regions[idx + 1].start);
  }
  return out;
}

std::vector<uint32_t> RegionBuilder::getPredecessors(uint32_t start) const {
  std::vector<uint32_t> out;
  for (const auto& r: regions) {
    for (auto succ: getSuccessors(r.start)) {
      if (succ == start) out.push_back(r.start);
    }
  }
  return out;
}

void RegionBuilder::dump(std::ostream& os) const {
  for (const auto& r: regions) {
    os << "Region [" << r.start << ", " << r.end << ")\n";
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
}

size_t RegionBuilder::getRegionIndex(uint32_t pos) const {
  auto it = std::upper_bound(regions.begin(), regions.end(), pos, [](uint32_t val, const Region& reg) { return val < reg.start; });
  if (it == regions.begin()) return 0;
  --it;
  return std::distance(regions.begin(), it);
}

uint32_t RegionBuilder::splitRegionAt(uint32_t pos) {
  size_t  idx = getRegionIndex(pos);
  Region& reg = regions[idx];

  if (pos == reg.start || pos >= reg.end) return reg.start;

  Region newReg(pos, reg.end);
  reg.end = pos;

  regions.emplace_back(newReg);
  if (idx + 1 != regions.size() - 1) std::swap(regions[idx + 1], regions.back());

  return newReg.start;
}

std::pair<uint32_t, uint32_t> RegionBuilder::splitRegionAround(uint32_t pos) {
  size_t  idx = getRegionIndex(pos);
  Region& reg = regions[idx];

  if (pos <= reg.start || pos >= reg.end) return {reg.start, reg.start};

  Region after(pos, reg.end);
  reg.end = pos;

  regions.emplace_back(after);
  if (idx + 1 != regions.size() - 1) std::swap(regions[idx + 1], regions.back());

  return {reg.start, after.start};
}
} // namespace compiler::ir