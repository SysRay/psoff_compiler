#include "region_graph.h"

#include <algorithm>
#include <ostream>

namespace compiler::ir {

void RegionBuilder::addReturn(regionid_t from) {
  auto [beforeStart, afterStart] = splitRegionAround(from + 1);

  auto& before   = _regions[getRegionIndex(beforeStart)];
  before.hasJump = true;
}

void RegionBuilder::addJump(regionid_t from, regionid_t to) {
  auto [beforeStart, afterStart] = splitRegionAround(from + 1);
  regionid_t toStart             = splitRegionAt(to);

  auto& before    = _regions[getRegionIndex(beforeStart)];
  before.trueSucc = toStart;
  before.hasJump  = true;
}

void RegionBuilder::addCondJump(regionid_t from, regionid_t to) {
  auto [beforeStart, afterStart] = splitRegionAround(from + 1);
  regionid_t toStart             = splitRegionAt(to);

  auto& before     = _regions[getRegionIndex(beforeStart)];
  before.trueSucc  = toStart;
  before.falseSucc = afterStart;
  before.hasJump   = true;
}

std::vector<regionid_t> RegionBuilder::getSuccessors(regionid_t start) const {
  std::vector<regionid_t> out;
  const auto&             r = _regions[getRegionIndex(start)];

  // Fallthrough
  if (!r.hasJump) {
    size_t idx = getRegionIndex(start);
    if (idx + 1 < _regions.size() && _regions[idx + 1].start == r.end) out.push_back(_regions[idx + 1].start);
  } else {
    if (r.hasTrueSucc()) out.push_back(r.trueSucc);
    if (r.hasFalseSucc()) out.push_back(r.falseSucc);
  }
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

std::pair<regionid_t, uint32_t> RegionBuilder::getRegion(regionid_t start) const {
  const auto& r = _regions[getRegionIndex(start)];

  if (r.end < start) return {0, 0}; // Sanity check
  return {r.start, r.end};
}

std::pair<regionid_t, uint32_t> RegionBuilder::findRegion(uint32_t index) const {
  auto it = std::upper_bound(_regions.begin(), _regions.end(), index, [](uint32_t val, const Region& reg) { return val < reg.start; });
  if (it != _regions.begin()) --it;

  if (it->end < index) return {0, 0}; // Sanity check
  return {it->start, it->end};
}

void RegionBuilder::dump(std::ostream& os) const {
  for (const auto& r: _regions) {
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

regionid_t RegionBuilder::getRegionIndex(uint32_t pos) const {
  auto it = std::upper_bound(_regions.begin(), _regions.end(), pos, [](uint32_t val, const Region& reg) { return val < reg.start; });
  if (it == _regions.begin()) return 0;
  --it;
  return std::distance(_regions.begin(), it);
}

uint32_t RegionBuilder::splitRegionAt(uint32_t pos) {
  auto const idx = getRegionIndex(pos);
  Region&    reg = _regions[idx];

  if (pos == reg.start || pos >= reg.end) return reg.start;

  Region newReg(pos, reg.end);
  reg.end = pos;

  _regions.emplace_back(newReg);
  if (idx + 1 != _regions.size() - 1) std::swap(_regions[idx + 1], _regions.back());

  return newReg.start;
}

std::pair<regionid_t, uint32_t> RegionBuilder::splitRegionAround(uint32_t pos) {
  auto const idx = getRegionIndex(pos);
  Region&    reg = _regions[idx];

  if (pos <= reg.start || pos >= reg.end) return {reg.start, reg.start};

  Region after(pos, reg.end);
  reg.end = pos;

  _regions.emplace_back(after);
  if (idx + 1 != _regions.size() - 1) std::swap(_regions[idx + 1], _regions.back());

  return {reg.start, after.start};
}
} // namespace compiler::ir