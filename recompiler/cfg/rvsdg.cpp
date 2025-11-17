#include "rvsdg.h"

#include <assert.h>

namespace compiler::cfg::rvsdg {
bool Builder::regionContains(rvsdg::regionid_t rid, rvsdg::nodeid_t bid) const {
  const auto& R = _regions[rid.value];
  for (auto b: R.nodes)
    if (b == bid) return true;
  return false;
}

void Builder::moveNodeToRegion(rvsdg::nodeid_t bid, rvsdg::regionid_t dest) {
  assert(bid.isValid() && dest.isValid());
  auto base = accessNodeBase(bid);

  // Remove from current region
  auto srcRegionId = base->parentRegion;
  if (srcRegionId.isValid()) {
    auto region = accessRegion(srcRegionId);
    auto it     = std::find(region->nodes.begin(), region->nodes.end(), bid);

    if (it != region->nodes.end()) {
      region->nodes.erase(it);
    }
  }

  // Insert into new region
  _regions[dest.value].nodes.push_back(bid);
  base->parentRegion = dest;
}

void Builder::swapNodeRegion(nodeid_t rid, nodeid_t bid) {
  assert(bid.isValid() && rid.isValid());
  auto baseR = accessNodeBase(rid);
  auto baseB = accessNodeBase(bid);

  if (baseR->parentRegion.isValid()) { // move baseB
    auto region = accessRegion(baseR->parentRegion);
    auto it     = std::find(region->nodes.begin(), region->nodes.end(), rid);
    if (it != region->nodes.end()) {
      *it = bid;
    }
  }

  if (baseB->parentRegion.isValid()) { // move baseB
    auto region = accessRegion(baseB->parentRegion);
    auto it     = std::find(region->nodes.begin(), region->nodes.end(), bid);
    if (it != region->nodes.end()) {
      *it = rid;
    }
  }

  std::swap(baseB->parentRegion, baseR->parentRegion);
}

void Builder::replaceBlockInRegion(rvsdg::regionid_t rid, rvsdg::nodeid_t oldB, rvsdg::nodeid_t newB) {
  auto& list = _regions[rid.value].nodes;
  for (auto& b: list)
    if (b == oldB) {
      b = newB;
      return;
    }
}

void Builder::removeBlockFromRegion(rvsdg::regionid_t rid, rvsdg::nodeid_t bid) {
  auto& list = _regions[rid.value].nodes;
  list.erase(std::remove(list.begin(), list.end(), bid), list.end());
}
} // namespace compiler::cfg::rvsdg
