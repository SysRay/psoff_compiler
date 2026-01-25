#include "blocks.h"

#include <assert.h>

namespace compiler::ir::rvsdg {
bool IRBlocks::regionContains(regionid_t rid, nodeid_t bid) const {
  const auto& R = _regions[rid.value];
  for (auto b: R.nodes)
    if (b == bid) return true;
  return false;
}

void IRBlocks::moveNodeToRegion(nodeid_t bid, regionid_t dst) {
  assert(bid.isValid() && dst.isValid());
  auto base = accessNodeBase(bid);
  if (base->parentRegion == dst) return;

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
  _regions[dst.value].nodes.push_back(bid);
  base->parentRegion = dst;
}

bool IRBlocks::insertNodeToRegion(nodeid_t src, nodeid_t dst) {
  assert(src.isValid() && dst.isValid());

  auto baseB = accessNodeBase(dst);
  if (!baseB->parentRegion.isValid()) return false;

  auto region = accessRegion(baseB->parentRegion);
  auto it     = std::find(region->nodes.begin(), region->nodes.end(), dst);
  if (it == region->nodes.end()) return false;
  *it = src;

  accessNodeBase(src)->parentRegion = region->id;
  baseB->parentRegion               = regionid_t(); // invalidate
  return true;
}

void IRBlocks::replaceBlockInRegion(regionid_t rid, nodeid_t oldB, nodeid_t newB) {
  auto& list = _regions[rid.value].nodes;
  for (auto& b: list)
    if (b == oldB) {
      b = newB;
      return;
    }
}

void IRBlocks::removeBlockFromRegion(regionid_t rid, nodeid_t bid) {
  auto& list = _regions[rid.value].nodes;
  list.erase(std::remove(list.begin(), list.end(), bid), list.end());
}
} // namespace compiler::ir::rvsdg
