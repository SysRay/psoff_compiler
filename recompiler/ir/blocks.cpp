#include "blocks.h"

#include <assert.h>

namespace compiler::ir::rvsdg {
bool IRBlocks::contains(regionid_t rid, blockid_t bid) const {
  const auto& R = _regions[rid.value];
  for (auto b: R.blocks)
    if (b == bid) return true;
  return false;
}

void IRBlocks::move(blockid_t bid, regionid_t dst) {
  assert(bid.isValid() && dst.isValid());
  auto base = accessBase(bid);
  if (base->parentRegion == dst) return;

  // Remove from current region
  auto srcRegionId = base->parentRegion;
  if (srcRegionId.isValid()) {
    auto region = accessRegion(srcRegionId);
    auto it     = std::find(region->blocks.begin(), region->blocks.end(), bid);

    if (it != region->blocks.end()) {
      region->blocks.erase(it);
    }
  }

  // Insert into new region
  _regions[dst.value].blocks.push_back(bid);
  base->parentRegion = dst;
}

bool IRBlocks::insertToRegion(blockid_t src, blockid_t dst) {
  assert(src.isValid() && dst.isValid());

  auto baseB = accessBase(dst);
  if (!baseB->parentRegion.isValid()) return false;

  auto region = accessRegion(baseB->parentRegion);
  auto it     = std::find(region->blocks.begin(), region->blocks.end(), dst);
  if (it == region->blocks.end()) return false;
  *it = src;

  accessBase(src)->parentRegion = region->id;
  baseB->parentRegion           = regionid_t(); // invalidate
  return true;
}

void IRBlocks::replaceBlockInRegion(regionid_t rid, blockid_t oldB, blockid_t newB) {
  auto& list = _regions[rid.value].blocks;
  for (auto& b: list)
    if (b == oldB) {
      b = newB;
      return;
    }
}

void IRBlocks::removeBlockFromRegion(regionid_t rid, blockid_t bid) {
  auto& list = _regions[rid.value].blocks;
  list.erase(std::remove(list.begin(), list.end(), bid), list.end());
}
} // namespace compiler::ir::rvsdg
