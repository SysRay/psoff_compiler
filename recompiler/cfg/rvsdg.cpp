#include "rvsdg.h"

#include <assert.h>

namespace compiler::cfg::rvsdg {
bool Builder::regionContains(rvsdg::regionid_t rid, rvsdg::nodeid_t bid) const {
  const auto& R = _regions[rid.value];
  for (auto b: R.nodes)
    if (b == bid) return true;
  return false;
}

void Builder::moveBlockToRegion(rvsdg::nodeid_t bid, rvsdg::regionid_t dest) {
  assert(bid.isValid() && dest.isValid());

  // Remove from any old region (the block must be in exactly 1 region)
  for (auto& R: _regions) {
    if (R.id.value == dest.value) continue;
    auto& vec = R.nodes;
    auto  it  = std::find(vec.begin(), vec.end(), bid);
    if (it != vec.end()) {
      vec.erase(it);
      break;
    }
  }

  // Insert into new region
  _regions[dest.value].nodes.push_back(bid);
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
