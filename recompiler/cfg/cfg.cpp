#include "cfg.h"

#include <assert.h>

namespace compiler::cfg {

void ControlFlow::addEdge(blocks::blockid_t from, blocks::blockid_t to) {
  assert(from.isValid() && to.isValid());
  _successors[from.value].push_back(to);
  _predecessors[to.value].push_back(from);
}

void ControlFlow::removeEdge(blocks::blockid_t from, blocks::blockid_t to) {
  assert(from.isValid() && to.isValid());

  auto& succ = _successors[from.value];
  auto& pred = _predecessors[to.value];

  succ.erase(std::remove(succ.begin(), succ.end(), to), succ.end());
  pred.erase(std::remove(pred.begin(), pred.end(), from), pred.end());
}

void ControlFlow::replaceSuccessor(blocks::blockid_t from, blocks::blockid_t oldSucc, blocks::blockid_t newSucc) {
  assert(from.isValid() && oldSucc.isValid() && newSucc.isValid());

  auto& succ = _successors[from.value];
  for (auto& s: succ)
    if (s == oldSucc) {
      s = newSucc;
      break;
    }

  auto& predOld = _predecessors[oldSucc.value];
  predOld.erase(std::remove(predOld.begin(), predOld.end(), from), predOld.end());

  _predecessors[newSucc.value].push_back(from);
}

bool ControlFlow::regionContains(blocks::regionid_t rid, blocks::blockid_t bid) const {
  const auto& R = _regions[rid.value];
  for (auto b: R.blocks)
    if (b == bid) return true;
  return false;
}

/* Move block into region */
void ControlFlow::moveBlockToRegion(blocks::blockid_t bid, blocks::regionid_t dest) {
  assert(bid.isValid() && dest.isValid());

  // Remove from any old region (the block must be in exactly 1 region)
  for (auto& R: _regions) {
    if (R.id.value == dest.value) continue;
    auto& vec = R.blocks;
    auto  it  = std::find(vec.begin(), vec.end(), bid);
    if (it != vec.end()) {
      vec.erase(it);
      break;
    }
  }

  // Insert into new region
  _regions[dest.value].blocks.push_back(bid);
}

void ControlFlow::addSubregion(blocks::regionid_t parent, blocks::regionid_t child) {
  assert(parent.isValid() && child.isValid());
  auto& R = _regions[parent.value];
  R.subregions.push_back(child);
}

void ControlFlow::removeSubregion(blocks::regionid_t parent, blocks::regionid_t child) {
  assert(parent.isValid() && child.isValid());
  auto& R = _regions[parent.value];
  R.subregions.erase(std::remove(R.subregions.begin(), R.subregions.end(), child), R.subregions.end());
}

void ControlFlow::addRegionToNode(blocks::blockid_t node, blocks::regionid_t rid) {
  auto* B = accessBlock(node);
  assert(B->type == blocks::eBlockType::RegionNode);
  auto* R = static_cast<blocks::RegionNode*>(B);
  R->regions.push_back(rid);
}

void ControlFlow::addStructuredSuccessor(blocks::blockid_t node, blocks::blockid_t succ) {
  auto* B = accessBlock(node);
  assert(B->type == blocks::eBlockType::RegionNode);
  auto* R = static_cast<blocks::RegionNode*>(B);
  R->successorsAfterRegions.push_back(succ);
}

void ControlFlow::replaceBlockInRegion(blocks::regionid_t rid, blocks::blockid_t oldB, blocks::blockid_t newB) {
  auto& list = _regions[rid.value].blocks;
  for (auto& b: list)
    if (b == oldB) {
      b = newB;
      return;
    }
}

void ControlFlow::removeBlockFromRegion(blocks::regionid_t rid, blocks::blockid_t bid) {
  auto& list = _regions[rid.value].blocks;
  list.erase(std::remove(list.begin(), list.end(), bid), list.end());
}
} // namespace compiler::cfg
