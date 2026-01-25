#include "cfg.h"

#include <assert.h>

namespace compiler::ir {

void ControlFlow::addEdge(blockid_t from, blockid_t to) {
  assert(from.isValid() && to.isValid());
  _successors[from.value].push_back(to);
  _predecessors[to.value].push_back(from);
}

void ControlFlow::removeEdge(blockid_t from, blockid_t to) {
  assert(from.isValid() && to.isValid());

  auto& succ = _successors[from.value];
  auto& pred = _predecessors[to.value];

  succ.erase(std::remove(succ.begin(), succ.end(), to), succ.end());
  pred.erase(std::remove(pred.begin(), pred.end(), from), pred.end());
}

void ControlFlow::redirectEdge(blockid_t from, blockid_t oldSucc, blockid_t newSucc) {
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

void ControlFlow::redirectEdgeReversed(blockid_t oldPred, blockid_t to, blockid_t newPred) {
  assert(oldPred.isValid() && to.isValid() && newPred.isValid());

  // 1. Fix successors of oldPred -> remove 'to'
  auto& succOld = _successors[oldPred.value];
  succOld.erase(std::remove(succOld.begin(), succOld.end(), to), succOld.end());

  // 2. Fix predecessors of 'to' -> replace oldPred with newPred
  auto& pred = _predecessors[to.value];
  for (auto& p: pred) {
    if (p == oldPred) {
      p = newPred;
      break;
    }
  }

  // 3. Add 'to' to successors of newPred
  _successors[newPred.value].push_back(to);
}
} // namespace compiler::ir
