#include "cfg_builder.h"

#include "../../debug_strings.h"

#include <algorithm>
#include <iostream>
#include <ostream>
#include <set>
#include <stack>
#include <unordered_set>

namespace compiler::ir {
static std::string indent(int level) {
  return std::string(level * 2, ' ');
}

static std::string nodeKindName(SimpleNodeKind kind) {
  switch (kind) {
    case SimpleNodeKind::Basic: return "Basic";
    case SimpleNodeKind::Branch: return "Branch";
    case SimpleNodeKind::Loop: return "Loop";
    case SimpleNodeKind::Sequence: return "Sequence";
    default: return "Unknown";
  }
}

void RegionBuilder::addReturn(regionid_t from) {
  auto itfrom     = splitRegion(from + 1);
  itfrom->hasJump = true;
}

void RegionBuilder::addJump(regionid_t from, regionid_t to) {
  auto itfrom      = splitRegion(from + 1);
  auto before      = std::prev(itfrom);
  before->trueSucc = to;
  before->hasJump  = true;
  auto itto        = splitRegion(to);
}

void RegionBuilder::addCondJump(regionid_t from, regionid_t to) {
  auto itfrom      = splitRegion(from + 1);
  auto before      = std::prev(itfrom);
  before->trueSucc = to;
  auto itto        = splitRegion(to);
}

fixed_containers::FixedVector<int32_t, 2> RegionBuilder::getSuccessorsIdx(uint32_t region_idx) const {
  fixed_containers::FixedVector<int32_t, 2> result;

  const auto& r = _regions[region_idx];
  if (r.hasTrueSucc()) {
    result.emplace_back(getRegionIndex(r.trueSucc));
  }
  if (r.hasFalseSucc()) { // Fallthrough
    if (1 + region_idx < _regions.size() && _regions[1 + region_idx].start == r.end) {
      result.emplace_back(1 + region_idx);
    }
  }
  return result;
}

std::vector<regionid_t> RegionBuilder::getSuccessors(regionid_t start) const {
  std::vector<regionid_t> out;
  const auto&             r = _regions[getRegionIndex(start)];

  if (r.hasTrueSucc()) out.push_back(r.trueSucc);
  if (r.hasFalseSucc()) { // Fallthrough
    size_t idx = getRegionIndex(start);
    if (idx + 1 < _regions.size() && _regions[idx + 1].start == r.end) out.push_back(_regions[idx + 1].start);
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

std::pair<regionid_t, uint32_t> RegionBuilder::findRegion(uint32_t index) const {
  auto it = std::upper_bound(_regions.begin(), _regions.end(), index, [](uint32_t val, const Region& reg) { return val < reg.start; });
  if (it != _regions.begin()) --it;

  if (it->end < index) return {0, 0}; // Sanity check
  return {it->start, it->end};
}

std::pair<uint32_t, uint32_t> RegionBuilder::getRegion(uint32_t index) const {
  if (index < 0) return {-1, -1};
  auto const& r = _regions[index];
  return {r.start, r.end};
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

  // printf("split range @%u [%u,%u) to [%u,%u) [%u,%u)\n", pos, it->start, it->end, it->start, pos, pos, it->end);

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
  if (r.hasFalseSucc()) os << "  False -> [" << r.end << "]\n";
  os << "  hasJump: " << r.hasJump << "\n";
  os << "-------------------------\n";
}

// // transform structured
using scc_t = std::pmr::vector<std::pmr::set<regionid_t>>;

struct SCCBuild {
  SCCBuild(std::pmr::memory_resource* pool, RegionBuilder const& regions)
      : _pool(pool), _regions(regions), _state(regions.getNumRegions(), TarjanState {}, _pool), _stack(_pool), _sccs(pool) {}

  inline scc_t calculate() {
    for (int32_t i = 0; i < _regions.getNumRegions(); ++i) {
      if (_state[i].index == -1) { // If node 'i' hasn't been visited yet
        strongConnect(i);
      }
    }
    return std::move(_sccs);
  }

  private:
  void strongConnect(int32_t v);

  struct TarjanState {
    int32_t index   = -1;
    int32_t lowlink = -1;
    bool    onStack = false;
  };

  std::pmr::memory_resource* _pool;
  RegionBuilder const&       _regions;

  std::pmr::vector<TarjanState> _state;
  std::pmr::vector<regionid_t>  _stack;
  int32_t                       _tarjanIndex = 0;
  scc_t                         _sccs;
};

void SCCBuild::strongConnect(int32_t v) {
  auto& state   = _state[v];
  state.index   = _tarjanIndex;
  state.lowlink = _tarjanIndex;
  _tarjanIndex++;
  _stack.push_back(v);
  state.onStack = true;

  auto succs = _regions.getSuccessorsIdx(v);
  for (auto w: succs) {
    auto& item = _state[w];
    if (item.index == -1) {
      strongConnect(w);
      state.lowlink = std::min(state.lowlink, _state[w].lowlink);
    } else if (item.onStack) {
      state.lowlink = std::min(state.lowlink, item.index);
    }
  }

  if (state.lowlink == state.index) {
    std::pmr::set<regionid_t> scc {_pool};
    regionid_t                w;
    do {
      w = _stack.back();
      _stack.pop_back();
      _state[w].onStack = false;
      scc.insert(w);
    } while (w != v);

    _sccs.push_back(std::move(scc));
  }
}

void dump(std::ostream& os, scc_t const& scc) {
  os << "\n Strongly Connected:\n";
  for (auto const& node: scc) {
    os << '{' << std::dec;
    for (auto id: node)
      os << id << ",";
    os << "}\n";
  }
}

using RegionMask       = std::pmr::vector<char>;
using SimpleNodeTemp_t = SimpleNode<std::list>;

class CFGBuilder {
  public:
  explicit CFGBuilder(std::pmr::memory_resource* tempPool, RegionBuilder& regions): _tempPool(tempPool), _regions(regions), _sccs(_tempPool) {}

  SimpleNodeTemp_t build();

  private:
  bool             isLinearIdx(regionid_t entry, const RegionMask& regionMask, const std::pmr::vector<std::pmr::vector<regionid_t>>& predsCache) const;
  SimpleNodeTemp_t restructureLoopIdx(const std::pmr::set<regionid_t>& scc, regionid_t entry, const std::pmr::vector<std::pmr::vector<regionid_t>>& predsCache);
  SimpleNodeTemp_t restructureAcyclicIdx(regionid_t entry, const RegionMask& regionMask, const std::pmr::vector<std::pmr::vector<regionid_t>>& predsCache);
  SimpleNodeTemp_t restructureLoopBodyIdx(regionid_t entry, const RegionMask& sccMask, const std::pmr::vector<std::pmr::vector<regionid_t>>& /*predsCache*/);

  private:
  std::pmr::memory_resource* _tempPool;
  const RegionBuilder&       _regions;

  scc_t                               _sccs;
  std::pmr::unordered_set<regionid_t> _processed;
};

static SimpleNode_t copySimpleNode(SimpleNodeTemp_t const& src, std::pmr::memory_resource* pool) {
  SimpleNode_t dst(pool);

  dst.kind       = src.kind;
  dst.instrStart = src.instrStart;
  dst.instrEnd   = src.instrEnd;

  dst.children.reserve(src.children.size());
  std::ranges::transform(src.children, std::back_inserter(dst.children), [pool](const auto& child) { return copySimpleNode(child, pool); });

  dst.alternatives.reserve(src.alternatives.size());
  std::ranges::transform(src.alternatives, std::back_inserter(dst.alternatives), [pool](const auto& alt) { return copySimpleNode(alt, pool); });
  return dst;
}

SimpleNode_t transformStructuredCFG(std::pmr::memory_resource* allocPool, std::pmr::memory_resource* tempPool, RegionBuilder& regions) {
  CFGBuilder builder(tempPool, regions);
  auto       tempNodes = builder.build();
  return copySimpleNode(tempNodes, allocPool);
}

static RegionMask make_mask_from_set(std::pmr::set<regionid_t> const& s, size_t regionCount, std::pmr::memory_resource* pool) {
  RegionMask mask(regionCount, 0, pool);
  for (auto r: s)
    mask[static_cast<size_t>(r)] = 1;
  return mask;
}

static std::pmr::vector<std::pmr::vector<regionid_t>> build_predecessors_cache(const RegionBuilder& regions, std::pmr::memory_resource* pool) {
  size_t N = regions.getNumRegions();

  std::pmr::vector<std::pmr::vector<regionid_t>> preds(pool);
  preds.resize(N);

  for (regionid_t r = 0; r < static_cast<regionid_t>(N); ++r) {
    auto succs = regions.getSuccessorsIdx(r);
    for (auto s: succs) {
      preds[static_cast<size_t>(s)].push_back(r);
    }
  }
  return preds;
}

SimpleNodeTemp_t CFGBuilder::build() {
  _sccs = SCCBuild(_tempPool, _regions).calculate();
  dump(std::cout, _sccs);

  auto predsCache = build_predecessors_cache(_regions, _tempPool);

  size_t     N = _regions.getNumRegions();
  RegionMask allMask(N, 1, _tempPool);

  return restructureAcyclicIdx(0, allMask, predsCache);
}

SimpleNodeTemp_t make_basic_node(regionid_t rid, std::pmr::memory_resource* pool, const RegionBuilder& regions) {
  SimpleNodeTemp_t n(pool);
  n.kind            = SimpleNodeKind::Basic;
  auto [start, end] = regions.getRegion(rid);
  n.instrStart      = start;
  n.instrEnd        = end;
  return n;
}

SimpleNodeTemp_t make_sequence_node(SimpleNodeTemp_t::NodeContainer&& children, std::pmr::memory_resource* pool) {
  SimpleNodeTemp_t n(pool);
  n.kind     = SimpleNodeKind::Sequence;
  n.children = std::move(children);
  if (!n.children.empty()) {
    n.instrStart = n.children.front().instrStart;
    n.instrEnd   = n.children.back().instrEnd;
  }
  return n;
}

SimpleNodeTemp_t make_branch_node(SimpleNodeTemp_t::NodeContainer&& alts, regionid_t header, std::pmr::memory_resource* pool, const RegionBuilder& regions) {
  SimpleNodeTemp_t n(pool);
  n.kind            = SimpleNodeKind::Branch;
  n.alternatives    = std::move(alts);
  auto [start, end] = regions.getRegion(header);
  n.instrStart      = start;
  n.instrEnd        = end;
  return n;
}

SimpleNodeTemp_t make_loop_node(SimpleNodeTemp_t body, std::pmr::memory_resource* pool) {
  SimpleNodeTemp_t::NodeContainer children(pool);
  children.push_back(std::move(body));
  SimpleNodeTemp_t n(pool);
  n.kind     = SimpleNodeKind::Loop;
  n.children = std::move(children);
  if (!n.children.empty()) {
    n.instrStart = n.children.front().instrStart;
    n.instrEnd   = n.children.back().instrEnd;
  }
  return n;
}

bool CFGBuilder::isLinearIdx(regionid_t entry, const RegionMask& regionMask, const std::pmr::vector<std::pmr::vector<regionid_t>>& predsCache) const {
  regionid_t current = entry;
  size_t     N       = _regions.getNumRegions();
  // small local visited mask
  RegionMask visited(N, 0, _tempPool);
  while (current != RegionBuilder::NO_REGION && regionMask[static_cast<size_t>(current)]) {
    if (visited[static_cast<size_t>(current)]) return false;
    visited[static_cast<size_t>(current)] = 1;

    auto succs = _regions.getSuccessorsIdx(current);
    if (succs.size() > 1) return false;
    if (succs.empty()) break;

    regionid_t next = succs[0];
    if (!regionMask[static_cast<size_t>(next)]) break;

    // check preds count (use cache to avoid repeated verification via iteration)
    const auto& preds = predsCache[static_cast<size_t>(next)];
    if (preds.size() > 1) return false;

    current = next;
  }
  return true;
}

// restructureLoopBody: DFS that builds SimpleNodeTemp_t sequence with masks
SimpleNodeTemp_t CFGBuilder::restructureLoopBodyIdx(regionid_t entry, const RegionMask& sccMask,
                                                    const std::pmr::vector<std::pmr::vector<regionid_t>>& /*predsCache*/) {

  size_t     N = _regions.getNumRegions();
  RegionMask visited(N, 0, _tempPool);

  SimpleNodeTemp_t::NodeContainer bodyNodes(_tempPool);

  struct Frame {
    regionid_t node;
    int        state; /* 0=enter, 1=after children */
  };

  std::vector<Frame> stack;
  stack.reserve(64);
  stack.push_back({entry, 0});

  while (!stack.empty()) {
    Frame f = stack.back();
    stack.pop_back();

    regionid_t current = f.node;
    if (visited[static_cast<size_t>(current)] || !sccMask[static_cast<size_t>(current)]) continue;
    visited[static_cast<size_t>(current)] = 1;

    auto succs = _regions.getSuccessorsIdx(current);

    // compute successors that are inside the SCC (dense check)
    std::pmr::vector<regionid_t> sccSuccs(_tempPool);
    std::pmr::vector<regionid_t> outsideSuccs(_tempPool);
    for (auto s: succs) {
      if (sccMask[static_cast<size_t>(s)])
        sccSuccs.push_back(s);
      else
        outsideSuccs.push_back(s);
    }

    // If node has both in-SCC and out-of-SCC successors -> this is a branching header:
    // emit the current basic node followed by a Branch with alts for in-SCC (loop continuation)
    // and out-of-SCC (loop exits / continuation).
    if (!sccSuccs.empty() && !outsideSuccs.empty()) {
      // emit header basic
      bodyNodes.push_back(make_basic_node(current, _tempPool, _regions));

      // build alternatives
      SimpleNodeTemp_t::NodeContainer alts(_tempPool);

      // in-SCC successors => continue the loop body (recurse / continue traversal)
      for (auto succ: sccSuccs) {
        if (visited[static_cast<size_t>(succ)]) {
          // already visited: conservative basic node alternative
          alts.push_back(make_basic_node(succ, _tempPool, _regions));
        } else {
          // Continue building the loop body from this successor
          // Mark visited now so we don't re-enter
          visited[static_cast<size_t>(succ)] = 1;
          alts.push_back(restructureLoopBodyIdx(succ, sccMask, /*predsCache*/ std::pmr::vector<std::pmr::vector<regionid_t>>()));
          // Note: passing empty predsCache because restructureLoopBodyIdx doesn't use it;
          // we only use its local sccMask and regions.
        }
      }

      // out-of-SCC successors => exits: create acyclic subtrees for those
      for (auto succ: outsideSuccs) {
        RegionMask rem(_regions.getNumRegions(), 0, _tempPool);
        if (succ != RegionBuilder::NO_REGION) rem[static_cast<size_t>(succ)] = 1;
        alts.push_back(restructureAcyclicIdx(succ, rem, /*predsCache*/ std::pmr::vector<std::pmr::vector<regionid_t>>()));
      }

      if (!alts.empty()) {
        bodyNodes.push_back(make_branch_node(std::move(alts), current, _tempPool, _regions));
      }
      continue;
    }

    if (sccSuccs.empty()) {
      // leaf
      bodyNodes.push_back(make_basic_node(current, _tempPool, _regions));
      continue;
    }

    if (sccSuccs.size() == 1) {
      // linear: emit current and push successor for traversal
      bodyNodes.push_back(make_basic_node(current, _tempPool, _regions));
      // push successor only if not already visited
      if (!visited[static_cast<size_t>(sccSuccs[0])]) stack.push_back({sccSuccs[0], 0});
      continue;
    }

    // Branch inside the loop with *only* in-SCC successors
    bodyNodes.push_back(make_basic_node(current, _tempPool, _regions));
    SimpleNodeTemp_t::NodeContainer alts(_tempPool);

    for (auto succ: sccSuccs) {
      if (visited[static_cast<size_t>(succ)]) continue;
      visited[static_cast<size_t>(succ)] = 1; // mark so we don't re-enter
      // each alternative: just a basic node (conservative)
      alts.push_back(make_basic_node(succ, _tempPool, _regions));
    }
    if (!alts.empty()) {
      bodyNodes.push_back(make_branch_node(std::move(alts), current, _tempPool, _regions));
    }
  }

  return make_sequence_node(std::move(bodyNodes), _tempPool);
}

SimpleNodeTemp_t CFGBuilder::restructureLoopIdx(const std::pmr::set<regionid_t>& scc, regionid_t entry,
                                                const std::pmr::vector<std::pmr::vector<regionid_t>>& predsCache) {
  // Mark processed
  for (auto r: scc)
    _processed.insert(r);

  // prepare masks
  auto sccMask = make_mask_from_set(scc, _regions.getNumRegions(), _tempPool);

  // Find successors of the header entry and classify them as in-SCC vs out-of-SCC.
  auto                         succs = _regions.getSuccessorsIdx(entry);
  std::pmr::vector<regionid_t> inSuccs(_tempPool);
  std::pmr::vector<regionid_t> outSuccs(_tempPool);

  for (auto s: succs) {
    if (sccMask[static_cast<size_t>(s)])
      inSuccs.push_back(s);
    else
      outSuccs.push_back(s);
  }

  // Conservative: if no in-SCC successor found, treat the loop body as empty.
  SimpleNodeTemp_t loopNode(_tempPool);
  if (!inSuccs.empty()) {
    // Build a body starting from the first in-SCC successor.
    // We conservatively handle multiple inSuccs by creating a sequence or branch
    // starting at the first inSucc (restructureLoopBodyIdx will explore other in-SCC nodes).
    regionid_t       bodyEntry = inSuccs[0];
    SimpleNodeTemp_t body      = restructureLoopBodyIdx(bodyEntry, sccMask, predsCache);
    loopNode                   = make_loop_node(std::move(body), _tempPool);
  } else {
    // empty loop body (degenerate)
    SimpleNodeTemp_t::NodeContainer emptyChildren(_tempPool);
    SimpleNodeTemp_t                emptyBody(_tempPool);
    emptyBody.kind     = SimpleNodeKind::Sequence;
    emptyBody.children = std::move(emptyChildren);
    loopNode           = make_loop_node(std::move(emptyBody), _tempPool);
  }

  // Build branch alternatives: first alt is the loop (continue).
  SimpleNodeTemp_t::NodeContainer alts(_tempPool);
  alts.push_back(std::move(loopNode));

  // If there are out-of-SCC successors, build the continuation alternative(s).
  if (!outSuccs.empty()) {
    // Build a remaining mask = regionMask \ scc
    RegionMask remainingMask = make_mask_from_set(scc, _regions.getNumRegions(), _tempPool);
    // flip mask to represent remaining: start from all 0 then set outs
    // But easiest: create remaining as the parent's mask with scc bits cleared is expected.
    // We'll build a mask with only the outSuccs set (conservative)
    RegionMask contMask(_regions.getNumRegions(), 0, _tempPool);
    for (auto os: outSuccs) {
      if (os != RegionBuilder::NO_REGION) contMask[static_cast<size_t>(os)] = 1;
    }

    // If there are multiple outSuccs, create a sequence/branch for each; choose conservative approach:
    // build a single continuation subtree via restructureAcyclicIdx on the first out succ,
    // but more accurate behavior would create alternatives for each out succ.
    regionid_t contEntry = outSuccs[0];
    alts.push_back(restructureAcyclicIdx(contEntry, contMask, predsCache));
  }

  // Build final structure: header basic, then branch with loop vs continuation
  SimpleNodeTemp_t::NodeContainer seq(_tempPool);
  seq.push_back(make_basic_node(entry, _tempPool, _regions));
  seq.push_back(make_branch_node(std::move(alts), entry, _tempPool, _regions));

  return make_sequence_node(std::move(seq), _tempPool);
}

SimpleNodeTemp_t CFGBuilder::restructureAcyclicIdx(regionid_t entry, const RegionMask& regionMask,
                                                   const std::pmr::vector<std::pmr::vector<regionid_t>>& predsCache) {
  // restructureAcyclic: returns a SimpleNodeTemp_t representing the structured subtree for `entry`
  // regionMask: dense mask of the 'region' set (which nodes are in the current region)

  // empty region => basic (degenerate)
  bool any = false;
  for (size_t i = 0; i < regionMask.size(); ++i) {
    if (regionMask[i]) {
      any = true;
      break;
    }
  }
  if (!any) return make_basic_node(RegionBuilder::NO_REGION, _tempPool, _regions);

  if (_processed.find(entry) != _processed.end()) {
    return make_basic_node(entry, _tempPool, _regions);
  }

  // Check if entry is part of a loop(s) found earlier
  for (const auto& scc: _sccs) {
    if (scc.find(entry) != scc.end() && _processed.find(entry) == _processed.end()) {
      // build loop node (now responsible for continuation as well)
      return restructureLoopIdx(scc, entry, predsCache);
    }
  }

  // linear fast-path
  if (isLinearIdx(entry, regionMask, predsCache)) {
    SimpleNodeTemp_t::NodeContainer nodes(_tempPool);
    regionid_t                      current = entry;
    RegionMask                      visited(regionMask.size(), 0, _tempPool);

    while (current != RegionBuilder::NO_REGION && regionMask[static_cast<size_t>(current)] && !visited[static_cast<size_t>(current)]) {
      visited[static_cast<size_t>(current)] = 1;
      nodes.push_back(make_basic_node(current, _tempPool, _regions));

      auto succs = _regions.getSuccessorsIdx(current);
      if (succs.empty() || succs.size() > 1) break;

      regionid_t next = succs[0];
      if (visited[static_cast<size_t>(next)]) break;

      const auto& preds = predsCache[static_cast<size_t>(next)];
      if (preds.size() > 1) break;

      current = next;
    }
    return make_sequence_node(std::move(nodes), _tempPool);
  }

  // branching
  auto succs = _regions.getSuccessorsIdx(entry);
  if (succs.empty()) return make_basic_node(entry, _tempPool, _regions);

  if (succs.size() == 1) {
    auto next = succs[0];
    if (next >= 0 && regionMask[static_cast<size_t>(next)] && _processed.find(next) == _processed.end()) {
      SimpleNodeTemp_t::NodeContainer seq(_tempPool);
      seq.push_back(make_basic_node(entry, _tempPool, _regions));

      RegionMask remaining(regionMask, _tempPool);
      remaining[static_cast<size_t>(entry)] = 0;
      seq.push_back(restructureAcyclicIdx(next, remaining, predsCache));
      return make_sequence_node(std::move(seq), _tempPool);
    }
    return make_basic_node(entry, _tempPool, _regions);
  }

  // Multiple successors -> branch node
  SimpleNodeTemp_t::NodeContainer sequence(_tempPool);
  sequence.push_back(make_basic_node(entry, _tempPool, _regions));

  SimpleNodeTemp_t::NodeContainer branchAlts(_tempPool);
  for (auto s: succs) {
    if (s >= 0 && regionMask[static_cast<size_t>(s)] && _processed.find(s) == _processed.end()) {
      RegionMask brRegion(regionMask.size(), 0, _tempPool);
      brRegion[static_cast<size_t>(s)] = 1;
      branchAlts.push_back(restructureAcyclicIdx(s, brRegion, predsCache));
    }
  }

  if (!branchAlts.empty()) {
    sequence.push_back(make_branch_node(std::move(branchAlts), entry, _tempPool, _regions));
    return make_sequence_node(std::move(sequence), _tempPool);
  }

  return make_basic_node(entry, _tempPool, _regions);
}

// // DUMP

void dumpAST(std::ostream& os, auto const& node, int depth = 0) {
  std::string ind       = indent(depth);
  std::string connector = depth > 0 ? "-- " : "";

  os << ind << connector << nodeKindName(node.kind) << " [" << node.instrStart << ", " << node.instrEnd << ")";

  uint32_t instrCount = node.instrEnd - node.instrStart;
  if (instrCount > 0) os << " (" << instrCount << " instr)";

  os << "\n";

  switch (node.kind) {
    case SimpleNodeKind::Branch: {
      uint32_t i = 0;
      for (auto const& node: node.alternatives) {
        os << indent(depth + 1) << "- Alt[" << i << "]:\n";
        dumpAST(os, node, depth + 2);
        ++i;
      }
      break;
    }

    case SimpleNodeKind::Loop: {
      if (!node.children.empty()) {
        os << indent(depth + 1) << "- Body:\n";
        dumpAST(os, node.children.front(), depth + 2);
      }
      break;
    }

    case SimpleNodeKind::Sequence: {
      uint32_t i = 0;
      for (auto const& node: node.children) {
        os << indent(depth + 1) << "- Step[" << i << "]:\n";
        dumpAST(os, node, depth + 2);
        ++i;
      }
      break;
    }

    case SimpleNodeKind::Basic:
    default: break;
  }
}

void dumpCode(std::ostream& os, auto const& node, InstCore const* instructions, int depth = 0) {
  std::string ind = indent(depth);
  switch (node.kind) {
    case SimpleNodeKind::Branch: {
      uint32_t i = 0;
      for (auto const& node: node.alternatives) {
        os << ind << "- Branch[" << i << "]:\n";
        dumpCode(os, node, instructions, depth + 2);
        ++i;
      }
      break;
    }

    case SimpleNodeKind::Loop: {
      if (!node.children.empty()) {
        os << ind << "- Loop:\n";
        dumpCode(os, node.children.front(), instructions, depth + 2);
      }
      break;
    }

    case SimpleNodeKind::Sequence: {
      uint32_t i = 0;
      for (auto const& node: node.children) {
        dumpCode(os, node, instructions, depth + 2);
        ++i;
      }
      break;
    }

    case SimpleNodeKind::Basic: {
      for (auto n = node.instrStart; n < node.instrEnd; ++n) {
        os << ind;
        ir::debug::getDebug(std::cout, instructions[n]);
      }
    } break;
    default: break;
  }
}

void dump(std::ostream& os, SimpleNode_t const* node) {
  dumpAST(os, *node, 0);
}

void dump(std::ostream& os, SimpleNode_t const* node, InstCore const* instructions) {
  // dumpCode(os, *node, instructions, 0);
}
} // namespace compiler::ir