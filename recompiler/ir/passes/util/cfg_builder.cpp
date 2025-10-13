#include "cfg_builder.h"

#include "../../debug_strings.h"

#include <algorithm>
#include <ostream>
#include <set>
#include <stack>
#include <unordered_set>

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
struct SCCData {
  struct TarjanState {
    int  index   = -1;
    int  lowlink = -1;
    bool onStack = false;
  };

  std::pmr::vector<TarjanState>        state;
  std::pmr::vector<regionid_t>         stack;
  int                                  tarjanIndex = 0;
  std::pmr::monotonic_buffer_resource& pool;

  SCCData(std::pmr::monotonic_buffer_resource& pool_, size_t regionCount): state(regionCount, TarjanState {}, &pool_), stack(&pool_), pool(pool_) {}
};

using RegionMask = std::pmr::vector<char>;

class CFGBuilder {
  public:
  explicit CFGBuilder(std::pmr::monotonic_buffer_resource& allocPool, std::pmr::monotonic_buffer_resource& tempPool, RegionBuilder& regions)
      : _allocPool(allocPool), _tempPool(tempPool), _regions(regions), _sccs(&_tempPool) {}

  SimpleNode build();

  private:
  void       strongConnect(SCCData& data, int32_t v);
  bool       isLinearIdx(regionid_t entry, const RegionMask& regionMask, const std::pmr::vector<std::pmr::vector<regionid_t>>& predsCache) const;
  SimpleNode restructureLoopIdx(const std::pmr::set<regionid_t>& scc, regionid_t entry, std::pmr::monotonic_buffer_resource& pool,
                                const std::pmr::vector<std::pmr::vector<regionid_t>>& predsCache);
  SimpleNode restructureAcyclicIdx(regionid_t entry, const RegionMask& regionMask, const std::pmr::vector<std::pmr::vector<regionid_t>>& predsCache,
                                   std::pmr::monotonic_buffer_resource& pool);
  SimpleNode restructureLoopBodyIdx(regionid_t entry, const RegionMask& sccMask, std::pmr::monotonic_buffer_resource& pool,
                                    const std::pmr::vector<std::pmr::vector<regionid_t>>& /*predsCache*/);

  private:
  std::pmr::monotonic_buffer_resource& _allocPool;
  std::pmr::monotonic_buffer_resource& _tempPool;
  const RegionBuilder&                 _regions;

  std::pmr::vector<std::pmr::set<regionid_t>> _sccs;
  std::pmr::unordered_set<regionid_t>         _processed;
};

SimpleNode transformStructuredCFG(std::pmr::monotonic_buffer_resource& allocPool, std::pmr::monotonic_buffer_resource& tempPool, RegionBuilder& regions) {
  CFGBuilder builder(allocPool, tempPool, regions);
  return builder.build();
}

static RegionMask make_mask_from_set(std::pmr::set<regionid_t> const& s, size_t regionCount, std::pmr::monotonic_buffer_resource& pool) {
  RegionMask mask(regionCount, 0, &pool);
  for (auto r: s)
    mask[static_cast<size_t>(r)] = 1;
  return mask;
}

static std::pmr::vector<std::pmr::vector<regionid_t>> build_predecessors_cache(const RegionBuilder& regions, std::pmr::monotonic_buffer_resource& pool) {
  size_t N = regions.getNumRegions();

  std::pmr::vector<std::pmr::vector<regionid_t>> preds(&pool);
  preds.resize(N);

  for (regionid_t r = 0; r < static_cast<regionid_t>(N); ++r) {
    auto succs = regions.getSuccessorsIdx(r);
    for (auto s: succs) {
      preds[static_cast<size_t>(s)].push_back(r);
    }
  }
  return preds;
}

SimpleNode CFGBuilder::build() {
  SCCData data(_tempPool, _regions.getNumRegions());
  strongConnect(data, 0);

  auto predsCache = build_predecessors_cache(_regions, _tempPool);

  size_t     N = _regions.getNumRegions();
  RegionMask allMask(N, 1, &_tempPool);

  return restructureAcyclicIdx(0, allMask, predsCache, _tempPool);
}

void CFGBuilder::strongConnect(SCCData& data, int32_t v) {
  auto& state   = data.state[v];
  state.index   = data.tarjanIndex;
  state.lowlink = data.tarjanIndex;
  data.tarjanIndex++;
  data.stack.push_back(v);
  state.onStack = true;

  auto succs = _regions.getSuccessorsIdx(v);
  for (auto w: succs) {
    auto& item = data.state[w];
    if (item.index == -1) {
      strongConnect(data, w);
      state.lowlink = std::min(state.lowlink, item.lowlink);
    } else if (item.onStack) {
      state.lowlink = std::min(state.lowlink, item.index);
    }
  }

  // Root of SCC
  if (state.lowlink == state.index) {
    std::pmr::set<regionid_t> scc {&_tempPool};
    regionid_t                w;
    do {
      w = data.stack.back();
      data.stack.pop_back();
      data.state[w].onStack = false;
      scc.insert(w);
    } while (w != v);

    if (scc.size() > 1) {
      _sccs.push_back(std::move(scc));
    }
  }
}

SimpleNode make_basic_node(regionid_t rid, std::pmr::monotonic_buffer_resource& pool, const RegionBuilder& regions) {
  SimpleNode n(pool);
  n.kind            = SimpleNode::Kind::Basic;
  n.rid             = rid;
  auto [start, end] = regions.getRegion(rid);
  n.instrStart      = start;
  n.instrEnd        = end;
  return n;
}

SimpleNode make_sequence_node(std::pmr::vector<SimpleNode>&& children, std::pmr::monotonic_buffer_resource& pool) {
  SimpleNode n(pool);
  n.kind     = SimpleNode::Kind::Sequence;
  n.children = std::move(children);
  if (!n.children.empty()) {
    n.instrStart = n.children.front().instrStart;
    n.instrEnd   = n.children.back().instrEnd;
  }
  return n;
}

SimpleNode make_branch_node(std::pmr::vector<SimpleNode>&& alts, regionid_t header, std::pmr::monotonic_buffer_resource& pool, const RegionBuilder& regions) {
  SimpleNode n(pool);
  n.kind            = SimpleNode::Kind::Branch;
  n.alternatives    = std::move(alts);
  auto [start, end] = regions.getRegion(header);
  n.instrStart      = start;
  n.instrEnd        = end;
  return n;
}

SimpleNode make_loop_node(SimpleNode body, std::pmr::monotonic_buffer_resource& pool) {
  std::pmr::vector<SimpleNode> children(&pool);
  children.push_back(std::move(body));
  SimpleNode n(pool);
  n.kind     = SimpleNode::Kind::Loop;
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
  RegionMask visited(N, 0, &_tempPool);
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

// restructureLoopBody: DFS that builds SimpleNode sequence with masks
SimpleNode CFGBuilder::restructureLoopBodyIdx(regionid_t entry, const RegionMask& sccMask, std::pmr::monotonic_buffer_resource& pool,
                                              const std::pmr::vector<std::pmr::vector<regionid_t>>& /*predsCache*/) {

  size_t     N = _regions.getNumRegions();
  RegionMask visited(N, 0, &pool);

  std::pmr::vector<SimpleNode> bodyNodes(&_allocPool);
  bodyNodes.reserve(16);

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
    std::pmr::vector<regionid_t> sccSuccs(&_tempPool);
    for (auto s: succs) {
      if (sccMask[static_cast<size_t>(s)]) sccSuccs.push_back(s);
    }

    if (sccSuccs.empty()) {
      // leaf
      bodyNodes.push_back(make_basic_node(current, _allocPool, _regions));
      continue;
    }

    if (sccSuccs.size() == 1) {
      // linear: emit current and push successor for traversal
      bodyNodes.push_back(make_basic_node(current, _allocPool, _regions));
      stack.push_back({sccSuccs[0], 0});
      continue;
    }

    // Branch inside the loop: create basic header + branch node with immediate-successor alternatives
    bodyNodes.push_back(make_basic_node(current, _allocPool, _regions));
    std::pmr::vector<SimpleNode> alts(&_allocPool);
    alts.reserve(sccSuccs.size());

    for (auto succ: sccSuccs) {
      if (visited[static_cast<size_t>(succ)]) continue;
      visited[static_cast<size_t>(succ)] = 1; // mark so we don't re-enter
      // each alternative: just a basic node (conservative)
      alts.push_back(make_basic_node(succ, _allocPool, _regions));
    }
    if (!alts.empty()) {
      bodyNodes.push_back(make_branch_node(std::move(alts), current, _allocPool, _regions));
    }
  }

  return make_sequence_node(std::move(bodyNodes), _allocPool);
}

SimpleNode CFGBuilder::restructureLoopIdx(const std::pmr::set<regionid_t>& scc, regionid_t entry, std::pmr::monotonic_buffer_resource& pool,
                                          const std::pmr::vector<std::pmr::vector<regionid_t>>& predsCache) {
  // restructureLoop: mark processed & wrap body into a Loop SimpleNode
  // Mark processed
  for (auto r: scc)
    _processed.insert(r);

  auto       sccMask = make_mask_from_set(scc, _regions.getNumRegions(), pool);
  SimpleNode body    = restructureLoopBodyIdx(entry, sccMask, pool, predsCache);
  return make_loop_node(std::move(body), _allocPool);
}

SimpleNode CFGBuilder::restructureAcyclicIdx(regionid_t entry, const RegionMask& regionMask, const std::pmr::vector<std::pmr::vector<regionid_t>>& predsCache,
                                             std::pmr::monotonic_buffer_resource& pool) {
  // restructureAcyclic: returns a SimpleNode representing the structured subtree for `entry`
  // regionMask: dense mask of the 'region' set (which nodes are in the current region)

  // empty region => basic (degenerate)
  bool any = false;
  for (size_t i = 0; i < regionMask.size(); ++i) {
    if (regionMask[i]) {
      any = true;
      break;
    }
  }
  if (!any) return make_basic_node(RegionBuilder::NO_REGION, _allocPool, _regions);

  if (_processed.find(entry) != _processed.end()) {
    return make_basic_node(entry, _allocPool, _regions);
  }

  // Check if entry is part of a loop(s) found earlier
  for (const auto& scc: _sccs) {
    if (scc.find(entry) != scc.end() && _processed.find(entry) == _processed.end()) {
      // build loop node
      SimpleNode loopNode = restructureLoopIdx(scc, entry, pool, predsCache);

      // remaining = region \ scc
      RegionMask remainingMask(regionMask, &pool);
      for (auto r: scc)
        remainingMask[static_cast<size_t>(r)] = 0;

      // If nothing left, return loop node
      bool hasRemaining = false;
      for (auto b: remainingMask) {
        if (b) {
          hasRemaining = true;
          break;
        }
      }
      if (!hasRemaining) return loopNode;

      // find next region after loop: look at successors of nodes in scc for a node in remaining
      regionid_t next = RegionBuilder::NO_REGION;
      for (auto r: scc) {
        auto succs = _regions.getSuccessorsIdx(r);
        for (auto s: succs) {
          if (remainingMask[static_cast<size_t>(s)]) {
            next = s;
            break;
          }
        }
        if (next != RegionBuilder::NO_REGION) break;
      }

      if (next != RegionBuilder::NO_REGION) {
        // sequence [loopNode, restructureAcyclic(next, remaining)]
        std::pmr::vector<SimpleNode> seq(&_allocPool);
        seq.reserve(2);
        seq.push_back(std::move(loopNode));
        seq.push_back(restructureAcyclicIdx(next, remainingMask, predsCache, pool));
        return make_sequence_node(std::move(seq), _allocPool);
      }

      return loopNode;
    }
  }

  // linear fast-path
  if (isLinearIdx(entry, regionMask, predsCache)) {
    std::pmr::vector<SimpleNode> nodes(&_allocPool);
    nodes.reserve(8);
    regionid_t current = entry;
    RegionMask visited(regionMask.size(), 0, &pool);
    while (current != RegionBuilder::NO_REGION && regionMask[static_cast<size_t>(current)] && !visited[static_cast<size_t>(current)]) {
      visited[static_cast<size_t>(current)] = 1;
      nodes.push_back(make_basic_node(current, _allocPool, _regions));

      auto succs = _regions.getSuccessorsIdx(current);
      if (succs.empty() || succs.size() > 1) break;

      regionid_t next = succs[0];
      if (visited[static_cast<size_t>(next)]) break;

      const auto& preds = predsCache[static_cast<size_t>(next)];
      if (preds.size() > 1) break;

      current = next;
    }
    return make_sequence_node(std::move(nodes), _allocPool);
  }

  // branching
  auto succs = _regions.getSuccessorsIdx(entry);
  if (succs.empty()) return make_basic_node(entry, _allocPool, _regions);

  if (succs.size() == 1) {
    auto next = succs[0];
    if (next >= 0 && regionMask[static_cast<size_t>(next)] && _processed.find(next) == _processed.end()) {
      std::pmr::vector<SimpleNode> seq(&_allocPool);
      seq.reserve(2);
      seq.push_back(make_basic_node(entry, _allocPool, _regions));

      RegionMask remaining(regionMask, &pool);
      remaining[static_cast<size_t>(entry)] = 0;
      seq.push_back(restructureAcyclicIdx(next, remaining, predsCache, pool));
      return make_sequence_node(std::move(seq), _allocPool);
    }
    return make_basic_node(entry, _allocPool, _regions);
  }

  // Multiple successors -> branch node
  std::pmr::vector<SimpleNode> sequence(&_allocPool);
  sequence.reserve(2);
  sequence.push_back(make_basic_node(entry, _allocPool, _regions));

  std::pmr::vector<SimpleNode> branchAlts(&_allocPool);
  branchAlts.reserve(succs.size());
  for (auto s: succs) {
    if (s >= 0 && regionMask[static_cast<size_t>(s)] && _processed.find(s) == _processed.end()) {
      RegionMask brRegion(regionMask.size(), 0, &pool);
      brRegion[static_cast<size_t>(s)] = 1;
      branchAlts.push_back(restructureAcyclicIdx(s, brRegion, predsCache, pool));
    }
  }

  if (!branchAlts.empty()) {
    sequence.push_back(make_branch_node(std::move(branchAlts), entry, _allocPool, _regions));
    return make_sequence_node(std::move(sequence), _allocPool);
  }

  return make_basic_node(entry, _allocPool, _regions);
}

// // DUMP
static std::string indent(int level) {
  return std::string(level * 2, ' ');
}

static std::string nodeKindName(SimpleNode::Kind kind) {
  using K = SimpleNode::Kind;
  switch (kind) {
    case K::Basic: return "Basic";
    case K::Branch: return "Branch";
    case K::Loop: return "Loop";
    case K::Sequence: return "Sequence";
    default: return "Unknown";
  }
}

void dumpAST(std::ostream& os, const SimpleNode* node, int depth = 0) {
  if (!node) return;

  std::string ind       = indent(depth);
  std::string connector = depth > 0 ? "-- " : "";

  os << ind << connector << nodeKindName(node->kind) << " [" << node->instrStart << ", " << node->instrEnd << ")";

  uint32_t instrCount = node->instrEnd - node->instrStart;
  if (instrCount > 0) os << " (" << instrCount << " instr)";

  os << "\n";

  switch (node->kind) {
    case SimpleNode::Kind::Branch: {
      for (size_t i = 0; i < node->alternatives.size(); ++i) {
        os << indent(depth + 1) << "- Alt[" << i << "]:\n";
        dumpAST(os, &node->alternatives[i], depth + 2);
      }
      break;
    }

    case SimpleNode::Kind::Loop: {
      if (!node->children.empty()) {
        os << indent(depth + 1) << "- Body:\n";
        dumpAST(os, &node->children.front(), depth + 2);
      }
      break;
    }

    case SimpleNode::Kind::Sequence: {
      for (size_t i = 0; i < node->children.size(); ++i) {
        os << indent(depth + 1) << "- Step[" << i << "]:\n";
        dumpAST(os, &node->children[i], depth + 2);
      }
      break;
    }

    case SimpleNode::Kind::Basic:
    default: break;
  }
}

void dump(std::ostream& os, const SimpleNode* node) {
  dumpAST(os, node, 0);
}
} // namespace compiler::ir