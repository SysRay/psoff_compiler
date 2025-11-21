#include "analysis/dom.h"
#include "analysis/scc.h"
#include "cfg/cfg.h"
#include "include/checkpoint_resource.h"
#include "include/common.h"
#include "ir/debug_strings.h"
#include "logging.h"
#include "transform.h"

#include <memory_resource>
#include <queue>
#include <stack>
#include <stdexcept>

namespace compiler::transform {

struct GraphAdapter {
  cfg::ControlFlow& g;

  auto getSuccessors(uint32_t idx) const {
    return g.getSuccessors(cfg::rvsdg::nodeid_t {idx}) | std::views::transform([](auto id) { return id.value; });
  }

  auto getPredecessors(uint32_t idx) const {
    return g.getPredecessors(cfg::rvsdg::nodeid_t {idx}) | std::views::transform([](auto id) { return id.value; });
  }

  size_t size() const { return g.nodes()->blocksCount(); }

  GraphAdapter(cfg::ControlFlow& g): g(g) {}
};

static void collapseCycles(util::checkpoint_resource& checkpoint_resource, cfg::ControlFlow& cfg, std::pmr::vector<cfg::rvsdg::regionid_t>& tasks,
                           cfg::rvsdg::regionid_t regionId) {
  // 1. Find SCCs
  // 2. find entries, exits and back edges
  // 3 .collaps into loop node

  auto region = cfg.nodes()->getRegion(regionId);
  if (region->nodes.empty()) return;

  auto checkpoint = checkpoint_resource.checkpoint();

  GraphAdapter adapter(cfg);
  auto const   sccs = compiler::analysis::SCCBuilder<GraphAdapter>(&checkpoint_resource, adapter).calculate(region->nodes.front());

  for (auto const& scc: sccs.get()) {
    auto checkpoint = checkpoint_resource.checkpoint();

    auto const sccEdges = compiler::analysis::classifySCC(&checkpoint_resource, adapter, scc);
    if (sccEdges.entryEdges.empty()) {
      throw std::runtime_error("scc with no entry");
    }

    auto const loopId = cfg.createThetaNode();
    auto       loop   = cfg.nodes()->accessNode<cfg::rvsdg::ThetaNode>(loopId);

    auto loopRegions = cfg.nodes()->accessRegion(loop->body);
    loopRegions->nodes.reserve(1 + scc.size()); //  exit + sccNodes

    // Theta node: First result is predicate for exit or continue (latch)
    // Body region contains single entry and single exit

    // // Restructure entries to one entry
    cfg::rvsdg::nodeid_t headerId = {};
    if (sccEdges.entryEdges.size() > 1) {
      // First value passed to Theta Node is the branch selector value (q-value)
      // redirect all edges to loop and set the branch value
      for (auto const& edge: sccEdges.entryEdges) {
        cfg.redirectEdge(cfg::rvsdg::nodeid_t(edge.first), cfg::rvsdg::nodeid_t(edge.second), loopId);
        // todo branch value
      }

      // todo create demux
      headerId = cfg::rvsdg::nodeid_t(sccEdges.entryEdges[0].second);
      cfg.nodes()->insertNodeToRegion(loopId, headerId);
    } else {
      headerId = cfg::rvsdg::nodeid_t(sccEdges.entryEdges[0].second);
      cfg.redirectEdge(cfg::rvsdg::nodeid_t(sccEdges.entryEdges[0].first), headerId, loopId);
      cfg.nodes()->insertNodeToRegion(loopId, headerId);
    }

    cfg.nodes()->moveNodeToRegion(headerId, loopRegions->id);

    // // Restructure exits to one exit

    // Special case:       // already tail based loop -> use node as latch
    if (sccEdges.exitEdges.size() == 1 && sccEdges.exitEdges[0].first == sccEdges.backEdges[0].first) {
      cfg::rvsdg::nodeid_t exitLatchId = cfg::rvsdg::nodeid_t(sccEdges.exitEdges[0].first);

      cfg.removeEdge(exitLatchId, cfg::rvsdg::nodeid_t(sccEdges.exitEdges[0].second));
      cfg.removeEdge(exitLatchId, cfg::rvsdg::nodeid_t(sccEdges.backEdges[0].second));
      // todo branch value

      cfg.addEdge(loopId, cfg::rvsdg::nodeid_t(sccEdges.exitEdges[0].second));

      // Add nodes to region
      for (auto id: std::ranges::reverse_view(scc)) {
        if (id == headerId || id == exitLatchId) continue;
        cfg.nodes()->moveNodeToRegion(cfg::rvsdg::nodeid_t(id), loopRegions->id);
      }

      cfg.nodes()->moveNodeToRegion(exitLatchId, loopRegions->id);
      tasks.push_back(loopRegions->id);
      return;
    }
    // - Special case

    // First value return from Theta Node is the loop predicate (r-value)
    // (Optional) Second value is for entry selection (q-value)
    // redirect all from loop to exits
    cfg::rvsdg::nodeid_t exitLatchId = cfg.createSimpleNode();
    for (auto const& [from, to]: sccEdges.exitEdges) {
      cfg.redirectEdge(cfg::rvsdg::nodeid_t(from), cfg::rvsdg::nodeid_t(to), exitLatchId);
      cfg.addEdge(loopId, cfg::rvsdg::nodeid_t(to));
      // todo branch value
    }

    if (sccEdges.backEdges.size() > 0) {
      for (auto const& edge: sccEdges.backEdges) {
        cfg.redirectEdge(cfg::rvsdg::nodeid_t(edge.first), cfg::rvsdg::nodeid_t(edge.second), exitLatchId);
        // todo branch value
      }
    }

    // todo demux for exits

    // // Add nodes to region
    // Note scc lists reversed nodes -> reverse to get previous order
    for (auto id: std::ranges::reverse_view(scc)) {
      if (id == headerId.value) continue;
      cfg.nodes()->moveNodeToRegion(cfg::rvsdg::nodeid_t(id), loopRegions->id);
    }

    cfg.nodes()->moveNodeToRegion(exitLatchId, loopRegions->id);

    tasks.push_back(loopRegions->id); // Collapse sub region
  }
}

static void collapseBranches(util::checkpoint_resource& checkpoint_resource, cfg::ControlFlow& cfg, std::pmr::vector<cfg::rvsdg::regionid_t>& tasks,
                             cfg::rvsdg::regionid_t regionId) {
  //  1. Find conditional branch
  //  2. Get merge point from post dominator
  //  3. todo (exits, tail resturcturing)

  auto region = cfg.nodes()->getRegion(regionId);
  if (region->nodes.empty()) return;

  auto checkpoint = checkpoint_resource.checkpoint();

  compiler::analysis::DominatorTreeDense<GraphAdapter> dom({&checkpoint_resource});

  auto headerId = region->nodes.front();
  while (true) {
    auto const& succs = cfg.getSuccessors(headerId);
    if (succs.size() == 0) break;
    if (succs.size() == 1) {
      headerId = succs[0];
      // todo merge
      continue;
    }
    assert(succs.size() == 2); // todo handle demux?

    auto const condId = cfg.createGammaNode();
    auto       cond   = cfg.nodes()->accessNode<cfg::rvsdg::GammaNode>(condId);
    cond->branches.reserve(succs.size());
    cfg.nodes()->insertNodeToRegion(condId, cfg::rvsdg::nodeid_t(succs[0]));

    // // Find branches
    dom.calculate(GraphAdapter(cfg), region->nodes.back().value); // build once on demand

    std::pmr::vector<cfg::rvsdg::nodeid_t> continuationPoints(&checkpoint_resource);
    std::pmr::vector<cfg::rvsdg::nodeid_t> work(&checkpoint_resource);
    work.reserve(64);
    for (uint8_t arc = 0; arc < succs.size(); ++arc) {
      work.push_back(succs[arc]);
      std::pmr::vector<std::pair<cfg::rvsdg::nodeid_t, cfg::rvsdg::nodeid_t>> exits(&checkpoint_resource);

      auto regionId = cond->branches.emplace_back(cfg.nodes()->createRegion());
      tasks.push_back(regionId); // Collapse sub region

      while (!work.empty()) {
        auto const v = work.back();
        work.pop_back();

        auto const& items = cfg.getSuccessors(v);
        cfg.nodes()->moveNodeToRegion(v, regionId);

        for (auto to: items) {
          auto idom = dom.get_idom(to);
          if (idom && *idom == v) {
            work.push_back(to);
          } else {
            // auto const cp =std::distance(continuationPoints.begin(), addUnique(continuationPoints, to));
            exits.emplace_back(v, to);
            addUnique(continuationPoints, to);
          }
        }
      }

      // handle exits
      if (exits.size() == 1) {
        auto const [from, to] = exits.front();
        cfg.removeEdge(from, to);
      } else if (exits.size() > 1) {
        // Multiple: create dummy node as new exit
        auto dummyId = cfg.createSimpleNode();
        cfg.nodes()->moveNodeToRegion(dummyId, regionId);

        for (auto const [from, to]: exits) {
          cfg.redirectEdge(from, to, dummyId);
        }
      }
    }

    // // Find tail

    cfg::rvsdg::nodeid_t tailId = {};
    if (continuationPoints.size() == 1) { // Special case:  One cp -> use as tail
      tailId = continuationPoints.front();
    } else {
      throw std::runtime_error("todo multiple cp");
    }

    cfg.accessSuccessors(headerId).clear();
    cfg.addEdge(headerId, condId);

    cfg.addEdge(condId, tailId);

    headerId = tailId; // Continue;
  }
}

void restructureCfg(util::checkpoint_resource& checkpoint_resource, cfg::ControlFlow& cfg) {
  auto checkpoint = checkpoint_resource.checkpoint();

  std::pmr::vector<cfg::rvsdg::regionid_t> tasks;
  tasks.push_back(cfg.nodes()->getMainFunction()->body);

  while (!tasks.empty()) {
    auto const regionId = tasks.back();
    tasks.pop_back();

    collapseCycles(checkpoint_resource, cfg, tasks, regionId);
    collapseBranches(checkpoint_resource, cfg, tasks, regionId);
  }
}
} // namespace compiler::transform