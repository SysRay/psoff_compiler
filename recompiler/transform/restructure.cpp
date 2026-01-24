#include "analysis/dom.h"
#include "analysis/scc.h"
#include "cfg/cfg.h"
#include "include/checkpoint_resource.h"
#include "include/common.h"
#include "ir/debug_strings.h"
#include "ir/dialects/core/builder.h"
#include "logging.h"
#include "transform.h"

#include <memory_resource>
#include <queue>
#include <ranges>
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

  auto im = cfg.accessInstructions();
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

      for (uint32_t n = 0; n < sccEdges.entryEdges.size(); ++n) {
        auto const& edge = sccEdges.entryEdges[n];
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

    // // Special case: already tail based loop -> use node as latch
    // if (sccEdges.exitEdges.size() == 1 && sccEdges.exitEdges[0].first == sccEdges.backEdges[0].first) {
    //   cfg::rvsdg::nodeid_t exitLatchId = cfg::rvsdg::nodeid_t(sccEdges.exitEdges[0].first);

    //   cfg.removeEdge(exitLatchId, cfg::rvsdg::nodeid_t(sccEdges.exitEdges[0].second));
    //   cfg.removeEdge(exitLatchId, cfg::rvsdg::nodeid_t(sccEdges.backEdges[0].second));
    //   // todo branch value

    //   cfg.addEdge(loopId, cfg::rvsdg::nodeid_t(sccEdges.exitEdges[0].second));

    //   // Add nodes to region
    //   for (auto id: std::ranges::reverse_view(scc)) {
    //     if (id == headerId || id == exitLatchId) continue;
    //     cfg.nodes()->moveNodeToRegion(cfg::rvsdg::nodeid_t(id), loopRegions->id);
    //   }

    //   cfg.nodes()->moveNodeToRegion(exitLatchId, loopRegions->id);
    //   tasks.push_back(loopRegions->id);
    //   return;
    // }
    // // - Special case

    // First value returned from theta-node is loop predicate (r-value)
    // (Optional) Second value is for entry selection (q-value)
    // redirect everything to exit latch (Must be one output node)
    cfg::rvsdg::nodeid_t exitLatchId = cfg.createSimpleNode();
    cfg::rvsdg::nodeid_t backLatchId = cfg.createSimpleNode();

    if (sccEdges.backEdges.size() > 0) {
      for (auto const& [from, to]: sccEdges.backEdges) {
        cfg.redirectEdge(cfg::rvsdg::nodeid_t(from), cfg::rvsdg::nodeid_t(to), backLatchId);
        auto node = cfg.nodes()->accessNode<cfg::rvsdg::SimpleNode>(cfg::rvsdg::nodeid_t(backLatchId));

        auto predValue = cfg.create<ir::dialect::core::ConstantOp>(ir::dialect::OpDst(), ir::ConstantValue {.value_u64 = 1}, ir::OperandType::i1());
        node->instructions.push_back(predValue);

        auto inId = im->createInput(ir::OperandType::i1());
        node->outputs.push_back(inId);
        cfg.accessInstructions()->connect(inId, predValue);
      }
    }

    for (auto const& [from, to]: sccEdges.exitEdges) {
      cfg.redirectEdge(cfg::rvsdg::nodeid_t(from), cfg::rvsdg::nodeid_t(to), exitLatchId);
      cfg.addEdge(loopId, cfg::rvsdg::nodeid_t(to));

      auto node = cfg.nodes()->accessNode<cfg::rvsdg::SimpleNode>(cfg::rvsdg::nodeid_t(exitLatchId));

      auto predValue = cfg.create<ir::dialect::core::ConstantOp>(ir::dialect::OpDst(), ir::ConstantValue {.value_u64 = 0}, ir::OperandType::i1());
      node->instructions.push_back(predValue);

      auto inId = im->createInput(ir::OperandType::i1());
      node->outputs.push_back(inId);
      cfg.accessInstructions()->connect(inId, predValue);
    }

    // todo demux for exits

    // // Add nodes to region
    // Note scc lists reversed nodes -> reverse to get previous order
    for (auto id: std::ranges::reverse_view(scc)) {
      if (id == headerId.value) continue;
      cfg.nodes()->moveNodeToRegion(cfg::rvsdg::nodeid_t(id), loopRegions->id);
    }

    cfg::rvsdg::nodeid_t latchId = cfg.createSimpleNode(); // one exit node
    cfg.addEdge(backLatchId, latchId);
    cfg.addEdge(exitLatchId, latchId);

    cfg.nodes()->moveNodeToRegion(backLatchId, loopRegions->id);
    cfg.nodes()->moveNodeToRegion(exitLatchId, loopRegions->id);
    cfg.nodes()->moveNodeToRegion(latchId, loopRegions->id);

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

    auto const condId = cfg.createGammaNode();
    auto       cond   = cfg.nodes()->accessNode<cfg::rvsdg::GammaNode>(condId);
    cond->branches.reserve(succs.size());
    cfg.nodes()->insertNodeToRegion(condId, cfg::rvsdg::nodeid_t(succs[0]));

    // // Find branches
    dom.calculate(GraphAdapter(cfg), headerId); // build once on demand

    std::pmr::vector<std::pair<cfg::rvsdg::nodeid_t, std::pmr::vector<cfg::rvsdg::nodeid_t>>> continuationPoints(&checkpoint_resource);
    std::pmr::vector<cfg::rvsdg::nodeid_t>                                                    work(&checkpoint_resource);
    work.reserve(64);
    for (uint8_t arc = 0; arc < succs.size(); ++arc) {
      auto regionId = cond->branches.emplace_back(cfg.nodes()->createRegion());
      tasks.push_back(regionId);

      auto arcStartNode = succs[arc];

      if (cfg.getPredecessors(arcStartNode).size() > 1) {
        // Check if empty arc -> insert dummy node
        // each arc must have min 1 node!
        auto dummyId = cfg.createSimpleNode();
        cfg.nodes()->moveNodeToRegion(dummyId, regionId);
        cfg.redirectEdge(headerId, arcStartNode, dummyId);
        cfg.addEdge(dummyId, arcStartNode);

        arcStartNode = dummyId;
      }

      work.push_back(arcStartNode);

      std::pmr::vector<std::pair<cfg::rvsdg::nodeid_t, cfg::rvsdg::nodeid_t>> exits(&checkpoint_resource);
      while (!work.empty()) {
        auto const v = work.back();
        work.pop_back();

        auto const& items = cfg.getSuccessors(v);
        if (items.size() > 0) cfg.nodes()->moveNodeToRegion(v, regionId);

        for (auto to: items) {
          auto idom = dom.get_idom(to);
          if (idom && *idom == v) {
            work.push_back(to);
          } else {
            // auto const cp =std::distance(continuationPoints.begin(), addUnique(continuationPoints, to));
            exits.emplace_back(v, to);
            auto it = std::find_if(continuationPoints.begin(), continuationPoints.end(), [to](auto const& rhs) { return rhs.first == to; });
            if (it == continuationPoints.end()) {
              auto& cp     = continuationPoints.emplace_back(to, 1);
              cp.second[0] = v;
            } else {
              it->second.push_back(v);
            }
          }
        }
      }

      // handle exits
      if (exits.size() == 1) {
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
      tailId = continuationPoints.front().first;
    } else {
      // todo branches
      tailId = cfg.createSimpleNode();
    }

    for (auto cp: continuationPoints) {
      for (auto pred: cp.second) {
        cfg.redirectEdge(pred, cp.first, tailId);
      }
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