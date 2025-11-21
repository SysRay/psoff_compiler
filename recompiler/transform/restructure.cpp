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

    tasks.push_back(loopRegions->id);
  }
}

static void collapseBranches(util::checkpoint_resource& checkpoint_resource, cfg::ControlFlow& cfg, std::pmr::vector<cfg::rvsdg::regionid_t>& tasks,
                             cfg::rvsdg::regionid_t regionId) {}

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

// #include "../analysis/regions.h"
// #include "analysis/dom.h"
// #include "analysis/scc.h"
// #include "include/checkpoint_resource.h"
// #include "include/common.h"
// #include "ir/debug_strings.h"
// #include "transform.h"

// #include <memory_resource>
// #include <queue>
// #include <stack>
// #include <stdexcept>

// namespace compiler::frontend::transform {

// struct GraphAdapter {
//   const frontend::analysis::RegionGraph& g;

//   auto getSuccessors(uint32_t idx) const {
//     return g.getSuccessors(analysis::regionid_t {idx}) | std::views::transform([](auto id) { return id.value; });
//   }

//   auto getPredecessors(uint32_t idx) const {
//     return g.getPredecessors(analysis::regionid_t {idx}) | std::views::transform([](auto id) { return id.value; });
//   }

//   size_t size() const { return g.getNodeCount(); }

//   GraphAdapter(analysis::RegionGraph& g): g(g) {}
// };

// static void collapseCycles(util::checkpoint_resource& checkpoint_resource, analysis::RegionGraph& regionGraph, analysis::regionid_t startId,
//                           analysis::regionid_t endIf) {
//   // Find SCCs
//   // find entry, exit and continue edges
//   // collaps into loop node

//   auto checkpoint = checkpoint_resource.checkpoint();

//   GraphAdapter adapter(regionGraph);
//   auto const   sccs = compiler::analysis::SCCBuilder<GraphAdapter>(&checkpoint_resource, adapter).calculate(startId);

//   for (auto const& scc: sccs.get()) {
//     auto checkpoint = checkpoint_resource.checkpoint();

//     auto const sccEdges = compiler::analysis::classifySCC(&checkpoint_resource, adapter, scc);
//     if (sccEdges.entryEdges.empty()) {
//       throw std::runtime_error("scc with no entry");
//     }

//     auto const loopId   = regionGraph.createNode<analysis::LoopRegion>();
//     auto const headerId = regionGraph.createNode<analysis::StartRegion>();
//     auto const exitId   = regionGraph.createNode<analysis::StopRegion>();
//     auto const contId   = regionGraph.createNode<analysis::StopRegion>();

//     {
//       auto& loopNode    = std::get<analysis::LoopRegion>(regionGraph.getNode(loopId));
//       loopNode.headerId = headerId;
//       loopNode.exitId   = exitId;
//       loopNode.contId   = contId;
//     }
//     // todo nodes insertBefore, insertAfter, redirectEdge
//     // todo replaceAllUsesWith

//     if (sccEdges.entryEdges.size() > 1) {
//       // Restructure entries into one loopHeader
//       uint32_t const sizePreds = sccEdges.entryEdges.size() + sccEdges.backEdges.size();

//     } else {
//       analysis::regionid_t loopHeader = analysis::regionid_t(sccEdges.entryEdges[0].second);
//     }
//   }
// }

// static void collapseBranches(util::checkpoint_resource& checkpoint_resource, analysis::RegionGraph& regionGraph, analysis::regionid_t startId,
//                             analysis::regionid_t endIf) {
//   auto checkpoint = checkpoint_resource.checkpoint();
// }

// void reconstruct(util::checkpoint_resource& checkpoint_resource, analysis::RegionGraph& regionGraph) {
//   auto checkpoint = checkpoint_resource.checkpoint();

//   std::pmr::vector<std::pair<analysis::regionid_t, analysis::regionid_t>> tasks;
//   tasks.push_back({regionGraph.getStartId(), regionGraph.getStopId()});

//   while (!tasks.empty()) {
//     auto const [startId, endId] = tasks.back();
//     tasks.pop_back();

//     collapseCycles(checkpoint_resource, regionGraph, startId, endId);
//     collapseBranches(checkpoint_resource, regionGraph, startId, endId);

//     // { // // detect loops
//     //   // // Turning CFG a DAG
//     //   auto checkpoint = checkpoint_resource.checkpoint();

//     //   GraphAdapter adapter(regionGraph);

//     //   // todo scc ranges
//     //   auto const sccs = compiler::analysis::SCCBuilder<GraphAdapter>(&checkpoint_resource, adapter).calculate();
//     //   for (size_t n = 0; n < sccs.get().size(); ++n) {
//     //     auto const& scc = sccs.get()[n];

//     //     auto const loopId  = regionGraph.createNode<analysis::LoopRegion>();
//     //     auto const startId = regionGraph.createNode<analysis::StartRegion>();
//     //     auto const exitId  = regionGraph.createNode<analysis::StopRegion>();
//     //     auto const contId  = regionGraph.createNode<analysis::StopRegion>();

//     //     {
//     //       auto& loopNode   = std::get<analysis::LoopRegion>(regionGraph.getNode(loopId));
//     //       loopNode.startId = startId;
//     //       loopNode.exitId  = exitId;
//     //       loopNode.contId  = contId;
//     //     }

//     //     { // classify scc and add loop

//     //       auto const meta = compiler::analysis::classifySCC(&checkpoint_resource, adapter, scc);

//     //       // Handle edges (insert loop node)
//     //       // link preds -> loop -> succs
//     //       auto& preds = regionGraph.accessPredecessors(loopId);
//     //       preds.reserve(preds.size() + meta.preds.size());
//     //       for (auto item: meta.preds) {
//     //         preds.push_back(analysis::regionid_t(item));
//     //       }

//     //       auto& succs = regionGraph.accessSuccessors(loopId);
//     //       succs.reserve(succs.size() + meta.succs.size());
//     //       for (auto item: meta.succs) {
//     //         succs.push_back(analysis::regionid_t(item));
//     //       }

//     //       // link pred <- loop
//     //       for (auto pred: regionGraph.accessPredecessors(loopId)) {
//     //         for (auto& succ: regionGraph.accessSuccessors(pred)) {
//     //           if (meta.incoming.contains(succ)) {
//     //             succ = loopId;
//     //             break;
//     //           }
//     //         }
//     //       }

//     //       // link loop <- succs
//     //       for (auto succ: regionGraph.accessSuccessors(loopId)) {
//     //         for (auto& pred: regionGraph.accessPredecessors(succ)) {
//     //           if (meta.outgoing.contains(pred)) {
//     //             pred = loopId;
//     //             break;
//     //           }
//     //         }
//     //       }
//     //       // -

//     //       // Handle entries
//     //       if (meta.incoming.size() > 1) {
//     //         // Needs restructure (q)
//     //         // auto const newEntry = regions.createRegion(0,0);
//     //         throw std::runtime_error("todo multiple loop entries");
//     //       } else if (!meta.incoming.empty()) {
//     //         analysis::regionid_t bodyStartId = analysis::regionid_t(*meta.incoming.begin());

//     //         // set continue node
//     //         auto& preds = regionGraph.accessPredecessors(bodyStartId);
//     //         for (auto pred: preds) {
//     //           if (meta.preds.contains(pred)) continue;
//     //           regionGraph.accessPredecessors(contId).push_back(pred);

//     //           for (auto& succ: regionGraph.accessSuccessors(pred)) {
//     //             if (succ == bodyStartId) {
//     //               succ = contId;
//     //               break;
//     //             }
//     //           }
//     //           break;
//     //         }
//     //         preds.clear();

//     //         // Connect with start
//     //         preds.push_back(startId);
//     //         regionGraph.accessSuccessors(startId).push_back(bodyStartId);
//     //       }

//     //       // Handle exits
//     //       if (meta.outgoing.size() > 1) {
//     //         // Needs restructure (r)
//     //         throw std::runtime_error("todo multiple loop exits");
//     //       } else if (!meta.outgoing.empty()) {
//     //         analysis::regionid_t bodyStopId = analysis::regionid_t(*meta.outgoing.begin());

//     //         // set exit node
//     //         auto& succs = regionGraph.accessSuccessors(bodyStopId);
//     //         for (auto& succ: succs) {
//     //           if (meta.succs.contains(succ)) {
//     //             succ = exitId;
//     //           }
//     //         }
//     //         regionGraph.accessPredecessors(exitId).push_back(bodyStopId);
//     //       }
//     //       // if (loopNode.startId != loopNode.stop) { // not a self loop
//     //       //   // tasks.push({loopNode.start, loopNode.stop}); // todo scc start stop
//     //       // }
//     //     }
//     //   }
//     // } // - detect loops

//     { // // detect conditions
//       //  1. Find conditional branch
//       //  2. Get merge point from post dominator
//       //  3. todo (exits, tail resturcturing)
//       auto checkpoint = checkpoint_resource.checkpoint();

//       compiler::analysis::DominatorTreeDense<GraphAdapter> dom({&checkpoint_resource});
//       std::pmr::vector<analysis::regionid_t>               work(&checkpoint_resource);
//       work.reserve(64);

//       auto headerId = startId;
//       while (true) {
//         auto const& succs = regionGraph.getSuccessors(headerId);
//         if (succs.size() == 0) break;
//         if (succs.size() == 1) {
//           headerId = succs[0];
//           continue;
//         }
//         assert(succs.size() == 2);
//         // work.clear();

//         // found condition -> find branch exits
//         dom.calculate(GraphAdapter(regionGraph), startId.value); // build once on demand

//         std::pmr::vector<std::pmr::vector<std::pair<analysis::regionid_t, uint32_t>>> exits(succs.size(), &checkpoint_resource); // edges from exit to cp
//         std::pmr::vector<analysis::regionid_t>                                        continuationPoints(&checkpoint_resource);
//         for (uint8_t arc = 0; arc < succs.size(); ++arc) {
//           work.push_back(succs[arc]);
//           while (!work.empty()) {
//             auto const v = work.back();
//             work.pop_back();

//             auto const& items = regionGraph.getSuccessors(v);

//             for (auto to: items) {
//               auto idom = dom.get_idom(to);
//               if (idom && *idom == v) {
//                 work.push_back(to);
//               } else {
//                 // exits.push_back(v);
//                 auto const cp = std::distance(continuationPoints.begin(), addUnique(continuationPoints, to));
//                 exits[arc].push_back({v, cp});
//               }
//             }
//           }
//         }

//         auto const condId  = regionGraph.createNode<analysis::CondRegion>();
//         auto const mergeId = regionGraph.createNode<analysis::StopRegion>();

//         std::array<analysis::regionid_t, 2> branchId = {regionGraph.createNode<analysis::StartRegion>(), regionGraph.createNode<analysis::StartRegion>()};

//         {
//           auto& condNode   = std::get<analysis::CondRegion>(regionGraph.getNode(condId));
//           condNode.b0      = branchId[0];
//           condNode.b1      = branchId[1];
//           condNode.mergeId = mergeId;
//         }

//         // Handle Tail
//         analysis::regionid_t tailId = analysis::NO_REGION;

//         assert(continuationPoints.size() > 0);
//         if (continuationPoints.size() == 1) {
//           tailId = continuationPoints[0];
//           printf("tail %u\n", tailId.value);
//           // edge exits to mergeId
//           std::pmr::vector<analysis::regionid_t> exitNodes;
//           exitNodes.reserve([&exits]() {
//             uint32_t count = 0;
//             for (auto const& b: exits)
//               count += b.size();
//             return count;
//           }());

//           for (uint8_t b = 0; b < exits.size(); ++b) {
//             if (exits[b].empty()) {
//               addUnique(exitNodes, branchId[b]);
//             } else {
//               for (auto const [from, to]: exits[b]) {
//                 // Note: If from is a continuatin point, then branch is empty
//                 // use startId (b[]) as exit
//                 if (std::find(continuationPoints.begin(), continuationPoints.end(), from) != continuationPoints.end()) {
//                   addUnique(exitNodes, branchId[b]); // startId -> mergeId
//                 } else {
//                   addUnique(exitNodes, from);
//                 }
//               }
//             }
//           }

//           for (auto id: exitNodes) {
//             regionGraph.accessSuccessors(id).clear();
//             regionGraph.accessSuccessors(id).push_back(mergeId);
//             regionGraph.accessPredecessors(mergeId).push_back(id);
//           }
//         } else {
//           // Needs restructuring
//           throw std::runtime_error("todo multiple cp");
//         }

//         // // Handle edges
//         // header <-> condNode
//         regionGraph.accessSuccessors(headerId).clear();
//         regionGraph.accessSuccessors(headerId).push_back(condId);
//         regionGraph.accessPredecessors(condId).push_back(headerId);

//         regionGraph.accessPredecessors(tailId).clear();           // condNode <- tailId
//         regionGraph.accessPredecessors(tailId).push_back(condId); // condNode <- tailId
//         regionGraph.accessSuccessors(condId).push_back(tailId);   // condNode -> tailId

//         // // branches start
//         // todo only when not continuation port
//         // regionGraph.accessPredecessors(succs[0]).clear();
//         // regionGraph.accessPredecessors(succs[0]).push_back(branchId[0]);
//         // regionGraph.accessPredecessors(succs[1]).clear();
//         // regionGraph.accessPredecessors(succs[1]).push_back(branchId[1]);

//         // regionGraph.accessSuccessors(branchId[0]).push_back(succs[0]);
//         // regionGraph.accessSuccessors(branchId[1]).push_back(succs[1]);
//         //  // // -

//         headerId = tailId; // continue search
//       }

//     } // - detect conditions
//   }
// }
// } // namespace compiler::frontend::transform