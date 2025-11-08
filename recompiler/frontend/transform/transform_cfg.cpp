#include "../analysis/regions.h"
#include "analysis/dom.h"
#include "analysis/scc.h"
#include "ir/debug_strings.h"
#include "transform.h"

#include <memory_resource>
#include <queue>
#include <stack>
#include <stdexcept>

namespace compiler::frontend::transform {

struct SCCAdapter {
  const frontend::analysis::RegionGraph& g;

  auto getSuccessors(uint32_t idx) const {
    return g.getSuccessors(analysis::regionid_t {idx}) | std::views::transform([](auto id) { return id.value; });
  }

  auto getPredecessors(uint32_t idx) const {
    return g.getPredecessors(analysis::regionid_t {idx}) | std::views::transform([](auto id) { return id.value; });
  }

  size_t size() const { return g.getNodeCount(); }

  SCCAdapter(analysis::RegionGraph& g): g(g) {}
};

ir::cfg::ControlFlow transformRegions(std::pmr::polymorphic_allocator<> allocPool, std::pmr::memory_resource* tempPool, analysis::RegionGraph& regionGraph) {
  ir::cfg::ControlFlow cfg(allocPool);

  // std::stack<std::pair<analysis::regionid_t, analysis::regionid_t>> tasks;
  // tasks.push({regionGraph.getStartId(), regionGraph.getStopId()});

  // while (!tasks.empty()) {
  //   auto const [startId, endId] = tasks.top();
  //   tasks.pop();

  //   std::pmr::monotonic_buffer_resource checkpoint(tempPool);
  //   { // // detect loops
  //     // Note: Create one entry and exit node, entry is also the reentry point
  //     SCCAdapter adapter(regionGraph);

  //     // todo scc ranges
  //     auto const sccs = compiler::analysis::SCCBuilder<SCCAdapter>(&checkpoint, adapter).calculate();
  //     for (size_t n = 0; n < sccs.get().size(); ++n) {
  //       auto const& scc = sccs.get()[n];

  //       analysis::regionid_t loopId   = regionGraph.createNode(analysis::LoopRegion());
  //       auto&                loopNode = std::get<analysis::LoopRegion>(regionGraph.getNode(loopId));

  //       { // classify scc and add loop
  //         std::pmr::monotonic_buffer_resource tempAllocator(&checkpoint);

  //         auto const meta = compiler::analysis::classifySCC(&tempAllocator, adapter, scc);

  //         // todo check what is header and body
  //         if (meta.entries.size() > 1) {
  //           // Needs restructure (q)
  //           // auto const newEntry = regions.createRegion(0,0);
  //           throw std::runtime_error("todo multiple loop entries");
  //         } else if (!meta.entries.empty()) {
  //           loopNode.start = analysis::regionid_t(*meta.entries.begin());

  //           // Adjust edges, point to loopId
  //           regionGraph.accessPredecessors(loopId).swap(regionGraph.accessPredecessors(loopNode.start));
  //           for (auto& pred: regionGraph.getPredecessors(loopId)) {
  //             for (auto& succ: regionGraph.accessSuccessors(loopNode.start)) {
  //               if (succ == loopNode.start) {
  //                 succ = loopId;
  //                 break;
  //               }
  //             }
  //           }
  //         }

  //         if (meta.exits.size() > 1) {
  //           // Needs restructure (r)
  //           throw std::runtime_error("todo multiple loop exits");
  //         } else if (!meta.exits.empty()) {
  //           loopNode.stop = analysis::regionid_t(*meta.exits.begin());

  //           // Adjust edges, replace stop with loopId
  //           regionGraph.accessSuccessors(loopId).swap(regionGraph.accessSuccessors(loopNode.stop));
  //           for (auto& succ: regionGraph.accessSuccessors(loopId)) {
  //             if (succ == loopNode.start) continue;

  //             for (auto& pred: regionGraph.accessPredecessors(succ)) {
  //               if (pred == loopNode.stop) {
  //                 pred = loopId;
  //                 break;
  //               }
  //             }
  //           }
  //         }

  //         if (loopNode.start != loopNode.stop) { // not a self loop
  //           // tasks.push({loopNode.start, loopNode.stop}); // todo scc start stop
  //         }
  //       }
  //     }
  //   } // - detect loops

  //   { // // detect conditions
  //   } // - detect conditions
  // }
  // // { // // detect loops
  // //   auto const sccs = compiler::analysis::SCCBuilder(&checkpoint, regionEdges).calculate();
  // //   for (size_t n = 0; n < sccs.get().size(); ++n) {
  // //     auto const& scc = sccs.get()[n];

  // //     auto  loopId   = cfg.createNode<ir::cfg::NodeLoop>();
  // //     auto& loopNode = cfg.getNode<ir::cfg::NodeLoop>(loopId);

  // //     {
  // //       std::pmr::monotonic_buffer_resource temp(&checkpoint);

  // //       compiler::analysis::SCCMeta meta(&temp);
  // //       compiler::analysis::classifySCC(&temp, regionEdges, scc, meta);

  // //       if (meta.entries.size() > 1) {
  // //         // Needs restructure (q)
  // //         // auto const newEntry = regions.createRegion(0,0);
  // //         throw std::runtime_error("todo multiple loop entries");
  // //       } else if (!meta.entries.empty()) {
  // //         loopNode.header = ir::cfg::NodeId(*meta.entries.begin(), ir::cfg::NodeType::Block);
  // //       }

  // //       if (meta.exits.size() > 1) {
  // //         // Needs restructure (r)
  // //         throw std::runtime_error("todo multiple loop exits");
  // //       } else if (!meta.exits.empty()) {
  // //         // loopNode.merge = ir::cfg::NodeId(*meta.exits.begin(), ir::cfg::NodeType::Block);
  // //       }

  // //       if (!meta.body.empty()) { // not a self loop

  // //         // todo body
  // //       }
  // //     }

  // //     // Adjust graph edges, point to&from loop
  // //     if (loopNode.header != ir::cfg::InvalidNode) {
  // //       auto& preds = graph.accessPredecessors(loopNode.header);
  // //       for (auto pred: preds) {
  // //         for (auto& succ: graph.accessSuccessors(pred)) {
  // //           if (succ != loopNode.header) continue;
  // //           succ = loopId;
  // //           break;
  // //         }
  // //       }

  // //       // if (loopNode.merge != ir::cfg::InvalidNode) {
  // //       //   auto& succHead = graph.accessSuccessors(loopId);
  // //       //   succHead       = graph.accessSuccessors(loopNode.merge);
  // //       // }
  // //     }
  // //   }
  // // }

  // // if (!blocks.succ[startIdx].empty()) { // // detect  branches
  // //   auto headerId = ir::cfg::NodeId(startIdx, ir::cfg::NodeType::Block);
  // //   auto endId    = ir::cfg::NodeId(endIdx, ir::cfg::NodeType::Block);

  // //   compiler::analysis::PostDominatorTreeSparse<GraphAdapter> postdom(GraphAdapter(graph), endId.raw, {&checkpoint});

  // //   // search condition (create head)
  // //   while (true) {
  // //     auto const& succs = graph.getSuccessors(headerId);
  // //     if (succs.size() == 0) break;

  // //     if (succs.size() == 1) {
  // //       cfg.connect(headerId, *succs.begin());
  // //       headerId = *succs.begin();
  // //       continue;
  // //     }

  // //     // found condition
  // //     assert(succs.size() == 2);

  // //     auto ipdom_opt = postdom.get_ipdom(headerId.raw);
  // //     if (!ipdom_opt) continue;

  // //     auto const mergeId = ir::cfg::NodeId(*ipdom_opt);

  // //     auto  condId   = cfg.createNode<ir::cfg::NodeCond>();
  // //     auto& condNode = cfg.getNode<ir::cfg::NodeCond>(condId);

  // //     // Note: inverted predicate to fit ifelse better
  // //     {

  // //       // todo handle subgraph

  // //       auto it                = succs.begin();
  // //       condNode.ifBranchFront = condNode.ifBranchBack = *it++;                       // falltrough
  // //       if (*it != mergeId) condNode.elseBranchFront = condNode.elseBranchBack = *it; // jump
  // //     }

  // //     cfg.connect(headerId, condId);
  // //     cfg.connect(condId, condNode.ifBranchFront);
  // //     cfg.connect(condId, condNode.elseBranchFront);
  // //     cfg.connect(condId, mergeId);

  // //     // Continue search
  // //     headerId = mergeId;
  // //   }
  // // }

  // // // -

  return cfg;
}

} // namespace compiler::frontend::transform