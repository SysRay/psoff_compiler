#include "analysis/dom.h"
#include "analysis/scc.h"
#include "frontend/ir_types.h"
#include "include/checkpoint_resource.h"
#include "include/common.h"
#include "ir/blocks.h"
#include "ir/cfg.h"
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
  ir::ControlFlow& g;

  auto getSuccessors(uint32_t idx) const {
    return g.getSuccessors(ir::blockid_t {idx}) | std::views::transform([](auto id) { return id.value; });
  }

  auto getPredecessors(uint32_t idx) const {
    return g.getPredecessors(ir::blockid_t {idx}) | std::views::transform([](auto id) { return id.value; });
  }

  size_t size() const { return g.size(); }

  GraphAdapter(ir::ControlFlow& g): g(g) {}
};

static void collapseCycles(util::checkpoint_resource& checkpoint_resource, ir::ControlFlow& cfg, std::pmr::vector<ir::regionid_t>& tasks,
                           ir::regionid_t regionId) {
  // 1. Find SCCs
  // 2. find entries, exits and back edges
  // 3 .collaps into loop node

  auto& builder = cfg.getBlocks();

  auto region = builder.getRegion(regionId);
  if (region->blocks.empty()) return;

  auto checkpoint = checkpoint_resource.checkpoint();

  GraphAdapter adapter(cfg);
  auto const   sccs = compiler::analysis::SCCBuilder<GraphAdapter>(&checkpoint_resource, adapter).calculate(region->blocks.front());

  auto& im = builder.getInstructions();
  for (auto const& scc: sccs.get()) {
    auto checkpoint = checkpoint_resource.checkpoint();

    auto const sccEdges = compiler::analysis::classifySCC(&checkpoint_resource, adapter, scc);
    if (sccEdges.entryEdges.empty()) {
      throw std::runtime_error("scc with no entry");
    }

    auto const loopId = cfg.createThetaNode();
    auto       loop   = builder.accessNode<ir::rvsdg::ThetaBlock>(loopId);

    auto loopRegions = builder.accessRegion(loop->body);
    loopRegions->blocks.reserve(1 + scc.size()); //  exit + sccNodes

    // Theta node: First result is predicate for exit or continue (latch)
    // Body region contains single entry and single exit

    // // Restructure entries to one entry
    ir::blockid_t headerId = {};
    if (sccEdges.entryEdges.size() > 1) {
      // First value passed to Theta Node is the branch selector value (q-value)
      // redirect all edges to loop and set the branch value

      for (uint32_t n = 0; n < sccEdges.entryEdges.size(); ++n) {
        auto const& edge = sccEdges.entryEdges[n];
        cfg.redirectEdge(ir::blockid_t(edge.first), ir::blockid_t(edge.second), loopId);
        // todo branch value
      }

      // todo create demux
      headerId = ir::blockid_t(sccEdges.entryEdges[0].second);
      builder.regionReplace(loopId, headerId);
    } else {
      headerId = ir::blockid_t(sccEdges.entryEdges[0].second);
      cfg.redirectEdge(ir::blockid_t(sccEdges.entryEdges[0].first), headerId, loopId);
      builder.regionReplace(loopId, headerId);
    }

    builder.move(headerId, loopRegions->id);

    // // Restructure exits to one exit

    // // Special case: already tail based loop -> use node as latch
    // if (sccEdges.exitEdges.size() == 1 && sccEdges.exitEdges[0].first == sccEdges.backEdges[0].first) {
    //   ir::blockid_t exitLatchId = ir::blockid_t(sccEdges.exitEdges[0].first);

    //   builder.removeEdge(exitLatchId, ir::blockid_t(sccEdges.exitEdges[0].second));
    //   builder.removeEdge(exitLatchId, ir::blockid_t(sccEdges.backEdges[0].second));
    //   // todo branch value

    //   builder.addEdge(loopId, ir::blockid_t(sccEdges.exitEdges[0].second));

    //   // Add nodes to region
    //   for (auto id: std::ranges::reverse_view(scc)) {
    //     if (id == headerId || id == exitLatchId) continue;
    //     builder.moveNodeToRegion(ir::blockid_t(id), loopRegions->id);
    //   }

    //   builder.moveNodeToRegion(exitLatchId, loopRegions->id);
    //   tasks.push_back(loopRegions->id);
    //   return;
    // }
    // // - Special case

    // First value returned from theta-node is loop predicate (r-value)
    // (Optional) Second value is for entry selection (q-value)
    // redirect everything to exit latch (Must be one output node)
    ir::blockid_t exitLatchId = cfg.createSimpleNode();
    ir::blockid_t backLatchId = cfg.createSimpleNode();

    if (sccEdges.backEdges.size() > 0) {
      for (auto const& [from, to]: sccEdges.backEdges) {
        cfg.redirectEdge(ir::blockid_t(from), ir::blockid_t(to), backLatchId);
        auto node = builder.accessNode<ir::rvsdg::SimpleBlock>(ir::blockid_t(backLatchId));

        auto predValue = builder.create<ir::dialect::core::ConstantOp>(ir::dialect::OpDst(), ir::ConstantValue {.value_u64 = 1}, ir::OperandType::i1());
        node->instructions.push_back(predValue);

        auto inId = im.createInput(ir::OperandType::i1());
        node->outputs.push_back(inId);
        im.connect(inId, predValue);
      }
    }

    for (auto const& [from, to]: sccEdges.exitEdges) {
      cfg.redirectEdge(ir::blockid_t(from), ir::blockid_t(to), exitLatchId);
      cfg.addEdge(loopId, ir::blockid_t(to));

      auto node = builder.accessNode<ir::rvsdg::SimpleBlock>(ir::blockid_t(exitLatchId));

      auto predValue = builder.create<ir::dialect::core::ConstantOp>(ir::dialect::OpDst(), ir::ConstantValue {.value_u64 = 0}, ir::OperandType::i1());
      node->instructions.push_back(predValue);

      auto inId = im.createInput(ir::OperandType::i1());
      node->outputs.push_back(inId);
      im.connect(inId, predValue);
    }

    // todo demux for exits

    // // Add nodes to region
    // Note scc lists reversed nodes -> reverse to get previous order
    for (auto id: std::ranges::reverse_view(scc)) {
      if (id == headerId.value) continue;
      builder.move(ir::blockid_t(id), loopRegions->id);
    }

    ir::blockid_t latchId = cfg.createSimpleNode(); // one exit node
    cfg.addEdge(backLatchId, latchId);
    cfg.addEdge(exitLatchId, latchId);

    builder.move(backLatchId, loopRegions->id);
    builder.move(exitLatchId, loopRegions->id);
    builder.move(latchId, loopRegions->id);

    tasks.push_back(loopRegions->id); // Collapse sub region
  }
}

static void collapseBranches(util::checkpoint_resource& checkpoint_resource, ir::ControlFlow& cfg, std::pmr::vector<ir::regionid_t>& tasks,
                             ir::regionid_t regionId) {
  //  1. Find conditional branch
  //  2. Get merge point from post dominator
  //  3. todo (exits, tail resturcturing)
  auto& builder = cfg.getBlocks();

  auto region = builder.getRegion(regionId);
  if (region->blocks.empty()) return;

  auto checkpoint = checkpoint_resource.checkpoint();

  compiler::analysis::DominatorTreeDense<GraphAdapter> dom({&checkpoint_resource});

  auto headerId = region->blocks.front();
  while (true) {
    auto const& succs = cfg.getSuccessors(headerId);
    if (succs.size() == 0) break;
    if (succs.size() == 1) {
      headerId = succs[0];
      // todo merge nodes
      continue;
    }

    auto const condId = cfg.createGammaNode(succs.size());
    auto       cond   = builder.accessNode<ir::rvsdg::GammaBlock>(condId);
    builder.regionReplace(condId, ir::blockid_t(succs[0]));

    auto headerBase = builder.accessBase(headerId);

    { // // Get predicate
      // Check if header is correct
      if (headerBase->type != ir::rvsdg::eBlockType::Simple) {
        throw std::runtime_error("wrong header type");
        return;
      }

      auto header = (ir::rvsdg::SimpleBlock*)headerBase;

      if (!header->instructions.empty()) {
        auto terminatorOp = header->instructions.back();

        if (terminatorOp.isValid()) {
          auto const& op = builder.getInstructions().getInstr(terminatorOp);
          if (op.isTerminator()) {
            cond->predicate = op.getInputId(0);
          }
        }
      }

      if (!cond->predicate.isValid()) {
        throw std::runtime_error("block has no terminator");
        return;
      }

      header->instructions.pop_back(); // RVSDG doesn't need terminator ops
    }

    // // Find branches
    dom.calculate(GraphAdapter(cfg), headerId); // build once on demand

    std::pmr::vector<std::pmr::vector<ir::edge_t>> branchExits(succs.size(), &checkpoint_resource);
    std::pmr::vector<ir::blockid_t>                work(&checkpoint_resource);

    work.reserve(64);
    for (uint8_t arc = 0; arc < succs.size(); ++arc) {
      auto branchRegionId = cond->branches[arc];
      tasks.push_back(branchRegionId);

      auto arcStartNode = succs[arc];

      if (cfg.getPredecessors(arcStartNode).size() > 1) {
        // Check if empty arc -> insert dummy node
        // Each arc must have min. one node
        auto dummyId = cfg.createSimpleNode();
        cfg.redirectEdge(headerId, arcStartNode, dummyId);
        cfg.addEdge(dummyId, arcStartNode);

        arcStartNode = dummyId;
      }

      work.push_back(arcStartNode);
      while (!work.empty()) {
        auto const v = work.back();
        work.pop_back();

        auto const& items = cfg.getSuccessors(v);
        if (items.size() > 0) {
          builder.move(v, branchRegionId);
        }

        for (auto to: items) {
          auto idom = dom.get_idom(to);
          if (idom && *idom == v) {
            work.push_back(to);
          } else {
            // auto const cp =std::distance(continuationPoints.begin(), addUnique(continuationPoints, to));

            branchExits[arc].push_back({v, to});
          }
        }
      }
    }

    std::pmr::vector<ir::blockid_t> continuationPoints(&checkpoint_resource);
    for (uint8_t arc = 0; arc < succs.size(); ++arc) {
      auto const& exits = branchExits[arc];

      if (exits.size() > 1) {
        // Multiple: create dummy node as new exit
        auto dummyId = cfg.createSimpleNode();

        auto branchRegionId = cond->branches[arc];
        builder.move(dummyId, branchRegionId);

        for (auto const [from, to]: exits) {
          cfg.redirectEdge(from, to, dummyId);
        }
      }

      // Collect contiunation points
      for (auto const [from, to]: exits) {
        addUnique(continuationPoints, to);
      }
    }

    // // Find tail
    ir::blockid_t tailId = {};
    if (continuationPoints.size() == 1) {
      // One continuation point -> use as tail
      tailId = continuationPoints.front();
    } else {
      // Multiple continuation points -> create tail with branches to the cp
      tailId = cfg.createSimpleNode();
      builder.regionInsertAfter(condId, tailId);

      for (auto const& cp: continuationPoints) {
        cfg.addEdge(tailId, cp);
      }

      auto predValue = ir::dialect::OpSrc(SsaId_t {0}); // todo, add p constants and use the output

      auto node = builder.accessNode<ir::rvsdg::SimpleBlock>(tailId);
      node->instructions.push_back(builder.create<ir::dialect::core::CjumpAbsOp>(
          predValue,
          ir::dialect::OpSrc(getOperandKind(frontend::eOperandKind::createImm(1))))); // Just a dummy to get predicate later
    }

    for (auto const& exits: branchExits) {
      for (auto [from, to]: exits) {
        cfg.removeEdge(from, to);
      }
    }

    cfg.accessSuccessors(headerId).clear();
    cfg.addEdge(headerId, condId);
    cfg.addEdge(condId, tailId);

    headerId = tailId; // Continue;
  }
}

void createRVSDG(util::checkpoint_resource& checkpoint_resource, ir::ControlFlow& cfg) {
  auto checkpoint = checkpoint_resource.checkpoint();

  std::pmr::vector<ir::regionid_t> tasks;
  tasks.push_back(cfg.getBlocks().getMainFunction()->body);

  while (!tasks.empty()) {
    auto const regionId = tasks.back();
    tasks.pop_back();

    collapseCycles(checkpoint_resource, cfg, tasks, regionId);
    collapseBranches(checkpoint_resource, cfg, tasks, regionId);
  }
  // todo linearize regions (check successors)
}
} // namespace compiler::transform