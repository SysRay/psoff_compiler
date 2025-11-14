#include "regions.h"

#include "analysis.h"
#include "analysis/dom.h"
#include "analysis/scc.h"
#include "builder.h"
#include "frontend/ir_types.h"
#include "include/checkpoint_resource.h"
#include "include/common.h"
#include "ir/debug_strings.h"
#include "ir/instructions.h"

#include <iostream>
#include <optional>
#include <set>
#include <span>
#include <stack>
#include <unordered_set>

namespace compiler::frontend::analysis {
void RegionBuilder::addReturn(region_t from) {
  auto itfrom     = splitRegion(from + 1);
  auto before     = std::prev(itfrom);
  before->hasJump = true;
}

void RegionBuilder::addJump(region_t from, region_t to) {
  auto itfrom      = splitRegion(from + 1);
  auto before      = std::prev(itfrom);
  before->trueSucc = to;
  before->hasJump  = true;
  auto itto        = splitRegion(to);
}

void RegionBuilder::addCondJump(region_t from, region_t to) {
  auto itfrom      = splitRegion(from + 1);
  auto before      = std::prev(itfrom);
  before->trueSucc = to;
  auto itto        = splitRegion(to);
}

fixed_containers::FixedVector<regionid_t, 2> RegionBuilder::getSuccessorsIdx(regionid_t id) const {
  fixed_containers::FixedVector<regionid_t, 2> result;

  const auto& r = _regions[id];
  if (r.hasFalseSucc()) { // Fallthrough
    if (1 + id < _regions.size() && _regions[1 + id].start == r.end) {
      result.emplace_back(1 + id);
    }
  }
  if (r.hasTrueSucc()) {
    result.emplace_back(getRegionIndex(r.trueSucc));
  }
  return result;
}

std::pair<region_t, region_t> RegionBuilder::findRegion(region_t from) const {
  auto it = std::upper_bound(_regions.begin(), _regions.end(), from, [](region_t val, const Region& reg) { return val < reg.start; });
  if (it != _regions.begin()) --it;

  if (it->end < from) return std::make_pair(0, 0); // Sanity check
  return std::make_pair(it->start, it->end);
}

std::pair<region_t, region_t> RegionBuilder::getRegion(regionid_t id) const {
  assert(id != NO_REGION);
  auto const& r = _regions[id];
  return std::make_pair(r.start, r.end);
}

regionid_t RegionBuilder::getRegionIndex(region_t pos) const {
  auto it = std::upper_bound(_regions.begin(), _regions.end(), pos, [](region_t val, const Region& reg) { return val < reg.start; });
  if (it == _regions.begin()) return regionid_t(0);
  --it;
  return regionid_t(std::distance(_regions.begin(), it));
}

RegionBuilder::regionsit_t RegionBuilder::splitRegion(region_t pos) {
  auto it = std::upper_bound(_regions.begin(), _regions.end(), pos, [](region_t val, const Region& reg) { return val < reg.start; });
  if (it != _regions.begin()) --it;
  if (pos <= it->start) {
    return it;
  }
  if (pos >= it->end) {
    return ++it;
  }
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

  os << "  Succ: ";
  visitSuccessors(r.start, [&os](region_t item) { os << "[" << item << "]"; });

  os << "\n  Pred: ";
  visitPredecessors(r.start, [&os](region_t item) { os << "[" << item << "]"; });
  os << "\n";

  if (r.hasTrueSucc()) os << "  True -> [" << r.trueSucc << "]\n";
  if (r.hasFalseSucc()) os << "  False -> [" << r.end << "]\n";
  os << "  hasJump: " << r.hasJump << "\n";
  os << "-------------------------\n";
}

using namespace compiler::ir;

bool createRegions(std::pmr::polymorphic_allocator<> allocator, std::span<ir::InstCore> instructions, pcmapping_t const& mapping) {
  RegionBuilder regions(instructions.size(), allocator);

  // Collect Labels first
  for (size_t n = 0; n < instructions.size(); ++n) {
    auto const& inst = instructions[n];
    if (inst.group != eInstructionGroup::kFlowControl) continue;

    switch (conv(inst.kind)) {
      case eInstKind::ReturnOp: {
        regions.addReturn(n);
      } break;
      case eInstKind::DiscardOp: {
        // todo needed?
      } break;
      case eInstKind::JumpAbsOp: {
        auto const targetPc = evaluate(allocator, instructions, regions, n, inst.srcOperands[0]);
        if (!targetPc) return false;

        auto const targetIt = std::lower_bound(mapping.begin(), mapping.end(), targetPc->value_u64, [](auto const& b, uint64_t val) { return b.first < val; });
        regions.addJump(n, targetIt->second);
      } break;
      case eInstKind::CondJumpAbsOp: {
        auto const targetPc = evaluate(allocator, instructions, regions, n, inst.srcOperands[1]);
        if (!targetPc) return false;

        auto const targetIt = std::lower_bound(mapping.begin(), mapping.end(), targetPc->value_u64, [](auto const& b, uint64_t val) { return b.first < val; });
        regions.addCondJump(n, targetIt->second);
      } break;

      default: break;
    }
  }

  regions.for_each([&](uint32_t start, uint32_t end, void* region) {
    regions.dump(std::cout, region);
    for (auto n = start; n < end; ++n) {
      auto it = std::find_if(mapping.begin(), mapping.end(), [n](auto const& item) { return item.second == n; });
      if (it == mapping.end())
        std::cout << "\t";
      else
        std::cout << std::hex << it->first;
      std::cout << '\t' << std::dec << n << "| ";
      ir::debug::getDebug(std::cout, instructions[n]);
    }
  });

  // // transform to hierarchical structured graph
  // ref: "Perfect Reconstructability of Control Flow from Demand Dependence Graphs"
  // auto rootNode = transformStructuredCFG(&builder.getBuffer(), &checkpoint, regions);
  // dump(std::cout, &rootNode);
  // dump(std::cout, &rootNode, instructions.data());

  return true;
}

RegionGraph::RegionGraph(std::pmr::polymorphic_allocator<> allocator, const RegionBuilder& rb): nodes(allocator), succ(allocator), pred(allocator) {
  constexpr auto offset = 1 + STOP_ID;

  size_t const N = offset + rb.getNumRegions();
  nodes.reserve(N);
  succ.resize(N);
  pred.resize(N);

  nodes.emplace_back(StartRegion {START_ID});
  nodes.emplace_back(StopRegion {STOP_ID});
  if (N == offset) return; // no nodes

  // Connect with start
  accessSuccessors(START_ID).push_back(regionid_t {offset});
  accessPredecessors(regionid_t {offset}).push_back(START_ID);

  // Create BasicRegion nodes from RegionBuilder
  // Note: offset id by 1+STOP_ID

  for (regionid_t i {0}; i.value < N - offset; ++i.value) {
    auto const id = regionid_t {offset + i.value};

    auto const [start, end] = rb.getRegion(i);
    nodes.emplace_back(BasicRegion {id, start, end});

    auto successors = rb.getSuccessorsIdx(i);
    if (successors.empty()) {
      // Connect with stop
      accessSuccessors(id).push_back(STOP_ID);
      accessPredecessors(STOP_ID).push_back(id);
    } else {
      for (auto const& sid_: successors) {
        auto const sid = regionid_t {offset + sid_.value};
        succ[id].push_back(sid);
        pred[sid.value].push_back(id);
      }
    }
  }
}

struct GraphAdapter {
  const frontend::analysis::RegionGraph& g;

  auto getSuccessors(uint32_t idx) const {
    return g.getSuccessors(analysis::regionid_t {idx}) | std::views::transform([](auto id) { return id.value; });
  }

  auto getPredecessors(uint32_t idx) const {
    return g.getPredecessors(analysis::regionid_t {idx}) | std::views::transform([](auto id) { return id.value; });
  }

  size_t size() const { return g.getNodeCount(); }

  GraphAdapter(analysis::RegionGraph& g): g(g) {}
};

void structurizeRegions(util::checkpoint_resource& checkpoint_resource, analysis::RegionGraph& regionGraph) {
  std::stack<std::pair<analysis::regionid_t, analysis::regionid_t>> tasks;
  tasks.push({regionGraph.getStartId(), regionGraph.getStopId()});

  while (!tasks.empty()) {
    auto const [startId, endId] = tasks.top();
    tasks.pop();

    { // // detect loops
      auto checkpoint = checkpoint_resource.checkpoint();

      // Note: Create one entry and exit node, entry is also the reentry point
      GraphAdapter adapter(regionGraph);

      // todo scc ranges
      auto const sccs = compiler::analysis::SCCBuilder<GraphAdapter>(&checkpoint_resource, adapter).calculate();
      for (size_t n = 0; n < sccs.get().size(); ++n) {
        auto const& scc = sccs.get()[n];

        auto const loopId  = regionGraph.createNode<analysis::LoopRegion>();
        auto const startId = regionGraph.createNode<analysis::StartRegion>();
        auto const exitId  = regionGraph.createNode<analysis::StopRegion>();
        auto const contId  = regionGraph.createNode<analysis::StopRegion>();

        {
          auto& loopNode   = std::get<analysis::LoopRegion>(regionGraph.getNode(loopId));
          loopNode.startId = startId;
          loopNode.exitId  = exitId;
          loopNode.contId  = contId;
        }

        { // classify scc and add loop

          auto const meta = compiler::analysis::classifySCC(&checkpoint_resource, adapter, scc);

          // Handle edges (insert loop node)
          // link preds -> loop -> succs
          auto& preds = regionGraph.accessPredecessors(loopId);
          preds.reserve(preds.size() + meta.preds.size());
          for (auto item: meta.preds) {
            preds.push_back(analysis::regionid_t(item));
          }

          auto& succs = regionGraph.accessSuccessors(loopId);
          succs.reserve(succs.size() + meta.succs.size());
          for (auto item: meta.succs) {
            succs.push_back(analysis::regionid_t(item));
          }

          // link pred <- loop
          for (auto pred: regionGraph.accessPredecessors(loopId)) {
            for (auto& succ: regionGraph.accessSuccessors(pred)) {
              if (meta.incoming.contains(succ)) {
                succ = loopId;
                break;
              }
            }
          }

          // link loop <- succs
          for (auto succ: regionGraph.accessSuccessors(loopId)) {
            for (auto& pred: regionGraph.accessPredecessors(succ)) {
              if (meta.outgoing.contains(pred)) {
                pred = loopId;
                break;
              }
            }
          }
          // -

          // Handle entries
          if (meta.incoming.size() > 1) {
            // Needs restructure (q)
            // auto const newEntry = regions.createRegion(0,0);
            throw std::runtime_error("todo multiple loop entries");
          } else if (!meta.incoming.empty()) {
            analysis::regionid_t bodyStartId = analysis::regionid_t(*meta.incoming.begin());

            // set continue node
            auto& preds = regionGraph.accessPredecessors(bodyStartId);
            for (auto pred: preds) {
              if (meta.preds.contains(pred)) continue;
              regionGraph.accessPredecessors(contId).push_back(pred);

              for (auto& succ: regionGraph.accessSuccessors(pred)) {
                if (succ == bodyStartId) {
                  succ = contId;
                  break;
                }
              }
              break;
            }
            preds.clear();

            // Connect with start
            preds.push_back(startId);
            regionGraph.accessSuccessors(startId).push_back(bodyStartId);
          }

          // Handle exits
          if (meta.outgoing.size() > 1) {
            // Needs restructure (r)
            throw std::runtime_error("todo multiple loop exits");
          } else if (!meta.outgoing.empty()) {
            analysis::regionid_t bodyStopId = analysis::regionid_t(*meta.outgoing.begin());

            // set exit node
            auto& succs = regionGraph.accessSuccessors(bodyStopId);
            for (auto& succ: succs) {
              if (meta.succs.contains(succ)) {
                succ = exitId;
              }
            }
            regionGraph.accessPredecessors(exitId).push_back(bodyStopId);
          }
          // if (loopNode.startId != loopNode.stop) { // not a self loop
          //   // tasks.push({loopNode.start, loopNode.stop}); // todo scc start stop
          // }
        }
      }
    } // - detect loops

    { // // detect conditions
      //  1. Find conditional branch
      //  2. Get merge point from post dominator
      //  3. todo (exits, tail resturcturing)
      auto checkpoint = checkpoint_resource.checkpoint();

      compiler::analysis::DominatorTreeDense<GraphAdapter> dom({&checkpoint_resource});
      std::pmr::vector<analysis::regionid_t>               work(&checkpoint_resource);
      work.reserve(64);

      auto headerId = startId;
      while (true) {
        auto const& succs = regionGraph.getSuccessors(headerId);
        if (succs.size() == 0) break;
        if (succs.size() == 1) {
          headerId = succs[0];
          continue;
        }
        assert(succs.size() == 2);
        // work.clear();

        // found condition -> find branch exits
        dom.calculate(GraphAdapter(regionGraph), startId.value); // build once on demand

        std::pmr::vector<std::pmr::vector<std::pair<analysis::regionid_t, uint32_t>>> exits(succs.size(), &checkpoint_resource); // edges from exit to cp
        std::pmr::vector<analysis::regionid_t>                                        continuationPoints(&checkpoint_resource);
        for (uint8_t arc = 0; arc < succs.size(); ++arc) {
          work.push_back(succs[arc]);
          while (!work.empty()) {
            auto const v = work.back();
            work.pop_back();

            auto const& items = regionGraph.getSuccessors(v);

            for (auto to: items) {
              auto idom = dom.get_idom(to);
              if (idom && *idom == v) {
                work.push_back(to);
              } else {
                // exits.push_back(v);
                auto const cp = std::distance(continuationPoints.begin(), addUnique(continuationPoints, to));
                exits[arc].push_back({v, cp});
              }
            }
          }
        }

        auto const condId  = regionGraph.createNode<analysis::CondRegion>();
        auto const mergeId = regionGraph.createNode<analysis::StopRegion>();

        std::array<analysis::regionid_t, 2> branchId = {regionGraph.createNode<analysis::StartRegion>(), regionGraph.createNode<analysis::StartRegion>()};

        {
          auto& condNode   = std::get<analysis::CondRegion>(regionGraph.getNode(condId));
          condNode.b0      = branchId[0];
          condNode.b1      = branchId[1];
          condNode.mergeId = mergeId;
        }

        // Handle Tail
        analysis::regionid_t tailId = NO_REGION;

        assert(continuationPoints.size() > 0);
        if (continuationPoints.size() == 1) {
          tailId = continuationPoints[0];
          printf("tail %u\n", tailId.value);
          // edge exits to mergeId
          std::pmr::vector<analysis::regionid_t> exitNodes;
          exitNodes.reserve([&exits]() {
            uint32_t count = 0;
            for (auto const& b: exits)
              count += b.size();
            return count;
          }());

          for (uint8_t b = 0; b < exits.size(); ++b) {
            if (exits[b].empty()) {
              addUnique(exitNodes, branchId[b]);
            } else {
              for (auto const [from, to]: exits[b]) {
                // Note: If from is a continuatin point, then branch is empty
                // use startId (b[]) as exit
                if (std::find(continuationPoints.begin(), continuationPoints.end(), from) != continuationPoints.end()) {
                  addUnique(exitNodes, branchId[b]); // startId -> mergeId
                } else {
                  addUnique(exitNodes, from);
                }
              }
            }
          }

          for (auto id: exitNodes) {
            regionGraph.accessSuccessors(id).clear();
            regionGraph.accessSuccessors(id).push_back(mergeId);
            regionGraph.accessPredecessors(mergeId).push_back(id);
          }
        } else {
          // Needs restructuring
          throw std::runtime_error("todo multiple cp");
        }

        // // Handle edges
        // header <-> condNode
        regionGraph.accessSuccessors(headerId).clear();
        regionGraph.accessSuccessors(headerId).push_back(condId);
        regionGraph.accessPredecessors(condId).push_back(headerId);

        regionGraph.accessPredecessors(tailId).clear();           // condNode <- tailId
        regionGraph.accessPredecessors(tailId).push_back(condId); // condNode <- tailId
        regionGraph.accessSuccessors(condId).push_back(tailId);   // condNode -> tailId

        // // branches start
        // todo only when not continuation port
        // regionGraph.accessPredecessors(succs[0]).clear();
        // regionGraph.accessPredecessors(succs[0]).push_back(branchId[0]);
        // regionGraph.accessPredecessors(succs[1]).clear();
        // regionGraph.accessPredecessors(succs[1]).push_back(branchId[1]);

       // regionGraph.accessSuccessors(branchId[0]).push_back(succs[0]);
        //regionGraph.accessSuccessors(branchId[1]).push_back(succs[1]);
        //  // // -

        headerId = tailId; // continue search
      }

    } // - detect conditions
  }
}

static void dumpNodeHeader(std::ostream& os, const RegionNode& node) {
  std::visit(
      [&](auto const& n) {
        using T = std::decay_t<decltype(n)>;
        if constexpr (std::is_same_v<T, StartRegion>) {
          os << "StartRegion{id=" << n.id.value << "}";
        } else if constexpr (std::is_same_v<T, StopRegion>) {
          os << "StopRegion{id=" << n.id.value << "}";
        } else if constexpr (std::is_same_v<T, BasicRegion>) {
          os << "BasicRegion{id=" << n.id.value << ", begin=" << n.begin << ", end=" << n.end << "}";
        } else if constexpr (std::is_same_v<T, CondRegion>) {
          os << "CondRegion{id=" << n.id.value << ", B_0=" << n.b0.value << ", B_1=" << n.b1.value << ", merge=" << n.mergeId.value << "}";
        } else if constexpr (std::is_same_v<T, LoopRegion>) {
          os << "LoopRegion{id=" << n.id.value << ", start=" << n.startId.value << ", exit=" << n.exitId.value << ", continue=" << n.contId.value << "}";
        }
      },
      node);
}

void printIndent(std::ostream& os, int indent) {
  for (int i = 0; i < indent; ++i)
    os << "  ";
}

static void dumpSuccPred(std::ostream& os, const RegionGraph& g, regionid_t id, int indent) {
  printIndent(os, indent);
  os << "succ: ";
  auto succ = g.getSuccessors(id);
  if (succ.empty())
    os << "(none)";
  else
    for (auto s: succ)
      os << s.value << " ";
  os << "\n";

  printIndent(os, indent);
  os << "pred: ";
  auto pred = g.getPredecessors(id);
  if (pred.empty())
    os << "(none)";
  else
    for (auto p: pred)
      os << p.value << " ";
  os << "\n";
}

static void dumpSubgraph(std::ostream& os, const RegionGraph& g, regionid_t id, regionid_t stop, std::unordered_set<uint32_t>& visited, int indent) {
  if (visited.contains(id.value) || id == NO_REGION) return;

  visited.insert(id.value);
  const auto& node = g.getNode(id);

  // Node header
  printIndent(os, indent);
  dumpNodeHeader(os, node);
  os << "\n";

  // Print succ/pred
  dumpSuccPred(os, g, id, indent + 1);

  if (id.value == stop.value) return;

  // Structural recursion only when needed
  std::visit(
      [&](auto const& n) {
        using T = std::decay_t<decltype(n)>;

        if constexpr (std::is_same_v<T, LoopRegion>) {
          os << std::string(indent + 1, ' ') << "{\n";
          dumpSubgraph(os, g, n.startId, n.exitId, visited, indent + 2);
          os << std::string(indent + 1, ' ') << "}\n";
        } else if constexpr (std::is_same_v<T, CondRegion>) {
          os << std::string(indent + 1, ' ') << "True:{\n";
          dumpSubgraph(os, g, n.b0, n.mergeId, visited, indent + 2);
          os << std::string(indent + 1, ' ') << "}\n";
          printIndent(os, indent + 1);

          os << std::string(indent + 1, ' ') << "False:{\n";
          dumpSubgraph(os, g, n.b1, n.mergeId, visited, indent + 2);
          os << std::string(indent + 1, ' ') << "}\n";
        }
      },
      node);

  // Continue linear chain only if it's linear
  for (auto succ: g.getSuccessors(id)) {
    dumpSubgraph(os, g, succ, stop, visited, indent);
  }
}

void dump(std::ostream& os, const RegionGraph& g) {
  os << "RegionGraph Structure:\n";
  std::unordered_set<uint32_t> visited;
  dumpSubgraph(os, g, g.getStartId(), g.getStopId(), visited, 0);
}
} // namespace compiler::frontend::analysis