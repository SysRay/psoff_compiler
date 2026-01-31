#include "../analysis/regions.h"
#include "ir/blocks.h"
#include "transform.h"

namespace compiler::frontend::transform {
bool transformRg2Cfg(std::pmr::polymorphic_allocator<> allocator, analysis::RegionBuilder const& rb, std::span<compiler::InstructionId_t> instructions,
                     ir::ControlFlow& cfg) {
  // todo
  // uint32_t const   expectedBlocks = 1 + 1.5 * rb.getNumRegions();
  // cfg::ControlFlow cfg(allocator, expectedBlocks); // Leave some place for extra branch nodes

  // auto nodes = cfg.nodes();
  // // Add all blocks to root regions (unstructured cyclic cfg)
  // auto funcId = cfg.createLambdaNode();
  // nodes->setMainFunction(funcId);

  // auto rootRegion = nodes->accessRegion(nodes->getMainFunction()->body);
  // rootRegion->nodes.reserve(expectedBlocks);

  // auto const stopId = cfg.createSimpleNode(); // add it at end later

  // if (rb.getNumRegions() == 0) {
  //   return cfg;
  // }

  // auto const offset = 1 + stopId.value;

  // // 1. Create all blocks for code regions
  // for (analysis::regionid_t i {0}; i.value < rb.getNumRegions(); ++i.value) {
  //   auto const blockId = cfg.createSimpleNode();
  //   nodes->moveNodeToRegion(blockId, rootRegion->id);
  //   assert(blockId == cfg::rvsdg::blockid_t(offset + i.value));
  // }

  // // 2. create edges
  // for (analysis::regionid_t i {0}; i.value < rb.getNumRegions(); ++i.value) {
  //   auto const blockId = cfg::rvsdg::blockid_t(offset + i.value);
  //   auto       block   = nodes->accessNode<cfg::rvsdg::SimpleNode>(blockId);

  //   auto const [start, end] = rb.getRegion(i);
  //   block->instructions.insert(block->instructions.end(), instructions.begin() + start, instructions.begin() + end);

  //   auto successors = rb.getSuccessorsIdx(i);
  //   if (successors.empty()) {
  //     cfg.addEdge(blockId, stopId); // Connect with stop
  //   } else {
  //     if (successors.size() == 1) {
  //       // Normal jmp
  //       cfg.addEdge(blockId, cfg::rvsdg::blockid_t(offset + successors[0].value));
  //     } else {
  //       // Conditional
  //       // Note: analysis needs min 1 node in each branch. create a dummy block for true branch
  //       cfg.addEdge(blockId, cfg::rvsdg::blockid_t(offset + successors[0].value));

  //       auto const branchId = cfg.createSimpleNode();
  //       nodes->moveNodeToRegion(branchId, rootRegion->id);

  //       cfg.addEdge(blockId, branchId);
  //       cfg.addEdge(branchId, cfg::rvsdg::blockid_t(offset + successors[1].value));
  //     }
  //   }
  // }

  // nodes->moveNodeToRegion(stopId, rootRegion->id); // exit at end
  // return cfg;
  return true;
}
} // namespace compiler::frontend::transform