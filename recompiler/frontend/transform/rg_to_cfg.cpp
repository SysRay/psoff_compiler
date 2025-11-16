#include "../analysis/regions.h"
#include "cfg/cfg.h"
#include "transform.h"

namespace compiler::frontend::transform {
cfg::ControlFlow transformRg2Cfg(std::pmr::polymorphic_allocator<> allocator, analysis::RegionBuilder const& rb) {
  uint32_t const   expectedBlocks = 1 + 1.5 * rb.getNumRegions();
  cfg::ControlFlow cfg(allocator, expectedBlocks); // Leave some place for extra branch nodes

  // Add all blocks to root regions (unstructured cyclic cfg)
  auto& rootRegion = cfg.accessRegion(cfg.getRootRegionId());
  rootRegion.blocks.reserve(expectedBlocks);

  auto const stopId = cfg.createBlock();

  rootRegion.blocks.push_back(stopId);
  rootRegion.exit = stopId;

  if (rb.getNumRegions() == 0) {
    rootRegion.entry = stopId;
    return cfg;
  }

  auto const offset = 1 + stopId.value;

  // 1. Create all blocks for code regions
  for (analysis::regionid_t i {0}; i.value < rb.getNumRegions(); ++i.value) {
    auto const blockId = cfg.createBlock();
    rootRegion.blocks.push_back(blockId);
    assert(blockId == cfg::blocks::blockid_t(offset + i.value));
  }
  rootRegion.entry = cfg::blocks::blockid_t(offset); // Set entry to code region [0]

  // 2. create edges
  for (analysis::regionid_t i {0}; i.value < rb.getNumRegions(); ++i.value) {
    auto const blockId = cfg::blocks::blockid_t(offset + i.value);
    auto       block   = (cfg::blocks::BlockNode*)cfg.accessBlock(blockId);

    auto const [start, end] = rb.getRegion(i);
    block->opbegin          = start;
    block->opend            = end;

    auto successors = rb.getSuccessorsIdx(i);
    if (successors.empty()) {
      cfg.addEdge(blockId, stopId); // Connect with stop
    } else {
      if (successors.size() == 1) {
        // Normal jmp
        cfg.addEdge(blockId, cfg::blocks::blockid_t(offset + successors[0].value));
      } else {
        // Conditional
        // Note: analysis needs min 1 node in each branch. create a dummy block for true branch
        cfg.addEdge(blockId, cfg::blocks::blockid_t(offset + successors[0].value));

        auto const branchId = cfg.createBlock();
        rootRegion.blocks.push_back(branchId);

        cfg.addEdge(blockId, branchId);
        cfg.addEdge(branchId, cfg::blocks::blockid_t(offset + successors[1].value));
      }
    }
  }
  return cfg;
}
} // namespace compiler::frontend::transform