#include "cfg/cfg.h"
#include "cfg/debug_strings.h"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory_resource>
#include <ranges>
#include <vector>

namespace {
// using namespace compiler::frontend::analysis;

// static void testGraph(std::span<RegionNode const> expected, RegionGraph const& regionGraph, regionid_t startId) {
//   auto curId = startId;
//   for (uint32_t i = 0; i < expected.size(); ++i) {
//     auto const& actualNode   = regionGraph.getNode(curId);
//     const auto& expectedNode = expected[i];

//     // Check matching *type*
//     ASSERT_EQ(actualNode.index(), expectedNode.index()) << "Node " << i << " type mismatch";

//     // If BasicRegion → validate begin/end
//     if (std::holds_alternative<BasicRegion>(expectedNode)) {
//       const auto& exp = std::get<BasicRegion>(expectedNode);
//       const auto& act = std::get<BasicRegion>(actualNode);
//       EXPECT_EQ(act.begin, exp.begin) << "BasicRegion.begin mismatch at node " << i;
//       EXPECT_EQ(act.end, exp.end) << "BasicRegion.end mismatch at node " << i;
//     }
//     if (!std::holds_alternative<StopRegion>(expectedNode)) {
//       auto succs = regionGraph.getSuccessors(curId);
//       curId      = *succs.begin();
//       EXPECT_EQ(succs.size(), 1);
//     }
//   }
// }
} // namespace

using namespace compiler::cfg;

static void createCFG(ControlFlow& cfg, uint32_t numBlocks, uint32_t start, uint32_t stop, std::initializer_list<blocks::edge_t> edges) {
  std::vector<blocks::blockid_t> blocks(numBlocks);

  auto& R = cfg.accessRegion(cfg.getRootRegionId());
  for (auto& block: blocks) {
    block = cfg.createBlock();
    R.blocks.push_back(block);
  }

  for (auto const edge: edges)
    cfg.addEdge(edge.from, edge.to);

  cfg.accessRegion(R.id).entry = blocks[start];
  cfg.accessRegion(R.id).exit  = blocks[stop];
}

TEST(ControlflowTransform, SimpleIf) {
  //   0
  //   |
  //   1
  //  / \
  // 2 -> 3
  //     /
  //   4

  std::pmr::monotonic_buffer_resource resource;
  std::pmr::polymorphic_allocator<>   allocator {&resource};

  ControlFlow cfg(allocator);
  createCFG(cfg, 5, 0, 4, {{0, 1}, {1, 2}, {1, 3}, {2, 3}, {3, 4}});
  dumpCFG(std::cout, cfg);

  // std::array<uint8_t, 10000>          buffer;
  // compiler::util::checkpoint_resource tempResource(buffer.data(), buffer.size());
  // compiler::frontend::transform::reconstruct(tempResource, regionGraph);
  // dump(std::cout, regionGraph);
  EXPECT_FALSE(true); // todo
}

TEST(ControlflowTransform, SimpleIfElse) {
  //   0
  //   |
  //   1
  //  / \
  // 2   3
  //  \ /
  //   4

  std::pmr::monotonic_buffer_resource resource;
  std::pmr::polymorphic_allocator<>   allocator {&resource};

  ControlFlow cfg(allocator);
  createCFG(cfg, 5, 0, 4, {{0, 1}, {1, 2}, {1, 3}, {2, 4}, {3, 4}});
  dumpCFG(std::cout, cfg);

  EXPECT_FALSE(true); // todo
}

TEST(ControlflowTransform, SimpleLoop) {
  //   0
  //  / \
  // 1   2 -> 0
  //  \
  //   3

  std::pmr::monotonic_buffer_resource resource;
  std::pmr::polymorphic_allocator<>   allocator {&resource};

  ControlFlow cfg(allocator);
  createCFG(cfg, 4, 0, 3, {{0, 1}, {0, 2}, {2, 0}, {1, 3}});
  dumpCFG(std::cout, cfg);
  EXPECT_FALSE(true); // todo
}

TEST(ControlflowTransform, SimpleDoLoop) {
  //   0
  //   |
  //   1
  //   |
  //   2
  //  / \
  // 3   1

  std::pmr::monotonic_buffer_resource resource;
  std::pmr::polymorphic_allocator<>   allocator {&resource};

  ControlFlow cfg(allocator);
  createCFG(cfg, 4, 0, 3, {{0, 1}, {1, 2}, {2, 3}, {2, 1}});
  dumpCFG(std::cout, cfg);
  EXPECT_FALSE(true); // todo
}

// fig. 4
TEST(PostDominatorTree, BranchWithMultipleExitsIntoTail) {
  //      0 (branch)
  //     / \
  //    1   5
  //    |   |
  //    2   6
  //   / \
  //  3   4
  //  |   |
  //  6   5
  //      |
  //      6

  std::pmr::monotonic_buffer_resource resource;
  std::pmr::polymorphic_allocator<>   allocator {&resource};

  ControlFlow cfg(allocator);
  createCFG(cfg, 4, 0, 6, {{0, 1}, {0, 5}, {1, 2}, {5, 6}, {2, 3}, {2, 4}, {3, 6}, {4, 5}, {5, 6}});
  dumpCFG(std::cout, cfg);
  EXPECT_FALSE(true); // todo
}

TEST(FindBranchExitsAndMerges, DeeplyNestedBranches) {
  /*
   *           0
   *          / \
   *         1   2
   *         |   |
   *         3   13
   *        / \   \
   *       4   5   \
   *       |   |    \
   *       6   11    \
   *      / \   \     \
   *     7   8   \     \
   *      \ /     \     \
   *       9       \     \
   *       |        \     \
   *      10         \     \
   *       \         /     /
   *        \       /     /
   *         \     /     /
   *           12       /
   *            \      /
   *             \    /
   *               14
   */

  std::pmr::monotonic_buffer_resource resource;
  std::pmr::polymorphic_allocator<>   allocator {&resource};

  ControlFlow cfg(allocator);
  createCFG(cfg, 15, 0, 14,
            {{0, 1}, {0, 2}, {1, 3}, {2, 13}, {3, 4}, {3, 5}, {4, 6}, {5, 11}, {6, 7}, {6, 8}, {11, 12}, {7, 9}, {8, 9}, {9, 10}, {10, 12}, {12, 14}});
  dumpCFG(std::cout, cfg);
  EXPECT_FALSE(true); // todo
}