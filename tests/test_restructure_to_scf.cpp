#include "include/checkpoint_resource.h"
#include "ir/debug_strings.h"
#include "ir/blocks.h"
#include "transform/transform.h"

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

using namespace compiler::ir;

static void createCFG(rvsdg::IRBlocks& builder, uint32_t numBlocks, uint32_t start, uint32_t end, std::initializer_list<edge_t> edges) {
  auto funcId = builder.createLambdaNode();
  builder.setMainFunction(funcId);

  auto R = builder.accessRegion(builder.getMainFunction()->body);
  for (uint32_t n = 0; n < numBlocks; ++n) {
    builder.createSimpleNode();
  }

  R->nodes.reserve(numBlocks);

  auto const offset = 1 + funcId;
  builder.move(blockid_t(offset + start), R->id);

  for (uint32_t n = 0; n < numBlocks; ++n) {
    if (n == start || n == end) continue;
    builder.move(blockid_t(offset + n), R->id);
  }

  builder.move(blockid_t(offset + end), R->id);

  for (auto const edge: edges)
    builder.getCfg().addEdge(blockid_t(offset + edge.from.value), blockid_t(offset + edge.to.value));
}

TEST(ControlflowTransform, SimpleIfElse) {
  //   0
  //   |
  //   1
  //  / \
  // 2   3
  // |   |
  // 4   |
  //  \ /
  //   5
  //   |
  //   6

  std::pmr::monotonic_buffer_resource resource;
  std::pmr::polymorphic_allocator<>   allocator {&resource};

  rvsdg::IRBlocks cfg(allocator, 7);
  createCFG(cfg, 7, 0, 5, {{0, 1}, {1, 2}, {1, 3}, {2, 4}, {4, 5}, {3, 5}, {5, 6}});
  // createCFG(cfg, 5, 0, 4, {{0, 1}, {1, 2}, {1, 4}, {2, 4}});
  debug::dumpCFG(std::cout, cfg);

  std::array<uint8_t, 10000>          buffer;
  compiler::util::checkpoint_resource tempResource(buffer.data(), buffer.size());
  compiler::transform::restructureCfg(tempResource, cfg);
  debug::dumpCFG(std::cout, cfg);
  EXPECT_FALSE(true); // todo
}

TEST(ControlflowTransform, CompactIfElse) {
  //   0
  //   |
  //   1
  //  / \
  // 2   |
  //  \ /
  //   3
  //   |
  //   4

  std::pmr::monotonic_buffer_resource resource;
  std::pmr::polymorphic_allocator<>   allocator {&resource};

  rvsdg::IRBlocks cfg(allocator, 7);
  createCFG(cfg, 5, 0, 4, {{0, 1}, {1, 2}, {1, 3}, {2, 3}, {3, 4}});
  debug::dumpCFG(std::cout, cfg);

  std::array<uint8_t, 10000>          buffer;
  compiler::util::checkpoint_resource tempResource(buffer.data(), buffer.size());
  compiler::transform::restructureCfg(tempResource, cfg);
  debug::dumpCFG(std::cout, cfg);
  EXPECT_FALSE(true); // todo
}

TEST(ControlflowTransform, SimpleWhileLoop) {
  //   0
  //   |
  //   1
  //  / \
  // 2   3 -> 1
  //  \
  //   4

  std::pmr::monotonic_buffer_resource resource;
  std::pmr::polymorphic_allocator<>   allocator {&resource};

  rvsdg::IRBlocks cfg(allocator, 7);
  createCFG(cfg, 5, 0, 4, {{0, 1}, {1, 2}, {1, 3}, {3, 1}, {2, 4}});
  debug::dumpCFG(std::cout, cfg);

  std::array<uint8_t, 10000>          buffer;
  compiler::util::checkpoint_resource tempResource(buffer.data(), buffer.size());
  compiler::transform::restructureCfg(tempResource, cfg);
  debug::dumpCFG(std::cout, cfg);
  EXPECT_FALSE(true); // todo
}

TEST(ControlflowTransform, SimpleDoLoop) {
  //   0
  //   |
  //   1
  //   |
  //   2 -> 1
  //  /
  // 3

  std::pmr::monotonic_buffer_resource resource;
  std::pmr::polymorphic_allocator<>   allocator {&resource};

  rvsdg::IRBlocks cfg(allocator, 7);
  createCFG(cfg, 4, 0, 3, {{0, 1}, {1, 2}, {2, 3}, {2, 1}});
  debug::dumpCFG(std::cout, cfg);

  std::array<uint8_t, 10000>          buffer;
  compiler::util::checkpoint_resource tempResource(buffer.data(), buffer.size());
  compiler::transform::restructureCfg(tempResource, cfg);
  debug::dumpCFG(std::cout, cfg);
  EXPECT_FALSE(true); // todo
}

TEST(ControlflowTransform, NestedDoLoop) {
  //   0
  //   |
  //   1
  //   |
  //   2
  //  / \
  // 3   4
  //     |
  //     5
  //    / \
  //   1   4

  std::pmr::monotonic_buffer_resource resource;
  std::pmr::polymorphic_allocator<>   allocator {&resource};

  rvsdg::IRBlocks cfg(allocator, 7);
  createCFG(cfg, 6, 0, 3, {{0, 1}, {1, 2}, {2, 3}, {2, 4}, {4, 5}, {5, 1}, {5, 4}});
  debug::dumpCFG(std::cout, cfg);

  std::array<uint8_t, 10000>          buffer;
  compiler::util::checkpoint_resource tempResource(buffer.data(), buffer.size());
  compiler::transform::restructureCfg(tempResource, cfg);
  debug::dumpCFG(std::cout, cfg);
  EXPECT_FALSE(true); // todo
}

TEST(ControlflowTransform, NestedDoLoopSelfs) {
  //   0
  //   |
  //   1
  //   |
  //   2
  //  / \
  // 3   4
  //     |
  //     5
  //    / \
  //   2   5

  std::pmr::monotonic_buffer_resource resource;
  std::pmr::polymorphic_allocator<>   allocator {&resource};

  rvsdg::IRBlocks cfg(allocator, 7);
  createCFG(cfg, 6, 0, 3, {{0, 1}, {1, 2}, {2, 3}, {2, 4}, {4, 5}, {5, 2}, {5, 5}});
  debug::dumpCFG(std::cout, cfg);

  std::array<uint8_t, 10000>          buffer;
  compiler::util::checkpoint_resource tempResource(buffer.data(), buffer.size());
  compiler::transform::restructureCfg(tempResource, cfg);
  debug::dumpCFG(std::cout, cfg);
  EXPECT_FALSE(true); // todo
}

// fig. 4
TEST(ControlflowTransform, BranchWithMultipleExitsIntoTail) {
  //      0
  //      |
  //      1 (branch)
  //     / \
  //    2   6
  //    |
  //    3
  //   / \
  //  4   5
  //  |   |
  //  7   6
  //      |
  //      7
  //      |
  //      8

  std::pmr::monotonic_buffer_resource resource;
  std::pmr::polymorphic_allocator<>   allocator {&resource};

  rvsdg::IRBlocks cfg(allocator, 7);
  createCFG(cfg, 9, 0, 8, {{0, 1}, {1, 2}, {1, 6}, {2, 3}, {6, 7}, {3, 4}, {3, 5}, {4, 7}, {5, 6}, {6, 7}, {7, 8}});
  // createCFG(cfg, 8, 0, 7, {{0, 1}, {0, 6}, {1, 2}, {2, 6}, {2, 4}, {4, 5}, {5, 6}});
  debug::dumpCFG(std::cout, cfg);

  std::array<uint8_t, 10000>          buffer;
  compiler::util::checkpoint_resource tempResource(buffer.data(), buffer.size());
  compiler::transform::restructureCfg(tempResource, cfg);
  debug::dumpCFG(std::cout, cfg);
  EXPECT_FALSE(true); // todo
}

TEST(ControlflowTransform, DeeplyNestedBranches) {
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

  rvsdg::IRBlocks cfg(allocator, 7);
  createCFG(cfg, 15, 0, 14,
            {{0, 1}, {0, 2}, {1, 3}, {2, 13}, {3, 4}, {3, 5}, {4, 6}, {5, 11}, {6, 7}, {6, 8}, {11, 12}, {7, 9}, {8, 9}, {9, 10}, {10, 12}, {12, 14}});
  debug::dumpCFG(std::cout, cfg);

  std::array<uint8_t, 10000>          buffer;
  compiler::util::checkpoint_resource tempResource(buffer.data(), buffer.size());
  compiler::transform::restructureCfg(tempResource, cfg);
  debug::dumpCFG(std::cout, cfg);
  EXPECT_FALSE(true); // todo
}