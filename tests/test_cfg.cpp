#include "analysis/dom.h"
#include "frontend/analysis/regions.h"
#include "frontend/transform/transform.h"
#include "include/checkpoint_resource.h"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory_resource>
#include <ranges>
#include <vector>

// TEST(ControlflowTransform, SimpleLinearGraph) {

//   TestGraph g(4);
//   g.addEdge(0, 1);
//   g.addEdge(1, 2);
//   g.addEdge(2, 3);

//   std::pmr::monotonic_buffer_resource                pool;
//   compiler::analysis::DominatorTreeSparse<TestGraph> dom(g, 0, {&pool});

//   EXPECT_EQ(dom.get_idom(0), std::nullopt); // entry has no idom
//   EXPECT_EQ(dom.get_idom(1), 0u);
//   EXPECT_EQ(dom.get_idom(2), 1u);
//   EXPECT_EQ(dom.get_idom(3), 2u);
// }

namespace {
using namespace compiler::frontend::analysis;

static void testGraph(std::span<RegionNode const> expected, RegionGraph const& regionGraph, regionid_t startId) {
  auto curId = startId;
  for (uint32_t i = 0; i < expected.size(); ++i) {
    auto const& actualNode   = regionGraph.getNode(curId);
    const auto& expectedNode = expected[i];

    // Check matching *type*
    ASSERT_EQ(actualNode.index(), expectedNode.index()) << "Node " << i << " type mismatch";

    // If BasicRegion → validate begin/end
    if (std::holds_alternative<BasicRegion>(expectedNode)) {
      const auto& exp = std::get<BasicRegion>(expectedNode);
      const auto& act = std::get<BasicRegion>(actualNode);
      EXPECT_EQ(act.begin, exp.begin) << "BasicRegion.begin mismatch at node " << i;
      EXPECT_EQ(act.end, exp.end) << "BasicRegion.end mismatch at node " << i;
    }
    if (!std::holds_alternative<StopRegion>(expectedNode)) {
      auto succs = regionGraph.getSuccessors(curId);
      curId      = *succs.begin();
      EXPECT_EQ(succs.size(), 1);
    }
  }
}
} // namespace

TEST(ControlflowTransform, SimpleIf) {
  //   0
  //   |
  //   2
  //  / \
  // 3 -> 4
  //     /
  //   1

  using namespace compiler::frontend::analysis;

  std::pmr::monotonic_buffer_resource resource;
  std::pmr::polymorphic_allocator<>   allocator {&resource};

  RegionBuilder builder(50, allocator);

  builder.addCondJump(9, 20);

  RegionGraph regionGraph(allocator, builder);
  dump(std::cout, regionGraph);

  std::array<uint8_t, 10000>          buffer;
  compiler::util::checkpoint_resource tempResource(buffer.data(), buffer.size());
  compiler::frontend::transform::reconstruct(tempResource, regionGraph);
  dump(std::cout, regionGraph);
}

TEST(ControlflowTransform, SimpleIfElse) {
  //   0
  //   |
  //   1
  //  / \
  // 2   3
  //  \ /
  //   4

  using namespace compiler::frontend::analysis;

  std::pmr::monotonic_buffer_resource resource;
  std::pmr::polymorphic_allocator<>   allocator {&resource};

  RegionBuilder builder(50, allocator);

  builder.addCondJump(9, 20);
  builder.addJump(19, 40);

  RegionGraph regionGraph(allocator, builder);
  dump(std::cout, regionGraph);

  std::array<uint8_t, 10000>          buffer;
  compiler::util::checkpoint_resource tempResource(buffer.data(), buffer.size());
  compiler::frontend::transform::reconstruct(tempResource, regionGraph);
  dump(std::cout, regionGraph);
}

TEST(ControlflowTransform, SimpleLoop) {
  //   0
  //  / \
  // 1   2 -> 0
  //  \
  //   3

  std::pmr::monotonic_buffer_resource resource;
  std::pmr::polymorphic_allocator<>   allocator {&resource};

  using namespace compiler::frontend::analysis;
  RegionBuilder builder(50, allocator);

  builder.addCondJump(9, 20);
  builder.addJump(19, 5);

  RegionGraph regionGraph(allocator, builder);

  std::array<uint8_t, 10000>          buffer;
  compiler::util::checkpoint_resource tempResource(buffer.data(), buffer.size());
  compiler::frontend::transform::reconstruct(tempResource, regionGraph);
  dump(std::cout, regionGraph);

  std::vector<RegionNode> expected = {
      StartRegion(), BasicRegion {.begin = 0, .end = 5}, LoopRegion(), BasicRegion {.begin = 20, .end = 50}, StopRegion(),
  };

  testGraph(expected, regionGraph, regionGraph.getStartId());
  // todo test loop
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

  using namespace compiler::frontend::analysis;
  RegionBuilder builder(50, allocator);

  builder.addCondJump(9, 5);

  RegionGraph regionGraph(allocator, builder);

  std::array<uint8_t, 10000>          buffer;
  compiler::util::checkpoint_resource tempResource(buffer.data(), buffer.size());
  compiler::frontend::transform::reconstruct(tempResource, regionGraph);
  dump(std::cout, regionGraph);

  std::vector<RegionNode> expected = {
      StartRegion(), BasicRegion {.begin = 0, .end = 5}, LoopRegion(), BasicRegion {.begin = 10, .end = 50}, StopRegion(),
  };

  testGraph(expected, regionGraph, regionGraph.getStartId());
  // todo test loop
}

TEST(ControlflowTransform, MultipleSimpleLoops) {
  std::pmr::monotonic_buffer_resource resource;
  std::pmr::polymorphic_allocator<>   allocator {&resource};

  using namespace compiler::frontend::analysis;
  RegionBuilder builder(50, allocator);

  builder.addCondJump(9, 20);
  builder.addJump(19, 5);

  builder.addCondJump(40, 30);

  RegionGraph regionGraph(allocator, builder);

  std::array<uint8_t, 10000>          buffer;
  compiler::util::checkpoint_resource tempResource(buffer.data(), buffer.size());
  compiler::frontend::transform::reconstruct(tempResource, regionGraph);
  dump(std::cout, regionGraph);

  std::vector<RegionNode> expected = {
      StartRegion(), BasicRegion {.begin = 0, .end = 5},   LoopRegion(), BasicRegion {.begin = 20, .end = 30},
      LoopRegion(),  BasicRegion {.begin = 41, .end = 50}, StopRegion(),
  };

  testGraph(expected, regionGraph, regionGraph.getStartId());
  // todo test loop
}

// TEST(ControlflowTransform, DiamondGraph) {
//   //   0
//   //  / \
//   // 1   2
//   //  \ /
//   //   3

//   TestGraph g(4);
//   g.addEdge(0, 1);
//   g.addEdge(0, 2);
//   g.addEdge(1, 3);
//   g.addEdge(2, 3);

//   std::pmr::monotonic_buffer_resource                pool;
//   compiler::analysis::PostDominatorTreeSparse<TestGraph> dom(g, 3, {&pool});

//   EXPECT_EQ(dom.get_ipdom(0), 3);
//   EXPECT_EQ(dom.get_ipdom(1), 3u);
//   EXPECT_EQ(dom.get_ipdom(2), 3u);
//   EXPECT_EQ(dom.get_ipdom(3), std::nullopt);
// }

// TEST(PostDominatorTree, BranchWithMultipleExitsIntoTail) {
//   //      0 (branch)
//   //     / \
//   //    1   5
//   //    |   |
//   //    2   6
//   //   / \
//   //  3   4
//   //  |   |
//   //  6   5
//   //      |
//   //      6

//   TestGraph g(7);
//   g.addEdge(0, 1);
//   g.addEdge(0, 5);
//   g.addEdge(1, 2);
//   g.addEdge(2, 3);
//   g.addEdge(2, 4);
//   g.addEdge(3, 6);
//   g.addEdge(4, 5);
//   g.addEdge(5, 6);

//   std::pmr::monotonic_buffer_resource                pool;
//   compiler::analysis::PostDominatorTreeSparse<TestGraph> dom(g, 6, {&pool});

//   EXPECT_EQ(dom.get_ipdom(0), 6);
//   EXPECT_EQ(dom.get_ipdom(1), 2u);
//   EXPECT_EQ(dom.get_ipdom(2), 6u);
//   EXPECT_EQ(dom.get_ipdom(3), 6u);
// }
// TEST(FindBranchExitsAndMerges, DiamondGraph) {
//   //   0
//   //  / \
//   // 1   2
//   //  \ /
//   //   3

//   TestGraph g(4);
//   g.addEdge(0, 1);
//   g.addEdge(0, 2);
//   g.addEdge(1, 3);
//   g.addEdge(2, 3);

//   std::vector<uint32_t> fanouts = {1, 2};

//   auto info = compiler::analysis::findBranchExitsAndMerges(g, 0, fanouts);

//   // Both branches should exit into node 3
//   ASSERT_EQ(info.merges.size(), 1u);
//   EXPECT_TRUE(info.merges.contains(3));

//   EXPECT_EQ(info.exits.at(1).size(), 1u);
//   EXPECT_TRUE(info.exits.at(1).contains(1));
//   EXPECT_EQ(info.exits.at(2).size(), 1u);
//   EXPECT_TRUE(info.exits.at(2).contains(2));
// }

// TEST(FindBranchExitsAndMerges, NestedBranches) {
//   //   0
//   //  / \
//   // 1   2
//   // |   |
//   // 3   4
//   //  \ /
//   //   5

//   TestGraph g(6);
//   g.addEdge(0, 1);
//   g.addEdge(0, 2);
//   g.addEdge(1, 3);
//   g.addEdge(2, 4);
//   g.addEdge(3, 5);
//   g.addEdge(4, 5);

//   std::vector<uint32_t> fanouts = {1, 2};

//   auto info = compiler::analysis::findBranchExitsAndMerges(g, 0, fanouts);

//   // merge at node 5
//   ASSERT_EQ(info.merges.size(), 1u);
//   EXPECT_TRUE(info.merges.contains(5));

//   EXPECT_TRUE(info.exits.at(1).contains(3));
//   EXPECT_TRUE(info.exits.at(2).contains(4));
// }

// TEST(FindBranchExitsAndMerges, MultipleContinuationPoints) {
//   //   0
//   //  / \
//   // 1   2
//   // |   |
//   // 3   4

//   TestGraph g(5);
//   g.addEdge(0, 1);
//   g.addEdge(0, 2);
//   g.addEdge(1, 3);
//   g.addEdge(2, 4);

//   std::vector<uint32_t> fanouts = {1, 2};

//   auto info = compiler::analysis::findBranchExitsAndMerges(g, 0, fanouts);

//   EXPECT_EQ(info.merges.size(), 0u);

//   EXPECT_TRUE(info.exits.at(1).contains(3));
//   EXPECT_TRUE(info.exits.at(2).contains(4));
// }

// TEST(FindBranchExitsAndMerges, ThreeWayMerge) {
//   //    0
//   //  / | \
//   // 1  2  3
//   //  \ | /
//   //    4
//   TestGraph g(5);
//   g.addEdge(0, 1);
//   g.addEdge(0, 2);
//   g.addEdge(0, 3);
//   g.addEdge(1, 4);
//   g.addEdge(2, 4);
//   g.addEdge(3, 4);

//   std::vector<uint32_t> fanouts = {1, 2, 3};
//   auto                  info    = compiler::analysis::findBranchExitsAndMerges(g, 0, fanouts);

//   ASSERT_EQ(info.merges.size(), 1u);
//   EXPECT_TRUE(info.merges.contains(4));

//   for (auto f: fanouts) {
//     EXPECT_TRUE(info.exits.at(f).contains(f));
//   }
// }

// // fig. 4
// TEST(FindBranchExitsAndMerges, BranchWithMultipleExitsIntoTail) {
//   //      0 (branch)
//   //     / \
//   //    1   5
//   //    |   |
//   //    2   6
//   //   / \
//   //  3   4
//   //  |   |
//   //  6   5
//   //      |
//   //      6

//   TestGraph g(7);
//   g.addEdge(0, 1);
//   g.addEdge(0, 5);
//   g.addEdge(1, 2);
//   g.addEdge(2, 3);
//   g.addEdge(2, 4);
//   g.addEdge(3, 6);
//   g.addEdge(4, 5);
//   g.addEdge(5, 6);

//   std::vector<uint32_t> fanouts = {1, 5};
//   auto                  info    = compiler::analysis::findBranchExitsAndMerges(g, 0, fanouts);
//   ASSERT_EQ(info.merges.size(), 2u);

//   EXPECT_TRUE(info.exits.at(1).contains(3));
//   EXPECT_TRUE(info.exits.at(1).contains(4));
//   EXPECT_TRUE(info.exits.at(5).contains(5));
// }

// TEST(FindBranchExitsAndMerges, DeeplyNestedBranches) {
//   TestGraph g(15);

//   // Level 0: outer branch
//   g.addEdge(0, 1);
//   g.addEdge(0, 2);

//   // Level 1: nested branch in branch 1
//   g.addEdge(1, 3);
//   g.addEdge(3, 4);
//   g.addEdge(3, 5);

//   // Level 2: nested branch in nested branch
//   g.addEdge(4, 6);
//   g.addEdge(6, 7);
//   g.addEdge(6, 8);

//   // Level 2 merge
//   g.addEdge(7, 9);
//   g.addEdge(8, 9);

//   // Level 1 continues and merges
//   g.addEdge(9, 10);
//   g.addEdge(5, 11);
//   g.addEdge(10, 12); // Level 1 merge
//   g.addEdge(11, 12);

//   // Branch 2 simple path
//   g.addEdge(2, 13);

//   // Level 0 merge
//   g.addEdge(12, 14);
//   g.addEdge(13, 14);

//   /*
//    *           0
//    *          / \
//    *         1   2
//    *         |   |
//    *         3   13
//    *        / \   \
//    *       4   5   \
//    *       |   |    \
//    *       6   11    \
//    *      / \   \     \
//    *     7   8   \     \
//    *      \ /     \     \
//    *       9       \     \
//    *       |        \     \
//    *      10         \     \
//    *       \         /     /
//    *        \       /     /
//    *         \     /     /
//    *           12       /
//    *            \      /
//    *             \    /
//    *               14
//    */

//   // Test deepest nested branch (node 6, fanouts {7, 8})
//   {
//     auto info = compiler::analysis::findBranchExitsAndMerges(g, 6, {7, 8});

//     EXPECT_EQ(info.merges.size(), 1);
//     EXPECT_TRUE(info.merges.count(9));

//     EXPECT_TRUE(info.exits[7].count(7));
//     EXPECT_TRUE(info.exits[8].count(8));
//   }

//   // Test middle nested branch (node 3, fanouts {4, 5})
//   {
//     auto info = compiler::analysis::findBranchExitsAndMerges(g, 3, {4, 5});

//     EXPECT_EQ(info.merges.size(), 1);
//     EXPECT_TRUE(info.merges.count(12));

//     EXPECT_TRUE(info.exits[4].count(10));
//     EXPECT_TRUE(info.exits[5].count(11));
//   }

//   // Test outer branch (node 0, fanouts {1, 2})
//   {
//     auto info = compiler::analysis::findBranchExitsAndMerges(g, 0, {1, 2});

//     EXPECT_EQ(info.merges.size(), 1);
//     EXPECT_TRUE(info.merges.count(14));

//     EXPECT_TRUE(info.exits[1].count(12));
//     EXPECT_TRUE(info.exits[2].count(13));
//   }
// }