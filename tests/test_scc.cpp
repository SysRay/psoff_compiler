#include "analysis/debug_strings.h"
#include "analysis/scc.h"
#include "fixed_containers/fixed_vector.hpp"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory_resource>
#include <ranges>
#include <vector>

// ---------------- Mock RegionBuilder ----------------
struct MockRegionBuilder {
  std::vector<fixed_containers::FixedVector<int32_t, 2>> edges;
  std::vector<fixed_containers::FixedVector<int32_t, 2>> preds;

  MockRegionBuilder(std::initializer_list<fixed_containers::FixedVector<int32_t, 2>> init): edges(init), preds(edges.size()) {
    // Build predecessor lists
    for (int32_t from = 0; from < static_cast<int32_t>(edges.size()); ++from) {
      for (int32_t to: edges[from]) {
        preds[to].push_back(from);
      }
    }
  }

  int32_t size() const { return static_cast<int32_t>(edges.size()); }

  auto getSuccessors(uint32_t idx) const { return edges[idx]; }

  auto getPredecessors(uint32_t idx) const { return preds[idx]; }
};

static bool containsComponent(compiler::analysis::SCC const& result, std::initializer_list<int32_t> expected) {
  std::vector<int32_t> sortedExpected(expected);
  std::ranges::sort(sortedExpected);

  return std::ranges::any_of(result.nodes, [&](auto const& comp) {
    std::vector<compiler::analysis::scc_node_t> sortedComp(comp.begin(), comp.end());
    std::ranges::sort(sortedComp);
    return std::ranges::equal(sortedComp, sortedExpected);
  });
}

// Helper function to compare edge vectors
bool containsEdge(const std::pmr::vector<std::pair<compiler::analysis::scc_node_t, compiler::analysis::scc_node_t>>& edges, compiler::analysis::scc_node_t from,
                  compiler::analysis::scc_node_t to) {
  return std::ranges::any_of(edges, [from, to](const auto& edge) { return edge.first == from && edge.second == to; });
}

// Helper to check if edge vectors are equal (order-independent)
bool edgesEqual(const std::pmr::vector<std::pair<compiler::analysis::scc_node_t, compiler::analysis::scc_node_t>>& actual,
                std::initializer_list<std::pair<compiler::analysis::scc_node_t, compiler::analysis::scc_node_t>>   expected) {
  if (actual.size() != expected.size()) return false;

  auto sortedActual   = std::vector(actual.begin(), actual.end());
  auto sortedExpected = std::vector(expected);

  std::ranges::sort(sortedActual);
  std::ranges::sort(sortedExpected);

  return std::ranges::equal(sortedActual, sortedExpected);
}

TEST(SCCBuilderTest, DetectsSelf) {
  std::pmr::monotonic_buffer_resource pool(2048);

  MockRegionBuilder regions({{1}, {2, 1}, {3}, {}});

  auto result = compiler::analysis::SCCBuilder<MockRegionBuilder>(&pool, regions).calculate(0);
  compiler::analysis::debug::dump(std::cout, result);

  EXPECT_EQ(result.get().size(), 1);
  EXPECT_TRUE(containsComponent(result, {1}));

  {
    auto const& nodes = result.get()[0];

    auto meta = compiler::analysis::classifySCC(&pool, regions, nodes);

    EXPECT_EQ(meta.entryEdges.size(), 1);
    EXPECT_EQ(meta.backEdges.size(), 1);
    EXPECT_EQ(meta.exitEdges.size(), 1);

    EXPECT_TRUE(containsEdge(meta.entryEdges, 0, 1));
    EXPECT_TRUE(containsEdge(meta.backEdges, 1, 1));
    EXPECT_TRUE(containsEdge(meta.exitEdges, 1, 2));
  }
}

TEST(SCCBuilderTest, DetectsNoExit) {
  std::pmr::monotonic_buffer_resource pool(2048);

  MockRegionBuilder regions({{1}, {2, 4}, {3}, {2}, {}});

  auto result = compiler::analysis::SCCBuilder<MockRegionBuilder>(&pool, regions).calculate(0);
  compiler::analysis::debug::dump(std::cout, result);

  EXPECT_EQ(result.get().size(), 1);
  EXPECT_TRUE(containsComponent(result, {2, 3}));

  {
    auto const& nodes = result.get()[0];

    auto meta = compiler::analysis::classifySCC(&pool, regions, nodes);

    // Entry: 0 -> 1 -> 2 (entering SCC)
    EXPECT_EQ(meta.entryEdges.size(), 1);
    EXPECT_TRUE(containsEdge(meta.entryEdges, 1, 2));

    // Back edges: 3 -> 2 (back edge within SCC)
    EXPECT_EQ(meta.backEdges.size(), 1);
    EXPECT_TRUE(containsEdge(meta.backEdges, 3, 2));

    // Exit: 1 -> 4 exits the SCC
    EXPECT_EQ(meta.exitEdges.size(), 0);
  }
}

TEST(SCCBuilderTest, DetectsSimpleLoopHead) {
  std::pmr::monotonic_buffer_resource pool(2048);

  MockRegionBuilder regions({{1}, {2, 4}, {3}, {1}, {}});

  auto result = compiler::analysis::SCCBuilder<MockRegionBuilder>(&pool, regions).calculate(0);
  compiler::analysis::debug::dump(std::cout, result);

  EXPECT_EQ(result.get().size(), 1);
  EXPECT_TRUE(containsComponent(result, {1, 2, 3}));

  {
    auto const& nodes = result.get()[0];

    auto meta = compiler::analysis::classifySCC(&pool, regions, nodes);

    // Entry: 0 -> 1 (entering SCC at head)
    EXPECT_EQ(meta.entryEdges.size(), 1);
    EXPECT_TRUE(containsEdge(meta.entryEdges, 0, 1));

    // Back edge: 3 -> 1 (loop back to head)
    EXPECT_EQ(meta.backEdges.size(), 1);
    EXPECT_TRUE(containsEdge(meta.backEdges, 3, 1));

    // Exit: 1 -> 4 exits the SCC
    EXPECT_EQ(meta.exitEdges.size(), 1);
    EXPECT_TRUE(containsEdge(meta.exitEdges, 1, 4));
  }
}

TEST(SCCBuilderTest, DetectsSimpleLoopTail) {
  std::pmr::monotonic_buffer_resource pool(2048);

  MockRegionBuilder regions({{1}, {2}, {3}, {1, 4}, {}});

  auto result = compiler::analysis::SCCBuilder<MockRegionBuilder>(&pool, regions).calculate(0);
  compiler::analysis::debug::dump(std::cout, result);

  EXPECT_EQ(result.get().size(), 1);
  EXPECT_TRUE(containsComponent(result, {1, 2, 3}));

  {
    auto const& nodes = result.get()[0];

    auto meta = compiler::analysis::classifySCC(&pool, regions, nodes);

    // Entry: 0 -> 1 (entering SCC)
    EXPECT_EQ(meta.entryEdges.size(), 1);
    EXPECT_TRUE(containsEdge(meta.entryEdges, 0, 1));

    // Back edge: 3 -> 1 (loop back from tail)
    EXPECT_EQ(meta.backEdges.size(), 1);
    EXPECT_TRUE(containsEdge(meta.backEdges, 3, 1));

    // Exit: 3 -> 4 exits the SCC
    EXPECT_EQ(meta.exitEdges.size(), 1);
    EXPECT_TRUE(containsEdge(meta.exitEdges, 3, 4));
  }
}

TEST(SCCBuilderTest, DetectsNestedLoops) {
  std::pmr::monotonic_buffer_resource pool(2048);

  MockRegionBuilder regions({{1},    // 0 -> 1
                             {2},    // 1 -> 2
                             {1, 3}, // 2 -> 1,3
                             {2, 4}, // 3 -> 2,4
                             {}});

  auto result = compiler::analysis::SCCBuilder<MockRegionBuilder>(&pool, regions).calculate(0);

  compiler::analysis::debug::dump(std::cout, result);

  EXPECT_EQ(result.get().size(), 1);
  EXPECT_TRUE(containsComponent(result, {1, 2, 3}));

  {
    auto const& nodes = result.get()[0];

    auto meta = compiler::analysis::classifySCC(&pool, regions, nodes);

    // Entry: 0 -> 1 (entering SCC)
    EXPECT_EQ(meta.entryEdges.size(), 1);
    EXPECT_TRUE(containsEdge(meta.entryEdges, 0, 1));

    EXPECT_EQ(meta.backEdges.size(), 1);
    EXPECT_TRUE(containsEdge(meta.backEdges, 2, 1));

    // Exit: 3 -> 4 exits the SCC
    EXPECT_EQ(meta.exitEdges.size(), 1);
    EXPECT_TRUE(containsEdge(meta.exitEdges, 3, 4));
  }
}

TEST(SCCBuilderTest, DetectsMultipeLoops) {
  std::pmr::monotonic_buffer_resource pool(2048);

  MockRegionBuilder regions({{1}, {2, 4}, {3}, {1}, {5, 7}, {6}, {4}, {}});

  auto result = compiler::analysis::SCCBuilder<MockRegionBuilder>(&pool, regions).calculate(0);
  compiler::analysis::debug::dump(std::cout, result);

  EXPECT_EQ(result.get().size(), 2);
  EXPECT_TRUE(containsComponent(result, {1, 2, 3}));
  EXPECT_TRUE(containsComponent(result, {4, 5, 6}));

  // Find and test first SCC {1, 2, 3}
  for (auto const& nodes: result.get()) {
    std::vector<int32_t> sortedNodes(nodes.begin(), nodes.end());
    std::ranges::sort(sortedNodes);

    if (std::ranges::equal(sortedNodes, std::vector<int32_t> {1, 2, 3})) {
      auto meta = compiler::analysis::classifySCC(&pool, regions, nodes);

      // Entry: 0 -> 1
      EXPECT_EQ(meta.entryEdges.size(), 1);
      EXPECT_TRUE(containsEdge(meta.entryEdges, 0, 1));

      // Back edge: 3 -> 1
      EXPECT_EQ(meta.backEdges.size(), 1);
      EXPECT_TRUE(containsEdge(meta.backEdges, 3, 1));

      // Exit: 1 -> 4
      EXPECT_EQ(meta.exitEdges.size(), 1);
      EXPECT_TRUE(containsEdge(meta.exitEdges, 1, 4));
    } else if (std::ranges::equal(sortedNodes, std::vector<int32_t> {4, 5, 6})) {
      auto meta = compiler::analysis::classifySCC(&pool, regions, nodes);

      // Entry: 1 -> 4 (from previous SCC)
      EXPECT_EQ(meta.entryEdges.size(), 1);
      EXPECT_TRUE(containsEdge(meta.entryEdges, 1, 4));

      // Back edge: 6 -> 4
      EXPECT_EQ(meta.backEdges.size(), 1);
      EXPECT_TRUE(containsEdge(meta.backEdges, 6, 4));

      // Exit: 4 -> 7
      EXPECT_EQ(meta.exitEdges.size(), 1);
      EXPECT_TRUE(containsEdge(meta.exitEdges, 4, 7));
    }
  }
}