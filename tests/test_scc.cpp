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

  int32_t getNumRegions() const { return static_cast<int32_t>(edges.size()); }

  fixed_containers::FixedVector<int32_t, 2> getSuccessorsIdx(uint32_t idx) const { return edges[idx]; }
};

static bool containsComponent(compiler::analysis::scc_t const& result, std::initializer_list<int32_t> expected) {
  std::vector<int32_t> sortedExpected(expected);
  std::ranges::sort(sortedExpected);

  return std::ranges::any_of(result, [&](auto const& comp) { return std::ranges::equal(comp, sortedExpected); });
}

TEST(SCCBuilderTest, DetectsLoops) {
  std::pmr::monotonic_buffer_resource pool(1024);

  MockRegionBuilder regions {.edges = {
                                 {1, 3}, // 0 -> 1
                                 {2},    // 1 -> 2
                                 {0},    // 2 -> 0 (back edge to form loop)
                                 {4},    // 3 -> 4
                                 {}      // 4
                             }};

  auto result = compiler::analysis::SCCBuilder<MockRegionBuilder>(&pool, regions).calculate();

  EXPECT_TRUE(containsComponent(result, {0, 1, 2}));
  EXPECT_EQ(result.size(), 3); // 1 loop + 2 singletons
}

TEST(SCCBuilderTest, DetectsNestedLoops) {
  std::pmr::monotonic_buffer_resource pool(2048);

  MockRegionBuilder regions {.edges = {
                                 {1},    // 0 -> 1
                                 {0, 2}, // 1 -> 0 (outer back edge), and 1 -> 2 (inner loop entry)
                                 {1}     // 2 -> 1 (inner back edge)
                             }};

  auto result = compiler::analysis::SCCBuilder<MockRegionBuilder>(&pool, regions).calculate();

  // In Tarjan’s SCC, nested loops that are connected become one SCC:
  // {0, 1, 2}
  EXPECT_TRUE(containsComponent(result, {0, 1, 2}));
  EXPECT_EQ(result.size(), 1);
}

TEST(SCCBuilderTest, DetectsNestedLoopStructure) {
  std::pmr::monotonic_buffer_resource pool(2048);

  MockRegionBuilder regions {.edges = {{1, 4}, // 0 -> 1
                                       {2},    // 1 -> 2
                                       {1, 3}, // 2 -> 1 (inner loop back-edge), 2 -> 3
                                       {0},    // 3 -> 0 (outer loop back-edge)
                                       {}}};

  auto result = compiler::analysis::SCCBuilder<MockRegionBuilder>(&pool, regions).calculate();

  EXPECT_TRUE(containsComponent(result, {0, 1, 2, 3}));
  EXPECT_TRUE(containsComponent(result, {4}));
  EXPECT_EQ(result.size(), 2);
}

TEST(SCCBuilderTest, LoopCallsAnotherLoop) {
  std::pmr::monotonic_buffer_resource pool(2048);

  MockRegionBuilder regions {.edges = {
                                 {1},    // 0 -> 1
                                 {0, 2}, // 1 -> 0, 1 -> 2 (call into loop B)
                                 {3},    // 2 -> 3
                                 {2}     // 3 -> 2
                             }};

  auto result = compiler::analysis::SCCBuilder<MockRegionBuilder>(&pool, regions).calculate();

  EXPECT_TRUE(containsComponent(result, {0, 1}));
  EXPECT_TRUE(containsComponent(result, {2, 3}));
  EXPECT_EQ(result.size(), 2);
}
