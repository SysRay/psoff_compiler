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

  int32_t size() const { return static_cast<int32_t>(edges.size()); }

  auto getSuccessors(uint32_t idx) const { return edges[idx]; }

  auto getPredecessors(uint32_t idx) const {
    assert(false); // not needed
    return edges[idx];
  }
};

static bool containsComponent(compiler::analysis::SCC const& result, std::initializer_list<int32_t> expected) {
  std::vector<int32_t> sortedExpected(expected);
  std::ranges::sort(sortedExpected);

  return std::ranges::any_of(result.get(), [&](auto const& comp) { return std::ranges::equal(comp, sortedExpected); });
}

TEST(SCCBuilderTest, DetectsSelf) {
  std::pmr::monotonic_buffer_resource pool(2048);

  MockRegionBuilder regions {.edges = {{1}, {2, 1}, {3}, {}}};

  auto result = compiler::analysis::SCCBuilder<MockRegionBuilder>(&pool, regions).calculate();
  compiler::analysis::debug::dump(std::cout, result);

  EXPECT_EQ(result.get().size(), 1);
  EXPECT_TRUE(containsComponent(result, {1}));

  {
    auto const& nodes = result.get()[0];

    std::pmr::monotonic_buffer_resource checkpoint(&pool);

    auto meta = compiler::analysis::classifySCC(&pool, regions, nodes);

    compiler::analysis::debug::dump(std::cout, meta);
  }
}

TEST(SCCBuilderTest, DetectsNoExit) {
  std::pmr::monotonic_buffer_resource pool(2048);

  MockRegionBuilder regions {.edges = {{1}, {2, 4}, {3}, {2}, {}}};

  auto result = compiler::analysis::SCCBuilder<MockRegionBuilder>(&pool, regions).calculate();
  compiler::analysis::debug::dump(std::cout, result);

  EXPECT_EQ(result.get().size(), 1);
  EXPECT_TRUE(containsComponent(result, {2, 3}));

  {
    auto const& nodes = result.get()[0];

    std::pmr::monotonic_buffer_resource checkpoint(&pool);

    auto meta = compiler::analysis::classifySCC(&pool, regions, nodes);

    compiler::analysis::debug::dump(std::cout, meta);
  }
}

TEST(SCCBuilderTest, DetectsNoStart) {
  std::pmr::monotonic_buffer_resource pool(2048);

  MockRegionBuilder regions {.edges = {{4}, {2, 4}, {3}, {1}, {}}};

  auto result = compiler::analysis::SCCBuilder<MockRegionBuilder>(&pool, regions).calculate();
  compiler::analysis::debug::dump(std::cout, result);

  EXPECT_EQ(result.get().size(), 1);
  EXPECT_TRUE(containsComponent(result, {1, 2, 3}));

  {
    auto const& nodes = result.get()[0];

    std::pmr::monotonic_buffer_resource checkpoint(&pool);

    auto meta = compiler::analysis::classifySCC(&pool, regions, nodes);

    compiler::analysis::debug::dump(std::cout, meta);
  }
}

TEST(SCCBuilderTest, DetectsSimpleLoopHead) {
  std::pmr::monotonic_buffer_resource pool(2048);

  MockRegionBuilder regions {.edges = {{1}, {2, 4}, {3}, {1}, {}}};

  auto result = compiler::analysis::SCCBuilder<MockRegionBuilder>(&pool, regions).calculate();
  compiler::analysis::debug::dump(std::cout, result);

  EXPECT_EQ(result.get().size(), 1);
  EXPECT_TRUE(containsComponent(result, {1, 2, 3}));

  {
    auto const& nodes = result.get()[0];

    std::pmr::monotonic_buffer_resource checkpoint(&pool);

    auto meta = compiler::analysis::classifySCC(&pool, regions, nodes);

    compiler::analysis::debug::dump(std::cout, meta);
  }
}

TEST(SCCBuilderTest, DetectsSimpleLoopTail) {
  std::pmr::monotonic_buffer_resource pool(2048);

  MockRegionBuilder regions {.edges = {{1}, {2}, {3}, {1, 4}, {}}};

  auto result = compiler::analysis::SCCBuilder<MockRegionBuilder>(&pool, regions).calculate();
  compiler::analysis::debug::dump(std::cout, result);

  EXPECT_EQ(result.get().size(), 1);
  EXPECT_TRUE(containsComponent(result, {1, 2, 3}));

  {
    auto const& nodes = result.get()[0];

    std::pmr::monotonic_buffer_resource checkpoint(&pool);

    auto meta = compiler::analysis::classifySCC(&pool, regions, nodes);

    compiler::analysis::debug::dump(std::cout, meta);
  }
}

TEST(SCCBuilderTest, DetectsNestedLoops) {
  std::pmr::monotonic_buffer_resource pool(2048);

  MockRegionBuilder regions {.edges = {{1},    // 0 -> 1
                                       {2},    // 1 -> 2
                                       {1, 3}, // 2 -> 1,3
                                       {2, 4}, // 3 -> 2,4
                                       {}}};

  auto result = compiler::analysis::SCCBuilder<MockRegionBuilder>(&pool, regions).calculate();

  compiler::analysis::debug::dump(std::cout, result);

  EXPECT_EQ(result.get().size(), 1);
  EXPECT_TRUE(containsComponent(result, {1, 2, 3}));

  for (auto const& nodes: result.get()) {
    std::pmr::monotonic_buffer_resource checkpoint(&pool);

    auto meta = compiler::analysis::classifySCC(&pool, regions, nodes);

    compiler::analysis::debug::dump(std::cout, meta);
  }
}

TEST(SCCBuilderTest, DetectsMultipeLoops) {
  std::pmr::monotonic_buffer_resource pool(2048);

  MockRegionBuilder regions {.edges = {{1}, {2, 4}, {3}, {1}, {5, 7}, {6}, {4}, {}}};

  auto result = compiler::analysis::SCCBuilder<MockRegionBuilder>(&pool, regions).calculate();
  compiler::analysis::debug::dump(std::cout, result);

  EXPECT_EQ(result.get().size(), 2);
  EXPECT_TRUE(containsComponent(result, {1, 2, 3}));
  EXPECT_TRUE(containsComponent(result, {5, 6}));

  {
    auto const& nodes = result.get()[0];

    std::pmr::monotonic_buffer_resource checkpoint(&pool);

    auto meta = compiler::analysis::classifySCC(&pool, regions, nodes);

    compiler::analysis::debug::dump(std::cout, meta);
  }
}
