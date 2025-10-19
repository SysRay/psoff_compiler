#include "frontend/analysis/regions.h"

#include <gtest/gtest.h>

class RegionBuilderTest: public ::testing::Test {
  protected:
  std::pmr::monotonic_buffer_resource resource;
  std::pmr::polymorphic_allocator<>   allocator {&resource};

  void SetUp() override {}

  void TearDown() override {}
};

using namespace compiler::frontend::analysis;

TEST_F(RegionBuilderTest, InitialRegionCreation) {
  RegionBuilder builder(100, allocator);

  EXPECT_EQ(builder.getNumRegions(), 1);

  auto [start, end] = builder.getRegion(0);
  EXPECT_EQ(start, 0);
  EXPECT_EQ(end, 100);
}

static bool isEqual(auto const& result, std::initializer_list<int32_t> expected) {
  if (expected.size() == 0 && result.size() == 0) return true;
  return std::ranges::any_of(result, [&](auto const& comp) { return std::ranges::equal(result, expected); });
}

static std::vector<regionid_t> collectPreds(RegionBuilder const& builder, regionid_t region) {
  std::vector<regionid_t> result;
  builder.visitPredecessors(region, [&result](regionid_t pred) { result.push_back(pred); });

  std::ranges::sort(result); // Helping with equal check
  return result;
}

TEST_F(RegionBuilderTest, SimpleJumpSplitsRegions) {
  RegionBuilder builder(100, allocator);

  builder.addJump(9, 50);
  // Regions: regions: [0,10), [10,50), [50,100)

  auto const& regions = builder.getRegions();

  EXPECT_EQ(regions.size(), 3);

  EXPECT_EQ(regions[0].start, 0);
  EXPECT_EQ(regions[0].end, 10);

  EXPECT_EQ(regions[1].start, 10);
  EXPECT_EQ(regions[1].end, 50);

  EXPECT_EQ(regions[2].start, 50);
  EXPECT_EQ(regions[2].end, 100);

  EXPECT_TRUE(isEqual(builder.getSuccessorsIdx(0), {2}));
  EXPECT_TRUE(isEqual(builder.getSuccessorsIdx(1), {2}));
  EXPECT_TRUE(isEqual(builder.getSuccessorsIdx(2), {}));
}

TEST_F(RegionBuilderTest, ConditionalJumpCreatesMultipleSuccessors) {
  RegionBuilder builder(100, allocator);

  builder.addCondJump(9, 50);
  // Regions: regions: [0,10), [10,50), [50,100)
  // Region [0,10) should have two successors: fallthrough to [10,50) and jump to [50,100)

  auto const& regions = builder.getRegions();

  EXPECT_EQ(regions.size(), 3);

  // Check region boundaries
  EXPECT_EQ(regions[0].start, 0);
  EXPECT_EQ(regions[0].end, 10);

  EXPECT_EQ(regions[1].start, 10);
  EXPECT_EQ(regions[1].end, 50);

  EXPECT_EQ(regions[2].start, 50);
  EXPECT_EQ(regions[2].end, 100);

  // Conditional jump: should have both successors
  EXPECT_TRUE(isEqual(builder.getSuccessorsIdx(0), {1, 2}));
  EXPECT_TRUE(isEqual(builder.getSuccessorsIdx(1), {2}));
  EXPECT_TRUE(isEqual(builder.getSuccessorsIdx(2), {}));
}

TEST_F(RegionBuilderTest, ReturnStopsFlow) {
  RegionBuilder builder(100, allocator);

  builder.addReturn(49);
  // Regions: regions: [0,50), [50,100)

  auto const& regions = builder.getRegions();

  EXPECT_EQ(regions.size(), 2);

  EXPECT_EQ(regions[0].start, 0);
  EXPECT_EQ(regions[0].end, 50);

  EXPECT_EQ(regions[1].start, 50);
  EXPECT_EQ(regions[1].end, 100);

  // Return means no successors
  EXPECT_TRUE(isEqual(builder.getSuccessorsIdx(0), {}));
  EXPECT_TRUE(isEqual(builder.getSuccessorsIdx(1), {}));
}

TEST_F(RegionBuilderTest, MultipleJumps) {
  RegionBuilder builder(100, allocator);

  builder.addJump(9, 30);
  builder.addJump(29, 70);
  builder.addReturn(99);
  // Regions: [0,10), [10,30), [30,70), [70,100)

  auto const& regions = builder.getRegions();

  EXPECT_EQ(regions.size(), 4);

  EXPECT_EQ(regions[0].start, 0);
  EXPECT_EQ(regions[0].end, 10);

  EXPECT_EQ(regions[1].start, 10);
  EXPECT_EQ(regions[1].end, 30);

  EXPECT_EQ(regions[2].start, 30);
  EXPECT_EQ(regions[2].end, 70);

  EXPECT_EQ(regions[3].start, 70);
  EXPECT_EQ(regions[3].end, 100);

  EXPECT_TRUE(isEqual(builder.getSuccessorsIdx(0), {2}));
  EXPECT_TRUE(isEqual(builder.getSuccessorsIdx(1), {3}));
  EXPECT_TRUE(isEqual(builder.getSuccessorsIdx(2), {3}));
  EXPECT_TRUE(isEqual(builder.getSuccessorsIdx(3), {}));

  // using region_id
  EXPECT_TRUE(isEqual(collectPreds(builder, 0), {}));
  EXPECT_TRUE(isEqual(collectPreds(builder, 30), {0}));
  EXPECT_TRUE(isEqual(collectPreds(builder, 70), {10, 30}));
}

TEST_F(RegionBuilderTest, BackwardJumpCreatesLoop) {
  RegionBuilder builder(100, allocator);

  builder.addJump(49, 10);
  builder.addJump(99, 10);
  // Regions: [0,10), [10,50), [50,100)
  // Region [10,50) jumps back to [10,50)

  auto const& regions = builder.getRegions();

  EXPECT_EQ(regions.size(), 3);

  EXPECT_EQ(regions[0].start, 0);
  EXPECT_EQ(regions[0].end, 10);

  EXPECT_EQ(regions[1].start, 10);
  EXPECT_EQ(regions[1].end, 50);

  EXPECT_EQ(regions[2].start, 50);
  EXPECT_EQ(regions[2].end, 100);

  EXPECT_TRUE(isEqual(builder.getSuccessorsIdx(0), {1}));
  EXPECT_TRUE(isEqual(builder.getSuccessorsIdx(1), {1})); // Loop to self
  EXPECT_TRUE(isEqual(builder.getSuccessorsIdx(2), {1}));
}

TEST_F(RegionBuilderTest, ComplexControlFlow) {
  RegionBuilder builder(100, allocator);

  builder.addCondJump(9, 50);  // [0,10) -> [10,50) and [50,100)
  builder.addJump(29, 70);     // [10,30) -> [70,100)
  builder.addCondJump(49, 20); // [30,50) -> [20,30) and [50,100)

  auto const& regions = builder.getRegions();

  EXPECT_GE(regions.size(), 4); // At least 4 regions created

  // Verify basic structure exists
  EXPECT_EQ(regions[0].start, 0);
}

TEST_F(RegionBuilderTest, VisitSuccessors) {
  RegionBuilder builder(100, allocator);

  builder.addCondJump(9, 50);

  std::vector<regionid_t> successors;
  builder.visitSuccessors(0, [&successors](regionid_t succ) { successors.push_back(succ); });

  EXPECT_EQ(successors.size(), 2);
  EXPECT_TRUE(std::ranges::find(successors, 10) != successors.end());
  EXPECT_TRUE(std::ranges::find(successors, 50) != successors.end());
}

TEST_F(RegionBuilderTest, VisitPredecessors) {
  RegionBuilder builder(100, allocator);

  builder.addJump(9, 50);
  builder.addCondJump(29, 50);

  std::vector<regionid_t> predecessors;
  builder.visitPredecessors(50, [&predecessors](regionid_t pred) { predecessors.push_back(pred); });

  // Both region starting at 0 and region starting at 10 can reach 50
  EXPECT_GE(predecessors.size(), 1);
}

TEST_F(RegionBuilderTest, FindRegionByIndex) {
  RegionBuilder builder(100, allocator);

  builder.addJump(9, 50);

  auto [region_start, region_end] = builder.findRegion(5);
  EXPECT_EQ(region_start, 0);
  EXPECT_EQ(region_end, 10);

  auto [region_start2, region_end2] = builder.findRegion(25);
  EXPECT_EQ(region_start2, 10);
  EXPECT_EQ(region_end2, 50);

  auto [region_start3, region_end3] = builder.findRegion(75);
  EXPECT_EQ(region_start3, 50);
  EXPECT_EQ(region_end3, 100);
}

TEST_F(RegionBuilderTest, JumpToSameRegion) {
  RegionBuilder builder(100, allocator);

  builder.addJump(20, 30);

  auto const& regions = builder.getRegions();

  EXPECT_EQ(regions.size(), 3);
  EXPECT_EQ(regions[0].start, 0);
  EXPECT_EQ(regions[0].end, 21);
}

TEST_F(RegionBuilderTest, EdgeCaseJumpToEnd) {
  RegionBuilder builder(100, allocator);

  builder.addJump(49, 100);

  auto const& regions = builder.getRegions();

  EXPECT_EQ(regions.size(), 2);
  EXPECT_EQ(regions[0].start, 0);
  EXPECT_EQ(regions[0].end, 50);
  EXPECT_EQ(regions[1].start, 50);
  EXPECT_EQ(regions[1].end, 100);
}