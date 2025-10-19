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

TEST_F(RegionBuilderTest, SimpleJumpSplitsRegions) {
  RegionBuilder builder(100, allocator);

  builder.addJump(9, 50);
  // Should create regions: [0,10), [10,50), [50,100)

  builder.for_each([&builder](regionid_t, uint32_t, void* region) { builder.dump(std::cout, region); });

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