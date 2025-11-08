#include "include/checkpoint_resource.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory_resource>
#include <ranges>
#include <vector>

TEST(PMRTest, revert) {
  std::array<uint8_t, 2048> data;
  std::fill(data.begin(), data.end(), 0xFF);

  compiler::util::checkpoint_resource pool(data.data(), 1024);
  std::pmr::polymorphic_allocator<>   alloc {&pool};

  std::pmr::vector<uint8_t> gVec(200, alloc);
  std::fill(gVec.begin(), gVec.end(), 0xab);

  EXPECT_EQ(0xab, gVec[199]);

  uint64_t pos0 = 0;
  {
    auto checkpoint = pool.checkpoint();

    std::pmr::polymorphic_allocator<> a {&pool};
    std::pmr::vector<uint8_t>         v(200, a); // allocate repeatedly}
    pos0 = (uint64_t)v.data();
  }
  {
    auto checkpoint = pool.checkpoint();

    std::pmr::polymorphic_allocator<> a {&pool};
    std::pmr::vector<uint8_t>         v(200, a); // allocate repeatedly}

    ASSERT_EQ(pos0, (uint64_t)v.data());
  }

  EXPECT_EQ(0xab, gVec[199]);
}
