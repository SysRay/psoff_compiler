

#include "builder.h"
#include "frontend/analysis/regions.h"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>

using namespace compiler::frontend::analysis;

static void inline test(std::pmr::memory_resource* pool, uint32_t N) {
  RegionBuilder builder(N, pool);

  std::mt19937_64 rng(42); // Fixed seed for reproducibility

  // 1. Sequential jumps (straight-line code)
  for (regionid_t i = 9; i < N / 10; i += 10) {
    builder.addJump(i, i + 11);
  }

  // 2. Conditional branches (if-else patterns)
  for (regionid_t i = N / 10; i < N / 5 && i < N - 20; i += 20) {
    builder.addCondJump(i, i + 15);
  }

  // 3. Simple loops (backward jumps)
  for (regionid_t i = N / 5 + 50; i < 2 * N / 5 && i < N - 50; i += 100) {
    builder.addCondJump(i + 40, i); // Loop back
  }

  // 4. Nested loops
  for (regionid_t i = 2 * N / 5 + 100; i < 3 * N / 5 && i < N - 100; i += 200) {
    builder.addCondJump(i + 80, i);      // Outer loop back
    builder.addCondJump(i + 40, i + 20); // Inner loop back
  }

  // 5. Complex control flow with mixed patterns
  for (regionid_t i = 3 * N / 5; i < 4 * N / 5 && i < N - 50; i += 50) {
    builder.addCondJump(i + 10, i + 30); // Conditional branch
    builder.addJump(i + 20, i + 40);     // Jump forward

    if (i % 100 == 0) {
      builder.addCondJump(i + 45, std::max(0u, i - 30)); // Loop back
    }
  }

  // 6. Returns scattered throughout
  for (regionid_t i = 4 * N / 5 + 49; i < 9 * N / 10 && i < N - 10; i += 100) {
    builder.addCondJump(i - 20, i + 20);
    builder.addReturn(i);
  }

  // 7. Dense jumps (many small regions)
  for (regionid_t i = 9 * N / 10 + 4; i < N - 5; i += 5) {
    if (i < N - 6) {
      builder.addJump(i, i + 6);
    }
  }

  // 8. Random jumps for realistic patterns
  std::uniform_int_distribution<regionid_t> pos_dist(0, N - 10);
  std::uniform_int_distribution<regionid_t> offset_dist(-50, 100);
  std::uniform_int_distribution<regionid_t> type_dist(0, 2);

  regionid_t num_random = N / 50; // 2% random jumps
  for (regionid_t i = 0; i < num_random; ++i) {
    regionid_t from = pos_dist(rng);
    regionid_t to   = std::clamp(from + offset_dist(rng), 0u, N - 1);

    regionid_t type = type_dist(rng);
    if (type == 0) {
      builder.addJump(from, to);
    } else if (type == 1) {
      builder.addCondJump(from, to);
    } else {
      builder.addReturn(from);
    }
  }

  // 9. Forward jumps (like switch statements)
  for (regionid_t i = 0; i < N / 20 && i < N / 2; i += 50) {
    if (N / 2 + i < N) {
      builder.addJump(i, N / 2 + i);
    }
  }

  // 10. Backward jumps (do-while loops)
  for (regionid_t i = 100; i < N - 10; i += 100) {
    builder.addCondJump(i, std::max(0u, i - 50));
  }
}

static void BM_RegionsBuilder_Calculate(benchmark::State& state) {
  std::pmr::monotonic_buffer_resource pool(100_MB);

  int N = state.range(0);
  for (auto _: state) {
    std::pmr::monotonic_buffer_resource checkpoint(&pool);
    test(&checkpoint, N);
  }

  state.SetComplexityN(N);
}

BENCHMARK(BM_RegionsBuilder_Calculate)->RangeMultiplier(2)->Range(60000, 120000)->MinWarmUpTime(0.05)->MinTime(1)->Unit(benchmark::kMillisecond);
