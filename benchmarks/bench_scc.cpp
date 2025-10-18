

#include "analysis/scc.h"
#include "builder.h"
#include "fixed_containers/fixed_vector.hpp"

#include <benchmark/benchmark.h>
#include <random>

// ---------------- Mock RegionBuilder ----------------
struct MockRegionBuilder {
  std::vector<fixed_containers::FixedVector<int32_t, 2>> edges;

  int32_t getNumRegions() const { return static_cast<int32_t>(edges.size()); }

  fixed_containers::FixedVector<int32_t, 2> getSuccessorsIdx(uint32_t idx) const { return edges[idx]; }
};

static MockRegionBuilder makeGraph(std::size_t num_nodes, std::size_t outer_loop_size = 20, std::size_t inner_loop_size = 4, unsigned int seed = 42) {
  MockRegionBuilder regions;
  regions.edges.resize(num_nodes);
  std::mt19937                       rng(seed);
  std::uniform_int_distribution<int> dist(0, static_cast<int>(num_nodes - 1));

  auto add_edge = [&](std::size_t from, std::size_t to) {
    if (from != to && regions.edges[from].size() < 2) regions.edges[from].push_back(static_cast<int32_t>(to));
  };

  // Outer loops
  for (std::size_t base = 0; base + outer_loop_size < num_nodes; base += outer_loop_size * 2) {
    for (std::size_t i = 0; i < outer_loop_size; ++i) {
      std::size_t from = base + i;
      std::size_t to   = (i + 1 < outer_loop_size) ? base + i + 1 : base;
      add_edge(from, to);
    }
  }

  // Nested inner loops (loop inside loop)
  for (std::size_t base = 0; base + inner_loop_size < num_nodes; base += 50) {
    for (std::size_t i = base; i < base + inner_loop_size; ++i) {
      add_edge(i, i + 1);
      add_edge(i + 1, i); // inner 2-way loop
    }
  }

  // Controlled random cross-edges
  std::size_t attempts = num_nodes / 3;
  for (std::size_t i = 0; i < attempts; ++i) {
    int from = dist(rng);
    int to   = dist(rng);
    add_edge(from, to);
  }

  // Long chain ending in a loop
  for (std::size_t i = num_nodes / 2; i + 1 < num_nodes; ++i)
    add_edge(i, i + 1);
  add_edge(num_nodes - 1, num_nodes / 2); // close loop

  return regions;
}

static void BM_SCCBuilder_Calculate(benchmark::State& state) {
  auto regions = makeGraph(16384, /*loop_size=*/20);

  std::pmr::monotonic_buffer_resource pool(100_MB);

  for (auto _: state) {
    std::pmr::monotonic_buffer_resource checkpoint(&pool);
    auto                                result = compiler::analysis::SCCBuilder<MockRegionBuilder>(&checkpoint, regions).calculate();
    benchmark::DoNotOptimize(result);
  }

  state.SetComplexityN(16384);
}

BENCHMARK(BM_SCCBuilder_Calculate)
    ->RangeMultiplier(2)
    ->MinWarmUpTime(0.05)
    ->MinTime(1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();