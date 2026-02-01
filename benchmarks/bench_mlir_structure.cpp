
#include "fixed_containers/fixed_vector.hpp"

#include <benchmark/benchmark.h>
#include <random>
// mlir

#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

// ---------------- Mock RegionBuilder ----------------

struct MockRegionBuilder {
  std::vector<fixed_containers::FixedVector<int32_t, 2>> edges;

  int32_t size() const { return static_cast<int32_t>(edges.size()); }
};

static MockRegionBuilder makeGraph(std::size_t num_nodes, std::size_t outer_loop_size = 20, std::size_t inner_loop_size = 4, unsigned int seed = 42) {
  MockRegionBuilder regions;
  regions.edges.resize(num_nodes);

  for (std::size_t i = 0; i < regions.edges.size() - 1; ++i)
    regions.edges[i].push_back(i + 1);

  std::mt19937                       rng(seed);
  std::uniform_int_distribution<int> dist(0, static_cast<int>(num_nodes - 1));

  // ------------------------------------------------------------------
  // 2. Safe edge insertion
  // ------------------------------------------------------------------
  auto add_edge = [&](std::size_t from, std::size_t to) {
    if (from == to) return;

    auto& out = regions.edges[from];
    if (out.size() >= 2) return;

    if (std::find(out.begin(), out.end(), static_cast<int32_t>(to)) != out.end()) return;

    out.push_back(static_cast<int32_t>(to));
  };

  for (std::size_t base = 1; base + inner_loop_size < num_nodes; base += 50) {
    for (std::size_t i = base; i + 1 < base + inner_loop_size; ++i) {
      add_edge(i, i + 1);
      add_edge(i + 1, i);
    }
  }

  for (std::size_t base = 1; base + outer_loop_size < num_nodes; base += outer_loop_size * 2) {
    for (std::size_t i = 0; i < outer_loop_size; ++i) {
      std::size_t from = base + i;
      std::size_t to   = (i + 1 < outer_loop_size) ? base + i + 1 : base;
      add_edge(from, to);
    }
  }
  return regions;
}

static mlir::func::FuncOp createCFG(mlir::MLIRContext& ctx, MockRegionBuilder const& regions) {
  mlir::OpBuilder builder(&ctx);

  auto mlirModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  auto funcOp = builder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "test_func", builder.getFunctionType({builder.getI1Type()}, {}));
  mlirModule.push_back(funcOp);

  std::vector<mlir::Block*> blocks(regions.size());
  blocks[0] = funcOp.addEntryBlock();

  for (uint32_t n = 1; n < blocks.size(); ++n) {
    blocks[n] = builder.createBlock(&funcOp.getBody());
  }

  for (uint32_t n = 0; n < regions.size(); n++) {
    auto const& edge = regions.edges[n];
    builder.setInsertionPointToEnd(blocks[n]);
    if (edge.size() == 2) {
      builder.create<mlir::cf::CondBranchOp>(mlir::UnknownLoc::get(&ctx), funcOp.getArgument(0), blocks[edge[0]], blocks[edge[1]]);
    } else if (edge.size() == 1) {
      builder.create<mlir::cf::BranchOp>(mlir::UnknownLoc::get(&ctx), blocks[edge[0]]);
    } else {
      builder.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&ctx));
    }
  }

  return funcOp;
}

static void BM_SCCBuilder_Calculate(benchmark::State& state) {
  mlir::MLIRContext mlirCtx;
  mlirCtx.disableMultithreading();
  mlirCtx.allowUnregisteredDialects();
  mlirCtx.loadDialect<mlir::func::FuncDialect, mlir::scf::SCFDialect, mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect>();

  auto mlirModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(&mlirCtx));

  int N = 5000; // state.range(0);

  auto regions = makeGraph(N, /*loop_size=*/20);
  mlirModule.push_back(createCFG(mlirCtx, regions));

  for (auto _: state) {
    mlir::PassManager pm(&mlirCtx);
    pm.enableVerifier(false);
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createLiftControlFlowToSCFPass());

    if (failed(pm.run(mlirModule))) {
      // printf("Failed to lower psoff\n");
    }
  }

  state.SetComplexityN(N);
}

BENCHMARK(BM_SCCBuilder_Calculate)->RangeMultiplier(2)->MinWarmUpTime(0.05)->MinTime(1)->Unit(benchmark::kMillisecond);