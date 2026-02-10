
#include "fixed_containers/fixed_vector.hpp"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory_resource>
#include <ranges>
#include <unordered_set>
#include <vector>
// mlir
#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <random>

class Restructure: public ::testing::Test {
  protected:
  void SetUp() override {
    _mlirCtx.disableMultithreading();
    _mlirCtx.allowUnregisteredDialects();

    _mlirCtx.loadDialect<mlir::func::FuncDialect, mlir::scf::SCFDialect, mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect>();
  }

  void TearDown() override {}

  mlir::MLIRContext _mlirCtx;
};

struct edge_t {
  uint32_t from;
  int32_t  to;
  int32_t  to2 = -1;
};

static void createCFG(mlir::MLIRContext& ctx, uint32_t numBlocks, std::initializer_list<edge_t> edges) {
  mlir::OpBuilder builder(&ctx);

  auto mlirModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  auto funcOp = builder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "test_func", builder.getFunctionType({builder.getI1Type()}, {}));
  mlirModule.push_back(funcOp);

  std::vector<mlir::Block*> blocks(numBlocks);
  blocks[0] = funcOp.addEntryBlock();

  for (uint32_t n = 1; n < blocks.size(); ++n) {
    blocks[n] = builder.createBlock(&funcOp.getBody());
  }

  for (auto const& edge: edges) {
    if (edge.to2 != -1) {
      builder.setInsertionPointToEnd(blocks[edge.from]);
      builder.create<mlir::cf::CondBranchOp>(mlir::UnknownLoc::get(&ctx), funcOp.getArgument(0), blocks[edge.to], blocks[edge.to2]);
    } else {
      builder.setInsertionPointToEnd(blocks[edge.from]);
      builder.create<mlir::cf::BranchOp>(mlir::UnknownLoc::get(&ctx), blocks[edge.to]);
    }
  }
  // mlir::Operation test;

  builder.setInsertionPointToEnd(blocks.back());
  builder.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&ctx));

  funcOp.dump();

  mlir::PassManager pm(&ctx);
  pm.enableVerifier(false);
  pm.addPass(mlir::createLiftControlFlowToSCFPass());

  if (failed(pm.run(mlirModule))) {
    printf("Failed to lower psoff\n");
    return;
  }

  funcOp.dump();
}

TEST_F(Restructure, SimpleIfElse) {
  //   0
  //   |
  //   1
  //  / \
  // 2   3
  // |   |
  // 4   |
  //  \ /
  //   5
  //   |
  //   6

  // std::pmr::monotonic_buffer_resource resource;
  // std::pmr::polymorphic_allocator<>   allocator {&resource};

  // rvsdg::IRBlocks blocks(allocator, 7);
  // ControlFlow     cfg(allocator, blocks);

  // createCFG(cfg, 7, 0, 6, {{0, 1}, {1, 2}, {1, 3}, {2, 4}, {4, 5}, {3, 5}, {5, 6}});
  // // createCFG(cfg, 5, 0, 4, {{0, 1}, {1, 2}, {1, 4}, {2, 4}});
  // debug::dump(std::cout, cfg);

  // std::array<uint8_t, 10000>          buffer;
  // compiler::util::checkpoint_resource tempResource(buffer.data(), buffer.size());
  // compiler::transform::createRVSDG(tempResource, cfg);
  // debug::dump(std::cout, blocks);
  // EXPECT_FALSE(true); // todo

  createCFG(_mlirCtx, 7, {{0, 1}, {1, 3, 2}, {2, 4}, {4, 5}, {3, 5}, {5, 6}});
}

TEST_F(Restructure, CompactIfElse) {
  //   0
  //   |
  //   1
  //  / \
  // 2   |
  //  \ /
  //   3
  //   |
  //   4

  createCFG(_mlirCtx, 5, {{0, 1}, {1, 3, 2}, {2, 3}, {3, 4}});
}

TEST_F(Restructure, GammaTwoCp) {
  //      0
  //      |
  //      1
  //     / \
  //    2   3
  //   / \  |
  //  4   5 |
  //  \    \|
  //   \    6
  //    \  /
  //      7

  createCFG(_mlirCtx, 8, {{0, 1}, {1, 3, 2}, {2, 5, 4}, {3, 6}, {4, 7}, {5, 6}, {6, 7}});
}

TEST_F(Restructure, SimpleWhileLoop) {
  //   0
  //   |
  //   1
  //  / \
  // 2   3 -> 1
  //  \
  //   4

  createCFG(_mlirCtx, 5, {{0, 1}, {1, 3, 2}, {2, 4}, {3, 1}});
}