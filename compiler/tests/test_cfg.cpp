#include "compiler_ctx.h"
#include "frontend/parser.h"
#include "mlir/custom.h"
#include "mlir/passes/psoff_passes.h"
#include "shaders.h"
#include "util/bump_allocator.h"

#include <gtest/gtest.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

class ControlFlow: public ::testing::Test {
  protected:
  void SetUp() override {

    mlir::OpBuilder builder(_ctx.getContext());

    _mlirModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(_ctx.getContext()));

    _func = builder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(_ctx.getContext()), "test_func", builder.getFunctionType({}, {}));
    _mlirModule.push_back(_func);

    _block = _func.addEntryBlock();
  }

  void TearDown() override {
    // const ::testing::TestInfo* test_info = ::testing::UnitTest::GetInstance()->current_test_info();

    // std::error_code      ec;
    // llvm::raw_fd_ostream out(std::string(test_info->name()) + ".txt", ec);

    // if (ec) {
    //   llvm::errs() << "Failed to open file: " << ec.message() << "\n";
    //   return;
    // }

    // _mlirModule.print(out);
  }

  compiler::ShaderBuildFeatures features {};
  compiler::CompilerCtx         _ctx {features};

  mlir::Block*   _block;
  mlir::ModuleOp _mlirModule;

  mlir::func::FuncOp _func;
};

TEST_F(ControlFlow, SimpleIfElse) {
  using namespace compiler::frontend;
  compiler::util::BumpAllocator allocator;

  Parser parser(_ctx, &allocator);

  static auto const binary = shader_ps_exec_ifelse;
  _ctx.setHostMapping(0, binary.data(), binary.size());

  mlir::OpBuilder mlirBuilder(_ctx.getContext());

  auto block = parser.getOrCreateBlock(0, &_func.getBody());

  mlirBuilder.setInsertionPointToStart(_block);
  mlirBuilder.create<mlir::cf::BranchOp>(mlir::UnknownLoc::get(_ctx.getContext()), block->mlirBlock);

  parser.process();

  mlir::PassManager pm(_ctx.getContext());
  // pm.enableVerifier(false);
  pm.addPass(mlir::createLiftControlFlowToSCFPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::psoff::createRegisterSSAPass(_ctx.allocator()));
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createRemoveDeadValuesPass());

  EXPECT_TRUE(succeeded(pm.run(_mlirModule)));
}

TEST_F(ControlFlow, Forloop) {
  using namespace compiler::frontend;
  compiler::util::BumpAllocator allocator;

  Parser parser(_ctx, &allocator);

  static auto const binary = shader_ps_forloop;
  _ctx.setHostMapping(0, binary.data(), binary.size());

  mlir::OpBuilder mlirBuilder(_ctx.getContext());

  auto block = parser.getOrCreateBlock(0, &_func.getBody());

  mlirBuilder.setInsertionPointToStart(_block);
  mlirBuilder.create<mlir::cf::BranchOp>(mlir::UnknownLoc::get(_ctx.getContext()), block->mlirBlock);

  parser.process();

  mlir::PassManager pm(_ctx.getContext());
  // pm.enableVerifier(false);

  pm.addPass(mlir::createLiftControlFlowToSCFPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::psoff::createRegisterSSAPass(_ctx.allocator()));
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createRemoveDeadValuesPass());

  EXPECT_TRUE(succeeded(pm.run(_func)));
}