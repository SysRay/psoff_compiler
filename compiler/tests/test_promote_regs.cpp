#include "compiler_ctx.h"
#include "mlir/custom.h"
#include "mlir/passes/psoff_passes.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

TEST(PromoteRegs, Simple) {
  compiler::ShaderBuildFeatures features {};
  compiler::CompilerCtx         ctx {features};

  static constexpr std::string_view sInputModule = R"(
  func.func @main(%arg0: f32) -> i1 {
    psoff.store s[4] = %arg0 : f32
    %cst = arith.constant 0.000000e+00 : f32
    %2 = psoff.load s[4] : f32
    %3 = arith.cmpf one, %cst, %2 : f32
    return %3 : i1
  }
)";

  auto inputModule = mlir::parseSourceString<mlir::ModuleOp>(sInputModule, ctx.getContext());
  ASSERT_TRUE(inputModule);

  mlir::PassManager pm(ctx.getContext());
  pm.enableVerifier(false);
  pm.addNestedPass<mlir::func::FuncOp>(mlir::psoff::createRegisterSSAPass(ctx.allocator()));
  pm.addPass(mlir::createRemoveDeadValuesPass());
  // pm.addPass(mlir::createCSEPass());
  EXPECT_FALSE(failed(pm.run(inputModule.get())));

  inputModule->dump();
}

TEST(PromoteRegs, SimpleIf) {
  compiler::ShaderBuildFeatures features {};
  compiler::CompilerCtx         ctx {features};

  static constexpr std::string_view sInputModule = R"(
func.func @simpleIf(%arg0: f32, %arg1: i1) -> i1 {
  psoff.store s[0] = %arg1 : i1
  %1 = psoff.load s[0] : i1
  %cst = arith.constant 0.000000e+00 : f32
  %cst1 = arith.constant 1.000000e+00 : f32
  psoff.store s[4] = %cst : f32
  psoff.store v[2] = %cst1 : f32

  scf.if %1 {
    psoff.store s[4] = %arg0 : f32
  } else {
    %2 = psoff.load s[4] : f32
    psoff.store v[2] = %2 : f32
  }

  %3 = psoff.load s[4] : f32
  %4 = psoff.load v[2] : f32
  %5 = arith.cmpf one, %3, %4 : f32
  return %5 : i1
}
)";

  auto inputModule = mlir::parseSourceString<mlir::ModuleOp>(sInputModule, ctx.getContext());
  ASSERT_TRUE(inputModule);

  mlir::PassManager pm(ctx.getContext());
  pm.enableVerifier(false);
  pm.addNestedPass<mlir::func::FuncOp>(mlir::psoff::createRegisterSSAPass(ctx.allocator()));
  pm.addPass(mlir::createRemoveDeadValuesPass());
  // pm.addPass(mlir::createCSEPass());
  EXPECT_FALSE(failed(pm.run(inputModule.get())));

  inputModule->dump();
}