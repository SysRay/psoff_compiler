#include "compiler_ctx.h"
#include "mlir/custom.h"
#include "mlir/passes/psoff_passes.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

TEST(PromoteRegs, Simple) {
  compiler::CompilerCtx ctx {};

  static constexpr std::string_view sInputModule = R"(
  func.func @main() {
    %cst2 = arith.constant 1 : i64
    psoff.StoreOp EXEC = %cst2 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst1 = arith.constant 1.000000e+00 : f32
    %3 = arith.cmpf one, %cst, %cst1 : f32
    psoff.StoreOp VCC_LO = %3 : i1
    %4 = psoff.LoadOp EXEC : i64
    %5 = psoff.LoadOp VCC : i64
    %6 = arith.andi %4, %5 : i64
    psoff.StoreOp VCC = %4 : i64
    psoff.StoreOp EXEC = %6 : i64
}
)";

  auto inputModule = mlir::parseSourceString<mlir::ModuleOp>(sInputModule, ctx.getContext());
  ASSERT_TRUE(inputModule);

  mlir::PassManager pm(ctx.getContext());
  pm.enableVerifier(false);
  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<mlir::psoff::PromoteRegisterPass>());

  EXPECT_FALSE(failed(pm.run(inputModule.get())));

  inputModule->dump();
}