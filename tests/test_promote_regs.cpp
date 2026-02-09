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
    %cst = arith.constant 0.000000e+00 : f32
    %2 = psoff.LoadOp v[4] : f32
    %3 = arith.cmpf one, %cst, %2 : f32
    psoff.StoreOp VCC_LO = %3 : i1
    %4 = psoff.LoadOp EXEC : i64
    %5 = psoff.LoadOp VCC : i64
    %6 = arith.andi %4, %5 : i64
    psoff.StoreOp VCC = %4 : i64
    psoff.StoreOp EXEC = %6 : i64
    %7 = psoff.LoadOp EXEC_LO : i1
    cf.cond_br %7, ^bb2, ^bb3
 ^bb2:
    cf.br ^bb3
 ^bb3:
    return
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