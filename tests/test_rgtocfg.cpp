#include "builder.h"
#include "frontend/transform/transform.h"
#include "mlir/custom.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

class RGToCF: public ::testing::Test {
  protected:
  void SetUp() override {

    mlir::OpBuilder builder(_builder.getContext());

    _mlirModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(_builder.getContext()));

    auto funcOp =
        builder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(_builder.getContext()), "test_func", builder.getFunctionType({builder.getI1Type()}, {}));
    _mlirModule.push_back(funcOp);

    _block = funcOp.addEntryBlock();
  }

  void TearDown() override {}

  compiler::Builder _builder {};

  mlir::Block*   _block;
  mlir::ModuleOp _mlirModule;
};

TEST_F(RGToCF, SimpleIfElse) {
  mlir::OpBuilder builder(_builder.getContext());
  builder.setInsertionPointToEnd(_block);

  using namespace compiler::frontend;
  std::vector<pcmapping_t> pcMappings;

  auto loc      = mlir::UnknownLoc::get(_builder.getContext());
  auto intConst = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64Type(), builder.getI32IntegerAttr(40));
  builder.create<mlir::psoff::Branch>(loc, intConst);
  pcMappings.emplace_back(10, &_block->back());

  _mlirModule.dump();
  // transform
  transform::transformRg2Cfg(_builder, pcMappings);
}
