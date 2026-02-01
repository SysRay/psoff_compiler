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

    _func = builder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(_builder.getContext()), "test_func", builder.getFunctionType({builder.getI1Type()}, {}));
    _mlirModule.push_back(_func);

    _block = _func.addEntryBlock();
  }

  void TearDown() override {}

  compiler::Builder _builder {};

  mlir::Block*   _block;
  mlir::ModuleOp _mlirModule;

  mlir::func::FuncOp _func;
};

TEST_F(RGToCF, SimpleIfElse) {
  mlir::OpBuilder builder(_builder.getContext());
  builder.setInsertionPointToEnd(_block);

  using namespace compiler::frontend;
  std::vector<pcmapping_t> pcMappings;

  auto loc = mlir::UnknownLoc::get(_builder.getContext());
  builder.create<mlir::psoff::Branch>(loc, builder.getI64IntegerAttr(40));
  pcMappings.emplace_back(10, &_block->back());

  builder.create<mlir::func::ReturnOp>(loc);
  pcMappings.emplace_back(40, &_block->back());
  _mlirModule.dump();

  // transform
  transform::transformRg2Cfg(_builder, _func, pcMappings);
}
