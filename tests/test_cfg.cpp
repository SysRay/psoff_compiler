#include "builder.h"
#include "frontend/parser.h"
#include "mlir/custom.h"
#include "shaders.h"
#include "util/bump_allocator.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

class ControlFlow: public ::testing::Test {
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

TEST_F(ControlFlow, SimpleIfElse) {
  using namespace compiler::frontend;
  compiler::util::BumpAllocator allocator;

  compiler::Builder builder {};
  Parser            parser(builder, &allocator);

  static auto const binary = shader_ps_exec_ifelse;
  builder.setHostMapping(0, binary.data(), binary.size());

  parser.getOrCreateBlock(0);
  parser.process();

  // analysis::RegionBuilder       regions(&allocator);

  // auto loc = mlir::UnknownLoc::get(_builder.getContext());
  // builder.create<mlir::psoff::Branch>(loc, builder.getI64IntegerAttr(40));
  // pcMappings.emplace_back(10, &_block->back());
  // regions.addJump(10, 40);

  // builder.create<mlir::func::ReturnOp>(loc);
  // pcMappings.emplace_back(40, &_block->back());
  // regions.addReturn(40);
  // _mlirModule.dump();

  // // transform
  // regions.finalize();
  // analysis::createRegions(_builder, _func, regions, pcMappings);
}

TEST_F(ControlFlow, Forloop) {
  using namespace compiler::frontend;
  compiler::util::BumpAllocator allocator;

  compiler::Builder builder {};
  Parser            parser(builder, &allocator);

  static auto const binary = shader_ps_forloop;
  builder.setHostMapping(0, binary.data(), binary.size());

  parser.getOrCreateBlock(0);
  parser.process();

}