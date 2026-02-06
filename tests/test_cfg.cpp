#include "builder.h"
#include "frontend/parser.h"
#include "mlir/custom.h"
#include "shaders.h"
#include "util/bump_allocator.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
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

  mlir::OpBuilder mlirBuilder(builder.getContext());
  auto            funcOp =
      mlirBuilder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(builder.getContext()), "main", mlirBuilder.getFunctionType({mlirBuilder.getI1Type()}, {}));

  builder.getModule()->push_back(funcOp);

  auto startBlock = funcOp.addEntryBlock();
  auto block      = parser.getOrCreateBlock(0, &funcOp.getBody());

  mlirBuilder.setInsertionPointToStart(startBlock);
  mlirBuilder.create<mlir::cf::BranchOp>(mlir::UnknownLoc::get(builder.getContext()), block->mlirBlock);

  parser.process();

  builder.getModule()->dump();
}

TEST_F(ControlFlow, Forloop) {
  using namespace compiler::frontend;
  compiler::util::BumpAllocator allocator;

  compiler::Builder builder {};
  Parser            parser(builder, &allocator);

  static auto const binary = shader_ps_forloop;
  builder.setHostMapping(0, binary.data(), binary.size());

  mlir::OpBuilder mlirBuilder(builder.getContext());
  auto            funcOp =
      mlirBuilder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(builder.getContext()), "main", mlirBuilder.getFunctionType({mlirBuilder.getI1Type()}, {}));

  builder.getModule()->push_back(funcOp);

  auto startBlock = funcOp.addEntryBlock();
  auto block      = parser.getOrCreateBlock(0, &funcOp.getBody());

  mlirBuilder.setInsertionPointToStart(startBlock);
  mlirBuilder.create<mlir::cf::BranchOp>(mlir::UnknownLoc::get(builder.getContext()), block->mlirBlock);

  parser.process();

  builder.getModule()->dump();
}