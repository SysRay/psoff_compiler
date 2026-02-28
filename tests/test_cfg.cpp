#include "compiler_ctx.h"
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

    mlir::OpBuilder builder(_ctx.getContext());

    _mlirModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(_ctx.getContext()));

    _func = builder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(_ctx.getContext()), "test_func", builder.getFunctionType({builder.getI1Type()}, {}));
    _mlirModule.push_back(_func);

    _block = _func.addEntryBlock();
  }

  void TearDown() override {}

  compiler::CompilerCtx _ctx {};

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
  auto            funcOp =
      mlirBuilder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(_ctx.getContext()), "main", mlirBuilder.getFunctionType({mlirBuilder.getI1Type()}, {}));

  _ctx.getModule()->push_back(funcOp);

  auto startBlock = funcOp.addEntryBlock();
  auto block      = parser.getOrCreateBlock(0, &funcOp.getBody());

  mlirBuilder.setInsertionPointToStart(startBlock);
  mlirBuilder.create<mlir::cf::BranchOp>(mlir::UnknownLoc::get(_ctx.getContext()), block->mlirBlock);

  parser.process();

  _ctx.getModule()->dump();
}

TEST_F(ControlFlow, Forloop) {
  using namespace compiler::frontend;
  compiler::util::BumpAllocator allocator;

  Parser parser(_ctx, &allocator);

  static auto const binary = shader_ps_forloop;
  _ctx.setHostMapping(0, binary.data(), binary.size());

  mlir::OpBuilder mlirBuilder(_ctx.getContext());
  auto            funcOp =
      mlirBuilder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(_ctx.getContext()), "main", mlirBuilder.getFunctionType({mlirBuilder.getI1Type()}, {}));

  _ctx.getModule()->push_back(funcOp);

  auto startBlock = funcOp.addEntryBlock();
  auto block      = parser.getOrCreateBlock(0, &funcOp.getBody());

  mlirBuilder.setInsertionPointToStart(startBlock);
  mlirBuilder.create<mlir::cf::BranchOp>(mlir::UnknownLoc::get(_ctx.getContext()), block->mlirBlock);

  parser.process();

  _ctx.getModule()->dump();
}