#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

namespace compiler::util {
class BumpAllocator;
}

namespace mlir::psoff {
/**
 * @brief Tries to fold value to constant
 *
 * @param val
 * @return mlir::Attribute
 */
mlir::Attribute evaluate(mlir::Value val);

struct PromoteRegisterPass: public PassWrapper<PromoteRegisterPass, OperationPass<mlir::func::FuncOp>> {
  PromoteRegisterPass(compiler::util::BumpAllocator& allocator): _allocator(allocator) {}

  void runOnOperation() override;

  private:
  compiler::util::BumpAllocator& _allocator;
};
} // namespace mlir::psoff