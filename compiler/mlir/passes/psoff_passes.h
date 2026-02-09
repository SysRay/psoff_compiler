#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

namespace liverpool::lift {
class LiftContext;
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
  PromoteRegisterPass() {}

  void runOnOperation() override;
};
} // namespace mlir::psoff