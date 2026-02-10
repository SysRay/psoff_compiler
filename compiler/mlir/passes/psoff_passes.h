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

std::unique_ptr<Pass> createRegisterSSAPass(compiler::util::BumpAllocator& allocator);
} // namespace mlir::psoff