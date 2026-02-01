#pragma once

#include <mlir/IR/PatternMatch.h>

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
} // namespace mlir::psoff