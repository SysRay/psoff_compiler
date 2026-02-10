#include "custom.h"

#include <mlir/IR/Matchers.h>

namespace mlir::psoff {
mlir::OpFoldResult Create64bOp::fold(FoldAdaptor adaptor) {
  auto loAttr = llvm::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getLo());
  auto hiAttr = llvm::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getHi());

  if (!loAttr || !hiAttr) return {};

  auto loVal = loAttr.getValue().zextOrTrunc(32);
  auto hiVal = hiAttr.getValue().zextOrTrunc(32);

  mlir::APInt combined(64, 0);
  combined.insertBits(loVal, 0);
  combined.insertBits(hiVal, 32);

  return mlir::IntegerAttr::get(getType(), combined);
}

LogicalResult Split64bOp::fold(FoldAdaptor adaptor, SmallVectorImpl<mlir::OpFoldResult>& results) {
  auto inputAttr = llvm::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getInput());
  if (!inputAttr) return failure();

  mlir::APInt value = inputAttr.getValue().zextOrTrunc(64);

  mlir::APInt lo = value.extractBits(32, 0);
  mlir::APInt hi = value.extractBits(32, 32);

  auto resultTypes = getResultTypes();

  results.push_back(mlir::IntegerAttr::get(resultTypes[0], lo));
  results.push_back(mlir::IntegerAttr::get(resultTypes[1], hi));
  return success();
}

struct FoldCreate64bOfSplit64b: mlir::OpRewritePattern<Create64bOp> {
  using mlir::OpRewritePattern<Create64bOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(Create64bOp op, mlir::PatternRewriter& rewriter) const override {
    auto split = op.getLo().getDefiningOp<Split64bOp>();
    if (!split || op.getHi().getDefiningOp() != split) return mlir::failure();

    // enforce the 1:1 lo/hi mapping
    if (op.getLo() != split.getResult(0) || op.getHi() != split.getResult(1)) return mlir::failure();

    rewriter.replaceOp(op, split.getInput());
    return mlir::success();
  }
};

struct FoldSplit64bOfCreate64b: mlir::OpRewritePattern<Split64bOp> {
  using mlir::OpRewritePattern<Split64bOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(Split64bOp op, mlir::PatternRewriter& rewriter) const override {

    auto createOp = op.getInput().getDefiningOp<Create64bOp>();
    if (!createOp) return mlir::failure();

    rewriter.replaceOp(op, {createOp.getLo(), createOp.getHi()});
    return mlir::success();
  }
};

void Create64bOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context) {
  patterns.add<FoldCreate64bOfSplit64b>(context);
}

void Split64bOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context) {
  patterns.add<FoldSplit64bOfCreate64b>(context);
}

} // namespace mlir::psoff