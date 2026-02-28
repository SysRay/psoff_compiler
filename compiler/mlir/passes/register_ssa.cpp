#include "frontend/gfx/register_types.h"
#include "frontend/shader_types.h"
#include "mlir/custom.h"
#include "psoff_passes.h"
#include "util/bump_allocator.h"

#include <array>
#include <unordered_map>

#define GEN_PASS_DEF_REGISTERSSAPASS
#include "psOff.td.pass.h.inc"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

namespace mlir::psoff {

static inline bool is64BitType(mlir::Type ty) {
  return ty.getIntOrFloatBitWidth() == 64;
}

static inline bool isBoolType(mlir::Type ty) {
  return ty.getIntOrFloatBitWidth() == 1;
}

// last two SGPRs are alias to VCC.
// However the hardware does not prevent direct user access to these registers.

// Ignore out of bounds writes

struct StorageItem {
  mlir::Value value {};
  uint8_t     index = 0;

  bool operator==(StorageItem const& rhs) const { return value == rhs.value && index == rhs.index; }
};

struct Storage {
  std::array<StorageItem, compiler::frontend::eOperandKind::size()> regs;

  void        set(compiler::frontend::eOperandKind kind, mlir::Value value);
  StorageItem get(compiler::frontend::eOperandKind kind);
};

void Storage::set(compiler::frontend::eOperandKind kind, mlir::Value value) {
  using namespace compiler::frontend;
  StorageItem* item = &regs[(eOperandKind_t)kind.value()];

  *item++ = {value, 0};
  if (is64BitType(value.getType()) || (isBoolType(value.getType()) && kind.is64bit())) {
    *item = {value, 1};
  }
}

StorageItem Storage::get(compiler::frontend::eOperandKind kind) {
  using namespace compiler::frontend;
  return regs[(eOperandKind_t)kind.value()];
}

struct ConflictItems {
  static constexpr uint32_t totalSize = compiler::frontend::eOperandKind::size();

  std::array<compiler::frontend::eOperandKind_t, totalSize> items;
  std::array<mlir::Type, totalSize>                         results;
  std::array<mlir::Value, totalSize>                        resultsThen;
  std::array<mlir::Value, totalSize>                        resultsElse;

  uint32_t numItems = 0;
};

struct RegisterSSAPass: public ::impl::RegisterSSAPassBase<RegisterSSAPass> {
  compiler::util::BumpAllocator& _allocator;
  ConflictItems                  _conflict;

  RegisterSSAPass(compiler::util::BumpAllocator& allocator): _allocator(allocator) {}

  mlir::Value getValue(Storage& storage, PatternRewriter& rewriter, uint32_t index, mlir::Type targetType, mlir::Operation* op) {
    auto const& item = storage.regs[index];
    if (!item.value) {
      rewriter.setInsertionPoint(op);
      if (targetType.isFloat()) {
        return rewriter.create<mlir::arith::ConstantFloatOp>(op->getLoc(), (mlir::FloatType)targetType, llvm::APFloat(0.f));
      } else {
        return rewriter.create<mlir::arith::ConstantIntOp>(op->getLoc(), targetType, 0);
      }
    }
    if (item.value.getType() == targetType) {
      return item.value;
    }

    auto const targetWidth = targetType.getIntOrFloatBitWidth();
    auto const valueWidth  = item.value.getType().getIntOrFloatBitWidth();

    rewriter.setInsertionPoint(op);
    if (targetWidth > valueWidth) {
      if (valueWidth == 1) {
        // Expand bool value to target type
        auto zero = mlir::spirv::ConstantOp::getZero(targetType, op->getLoc(), rewriter);
        auto one  = mlir::spirv::ConstantOp::getOne(targetType, op->getLoc(), rewriter);
        return rewriter.create<mlir::spirv::SelectOp>(op->getLoc(), targetType, item.value, one, zero);
      } else {
        // 64 Bit value. Gather lower part of 64 bit value and combine to 64 bit value
        auto itemL = storage.regs[1 + index];
        if (!itemL.value) {
          itemL = {rewriter.create<mlir::arith::ConstantIntOp>(op->getLoc(), item.value.getType(), 0), 0};
        }
        return rewriter.create<mlir::psoff::Create64bOp>(op->getLoc(), targetType, itemL.value, item.value);
      }
    } else if (targetWidth < valueWidth) {
      if (targetWidth == 1) {
        // Reduce to boolean
        auto zero = mlir::spirv::ConstantOp::getZero(item.value.getType(), op->getLoc(), rewriter);
        return rewriter.create<mlir::spirv::INotEqualOp>(op->getLoc(), targetType, item.value, zero);
      } else {
        // Split 64Bit into 2x 32 bit
        auto newOp = rewriter.create<mlir::psoff::Split64bOp>(op->getLoc(), targetType, targetType, item.value);

        if (item.index == 0) return newOp.getLo();
        return newOp.getHi();
      }
    }
    return rewriter.create<mlir::arith::BitcastOp>(op->getLoc(), targetType, item.value).getResult();
  }

  void visitRegion(mlir::Region& region, Storage& storage, PatternRewriter& rewriter) {
    using namespace compiler::frontend;
    // todo handle boolean casting, type casting

    for (auto& block: region.getBlocks()) {
      for (Operation& opBase: llvm::make_early_inc_range(block)) {
        if (auto op = dyn_cast<scf::IfOp>(opBase)) {
          ConflictItems conflicts {.numItems = 0}; // todo check bump allocator for nested cases

          Storage storageThen = storage;
          visitRegion(op.getThenRegion(), storageThen, rewriter);
          visitRegion(op.getElseRegion(), storage, rewriter);

          for (uint16_t n = 0; n < storage.regs.size(); ++n) {
            auto&       lhs = storage.regs[n];
            auto const& rhs = storageThen.regs[n];
            if (lhs != rhs) {
              // Get type

              mlir::Type resultType;
              if (!lhs.value || !rhs.value) {
                resultType = lhs.value ? lhs.value.getType() : rhs.value.getType();
              } else
                resultType =
                    lhs.value.getType().getIntOrFloatBitWidth() >= rhs.value.getType().getIntOrFloatBitWidth() ? lhs.value.getType() : rhs.value.getType();

              auto thenValue = getValue(storageThen, rewriter, n, resultType, op.getThenRegion().back().getTerminator());
              auto elseValue = getValue(storage, rewriter, n, resultType, op.getElseRegion().back().getTerminator());

              auto const index             = conflicts.numItems;
              conflicts.results[index]     = resultType;
              conflicts.resultsElse[index] = elseValue;
              conflicts.resultsThen[index] = thenValue;
              conflicts.items[index]       = (eOperandKind_t)n;

              ++conflicts.numItems;

              if (is64BitType(resultType)) ++n;
            }
          }

          if (conflicts.numItems > 0) {
            auto yield = op.getThenRegion().back().getTerminator();
            rewriter.setInsertionPoint(yield);
            rewriter.replaceOpWithNewOp<scf::YieldOp>(yield, mlir::ValueRange(conflicts.resultsThen.data(), conflicts.numItems));
            yield = op.getElseRegion().back().getTerminator();
            rewriter.setInsertionPoint(yield);
            rewriter.replaceOpWithNewOp<scf::YieldOp>(yield, mlir::ValueRange(conflicts.resultsElse.data(), conflicts.numItems));

            rewriter.setInsertionPoint(op);
            auto newOp =
                rewriter.create<mlir::scf::IfOp>(op.getLoc(), mlir::TypeRange(conflicts.results.data(), conflicts.numItems), op.getCondition(), false, false);
            newOp.getThenRegion().takeBody(op.getThenRegion());
            newOp.getElseRegion().takeBody(op.getElseRegion());
            rewriter.eraseOp(op);

            for (uint16_t n = 0; n < conflicts.numItems; ++n) {
              storage.set(eOperandKind(conflicts.items[n]), newOp.getResult(n));
            }
          }
        } else if (auto op = dyn_cast<psoff::StoreOp>(opBase)) {
          auto const kind = eOperandKind((eOperandKind_t)op.getId().getZExtValue());
          storage.set(kind, op.getVal());
        } else if (auto op = dyn_cast<psoff::LoadOp>(opBase)) {
          auto const kind = eOperandKind((eOperandKind_t)op.getId().getZExtValue());
          auto       item = storage.get(kind);

          auto const targetType = op.getType();
          if (!item.value) {
            rewriter.setInsertionPoint(op);
            if (targetType.isFloat()) {
              item.value = rewriter.replaceOpWithNewOp<mlir::arith::ConstantFloatOp>(op, (mlir::FloatType)targetType, llvm::APFloat(0.f));
            } else {
              item.value = rewriter.replaceOpWithNewOp<mlir::arith::ConstantIntOp>(op, targetType, 0);
            }
            // signalPassFailure();
            // return;

            storage.set(kind, item.value);
            continue;
          }

          auto const valueType = item.value.getType();
          if (targetType != valueType) { // Handle different type width
            auto const targetWidth = targetType.getIntOrFloatBitWidth();
            auto const valueWidth  = valueType.getIntOrFloatBitWidth();

            if (targetWidth > valueWidth) {
              rewriter.setInsertionPoint(op);

              if (valueWidth == 1) {
                // Expand bool value to target type
                auto zero = mlir::spirv::ConstantOp::getZero(targetType, op.getLoc(), rewriter);
                auto one  = mlir::spirv::ConstantOp::getOne(targetType, op.getLoc(), rewriter);

                storage.set(kind, rewriter.replaceOpWithNewOp<mlir::spirv::SelectOp>(op, item.value, one, zero));
              } else {
                // 64 Bit value. Gather lower part of 64 bit value and combine to 64 bit value
                auto itemH = storage.get(eOperandKind((eOperandKind_t)kind.value() + 1));
                if (!itemH.value) {
                  signalPassFailure();
                  return;
                }

                storage.set(kind, rewriter.replaceOpWithNewOp<mlir::psoff::Create64bOp>(op, targetType, item.value, itemH.value));
              }

            } else if (targetWidth < valueWidth) {
              rewriter.setInsertionPoint(op);

              if (targetWidth == 1) {
                // Reduce to boolean
                auto zero = mlir::spirv::ConstantOp::getZero(valueType, op.getLoc(), rewriter);
                storage.set(kind, rewriter.replaceOpWithNewOp<mlir::spirv::INotEqualOp>(op, targetType, item.value, zero));
              } else {
                // Split 64Bit into 2x 32 bit
                auto newOp = rewriter.create<mlir::psoff::Split64bOp>(op.getLoc(), targetType, targetType, item.value);
                rewriter.replaceOp(op, item.index == 0 ? newOp.getLo() : newOp.getHi());
                storage.set(kind, newOp.getLo());
                if (item.index == 0) {
                  storage.set(eOperandKind((eOperandKind_t)kind.value() + 1), newOp.getHi());
                }
              }

            } else {
              rewriter.setInsertionPoint(op);
              auto newOp = rewriter.replaceOpWithNewOp<mlir::spirv::BitcastOp>(op, targetType, item.value);
            }
          }

          else
            op.replaceAllUsesWith(item.value);
        }
      }
    }
  }

  void runOnOperation() final {
    PatternRewriter rewriter(&getContext());

    Storage storage;
    visitRegion(getOperation()->getRegions().front(), storage, rewriter);
  }
};

std::unique_ptr<Pass> createRegisterSSAPass(compiler::util::BumpAllocator& allocator) {
  return std::make_unique<RegisterSSAPass>(allocator);
}
} // namespace mlir::psoff