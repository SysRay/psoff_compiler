#include "frontend/gfx/register_types.h"
#include "frontend/shader_types.h"
#include "mlir/custom.h"
#include "psoff_passes.h"
#include "util/bump_allocator.h"

#include <unordered_map>

#define GEN_PASS_DEF_REGISTERSSAPASS
#include "psOff.td.pass.h.inc"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

namespace mlir::psoff {

static inline bool is64BitType(mlir::Type ty) {
  return ty.getIntOrFloatBitWidth() == 64;
}

// last two SGPRs are alias to VCC.
// However the hardware does not prevent direct user access to these registers.

// Ignore out of bounds writes

struct StorageItem {
  mlir::Value value {};
  uint8_t     index = 0;
};

struct Storage {
  struct {
    std::array<StorageItem, compiler::frontend::SPEC_TOTAL_SGPR> sgpr;
    std::array<StorageItem, compiler::frontend::SPEC_TOTAL_VGPR> vgpr;

    // Value exec_lo;
    // Value exec_hi;
    // Value m0;
    // Value vskip;
  };

  void        set(mlir::Location loc, compiler::frontend::eOperandKind kind, mlir::Value value);
  StorageItem get(mlir::Location loc, compiler::frontend::eOperandKind kind);
};

void Storage::set(mlir::Location loc, compiler::frontend::eOperandKind kind, mlir::Value value) {
  using namespace compiler::frontend;

  StorageItem* item = nullptr;
  switch (kind.base()) {
    case eOperandKind::eBase::SGPR: {
      item = &sgpr[kind.getSGPR()];
    } break;
    case eOperandKind::eBase::VGPR: {
      item = &vgpr[kind.getVGPR()];
    } break;
    default: {
      emitError(loc, "Unknown kind");
    } break;
  }

  *item++ = {value, 0};
  if (is64BitType(value.getType())) {
    *item = {value, 1};
  }
}

StorageItem Storage::get(mlir::Location loc, compiler::frontend::eOperandKind kind) {
  using namespace compiler::frontend;
  switch (kind.base()) {
    case eOperandKind::eBase::SGPR: {
      return sgpr[kind.getSGPR()];
    } break;
    case eOperandKind::eBase::VGPR: {
      return vgpr[kind.getVGPR()];
    } break;
    default: {
      emitError(loc, "Unknown kind");
    } break;
  }

  return {};
}

// static mlir::Value getB32(std::pair<StorageKey_t, Storage> const& value) {
//   auto const width = value.second.value.getType().getIntOrFloatBitWidth();
//   if (width == 32)
//     return value.second.value;
//   else if (width < 32)
//     return _mlirData.create<mlir::arith::ExtUIOp>(_types._i32, value.second.value).getResult();

//   auto splitValue = _mlirData.create<mlir::psoff::Split64bOp>(_types._i32, _types._i32, getB64(value));

//   if (value.first.second == 0) return splitValue.getLo();
//   return splitValue.getHi();
// };

struct RegisterSSAPass: public ::impl::RegisterSSAPassBase<RegisterSSAPass> {
  compiler::util::BumpAllocator& _allocator;

  RegisterSSAPass(compiler::util::BumpAllocator& allocator): _allocator(allocator) {}

  void runOnOperation() final {
    auto const funcOp = Pass::getOperation();

    PatternRewriter rewriter(&getContext());

    std::pmr::unordered_map<Block*, Storage> values(&_allocator); // todo or use stack
    using namespace compiler::frontend;

    // funcOp->walk([&](Block* block)
    {
      auto& block   = funcOp->getRegions().front().getBlocks().front();
      auto  curItem = values.emplace(&block, Storage {}).first;
      for (Operation& opBase: llvm::make_early_inc_range(block)) {
        if (auto op = dyn_cast<scf::IfOp>(opBase)) {

        } else if (auto op = dyn_cast<psoff::StoreOp>(opBase)) {
          auto const kind = eOperandKind((eOperandKind_t)op.getId().getZExtValue());
          curItem->second.set(op.getLoc(), kind, op.getVal());
        } else if (auto op = dyn_cast<psoff::LoadOp>(opBase)) {
          auto const kind = eOperandKind((eOperandKind_t)op.getId().getZExtValue());
          auto       item = curItem->second.get(op.getLoc(), kind);

          if (!item.value) {
            signalPassFailure();
            return;
          }

          auto const targetType = op.getType();
          auto const valueType  = item.value.getType();
          if (targetType != valueType) { // Handle different type width
            auto const targetWidth = targetType.getIntOrFloatBitWidth();
            auto const valueWidth  = valueType.getIntOrFloatBitWidth();

            if (targetWidth > valueWidth) {
              // Gather lower part of 64 bit value and combine to 64 bit value
              auto itemL = curItem->second.get(op.getLoc(), eOperandKind((eOperandKind_t)kind.value() + 1));
              if (!itemL.value) {
                signalPassFailure();
                return;
              }

              rewriter.setInsertionPoint(op);
              auto newOp = rewriter.replaceOpWithNewOp<mlir::psoff::Create64bOp>(op, targetType, itemL.value, item.value);
              curItem->second.set(op.getLoc(), kind, newOp);
            } else if (targetWidth < valueWidth) {
              rewriter.setInsertionPoint(op);
              auto newOp = rewriter.create<mlir::psoff::Split64bOp>(op.getLoc(), targetType, targetType, item.value);
              rewriter.replaceOp(op, item.index == 0 ? newOp.getLo() : newOp.getHi());

              curItem->second.set(op.getLoc(), kind, newOp.getLo());
              if (item.index == 0) {
                curItem->second.set(op.getLoc(), eOperandKind((eOperandKind_t)kind.value() + 1), newOp.getHi());
              }
            } else {
              rewriter.setInsertionPoint(op);
              auto newOp = rewriter.replaceOpWithNewOp<mlir::arith::BitcastOp>(op, targetType, item.value);
            }
          }

          else
            op.replaceAllUsesWith(item.value);
        }
      }
      // return WalkResult::advance();
    }
  }
};

std::unique_ptr<Pass> createRegisterSSAPass(compiler::util::BumpAllocator& allocator) {
  return std::make_unique<RegisterSSAPass>(allocator);
}
} // namespace mlir::psoff