#include "frontend/gfx/register_types.h"
#include "frontend/shader_types.h"
#include "mlir/custom.h"
#include "psoff_passes.h"
#include "util/bump_allocator.h"

#include <unordered_map>

namespace mlir::psoff {
static inline bool is64BitType(mlir::Type ty) {
  return ty.getIntOrFloatBitWidth() == 64;
}

// last two SGPRs are alias to VCC.
// However the hardware does not prevent direct user access to these registers.

// Ignore out of bounds writes
struct Storage {
  struct {
    std::array<Value, compiler::frontend::SPEC_TOTAL_SGPR> sgpr;
    std::array<Value, compiler::frontend::SPEC_TOTAL_VGPR> vgpr;

    // Value exec_lo;
    // Value exec_hi;
    // Value m0;
    // Value vskip;
  };

  void        set(compiler::frontend::eOperandKind kind, mlir::Value value);
  mlir::Value get(compiler::frontend::eOperandKind kind, mlir::Type type);
};

void Storage::set(compiler::frontend::eOperandKind kind, mlir::Value value) {
  using namespace compiler::frontend;
  switch (kind.base()) {
    case eOperandKind::eBase::SGPR: {
      sgpr[kind.getSGPR()] = value;
    } break;
    case eOperandKind::eBase::VGPR: {
      vgpr[kind.getVGPR()] = value;
    } break;
    default: {

    } break;
  }
}

mlir::Value Storage::get(compiler::frontend::eOperandKind kind, mlir::Type type) {
  using namespace compiler::frontend;
  switch (kind.base()) {
    case eOperandKind::eBase::SGPR: {
      return sgpr[kind.getSGPR()];
    } break;
    case eOperandKind::eBase::VGPR: {
      return vgpr[kind.getVGPR()];
    } break;
    default: {

    } break;
  }

  return {};
}

void PromoteRegisterPass::runOnOperation() {
  auto const& funcOp = Pass::getOperation();

  std::pmr::unordered_map<Block*, Storage> values(&_allocator); // todo or use stack

  auto curItem = values.emplace(funcOp->getBlock(), Storage {}).first;

  funcOp->getBlock()->walk([&](Operation* opBase) {
    if (auto op = dyn_cast<psoff::StoreOp>(opBase)) {
      auto const kind = compiler::frontend::eOperandKind((compiler::frontend::eOperandKind_t)op.getId().getZExtValue());
      curItem->second.set(kind, op.getVal());
    } else if (auto op = dyn_cast<psoff::LoadOp>(opBase)) {
      auto const kind  = compiler::frontend::eOperandKind((compiler::frontend::eOperandKind_t)op.getId().getZExtValue());
      auto       value = curItem->second.get(kind, op.getType());
      op.replaceAllUsesWith(value);
    }
  });
}
} // namespace mlir::psoff