#include "compiler_ctx.h"

#include "alpaca/alpaca.h"
#include "frontend/parser.h"
#include "logging.h"
#include "util/bump_allocator.h"

#include <cstring>
#include <filesystem>

// mlir
#include "mlir/custom.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

namespace compiler {

CompilerCtx::CompilerCtx(util::Flags<ShaderBuildFlags> const& flags): _debugFlags(flags), _mlirCtx(mlir::MLIRContext::Threading::DISABLED) {
  _mlirCtx.allowUnregisteredDialects();
  _mlirCtx.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect, mlir::psoff::PSOFFDialect>();

  auto location = mlir::UnknownLoc::get(&_mlirCtx);
  _mlirModule   = mlir::ModuleOp::create(location);
}

HostMapping* CompilerCtx::getHostMapping(uint64_t pc) {
  { // Search existing
    auto it = std::find_if(_hostMapping.begin(), _hostMapping.end(), [pc](auto const& item) { return item.pc <= pc; });
    if (it != _hostMapping.end()) {
      return &*it;
    }
  }
  return nullptr;
}

void CompilerCtx::setHostMapping(uint64_t pc, uint32_t const* vaddr, uint32_t size_dw) {
  { // Search free
    auto it = std::find_if(_hostMapping.begin(), _hostMapping.end(), [](auto const& item) { return item.pc == std::numeric_limits<uint64_t>::max(); });
    if (it != _hostMapping.end()) {
      it->pc      = pc;
      it->host    = (uint64_t)vaddr;
      it->size_dw = size_dw;
    }
  }

  std::ranges::sort(_hostMapping, [](const HostMapping& a, const HostMapping& b) { return a.pc < b.pc; });
}

bool CompilerCtx::processBinary() {
  auto const      pcStart = _hostMapping[0].pc;
  uint32_t const* pCode   = (uint32_t const*)_hostMapping[0].host;
  auto const      size    = _hostMapping[0].size_dw;
  if (pCode == nullptr) return false;

  compiler::util::BumpAllocator allocator;

  frontend::Parser parser(*this, &allocator);

  mlir::OpBuilder mlirBuilder(getContext());

  auto loc = mlir::UnknownLoc::get(getContext());

  auto funcOp = mlirBuilder.create<mlir::func::FuncOp>(loc, "main", mlirBuilder.getFunctionType({}, {}));
  getModule()->push_back(funcOp);

  auto startBlock = funcOp.addEntryBlock();
  auto block      = parser.getOrCreateBlock(0, &funcOp.getBody());

  mlirBuilder.setInsertionPointToStart(startBlock);
  mlirBuilder.create<mlir::cf::BranchOp>(mlir::UnknownLoc::get(getContext()), block->mlirBlock);

  parser.process();

  return true;
}
} // namespace compiler