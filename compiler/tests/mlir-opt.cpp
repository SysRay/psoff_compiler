#include "mlir/custom.h"
#include "mlir/passes/psoff_passes.h"
#include "util/bump_allocator.h"

#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;

  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::psoff::PSOFFDialect>();

  compiler::util::BumpAllocator allocator;
  mlir::registerPass([&allocator] { return mlir::psoff::createRegisterSSAPass(allocator); });
  mlir::registerRemoveDeadValues();
  mlir::registerCSE();
  mlir::registerCanonicalizer();

  return failed(mlir::MlirOptMain(argc, argv, "psOff compiler test\n", registry));
}