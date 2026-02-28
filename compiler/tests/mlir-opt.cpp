#include "mlir/custom.h"
#include "mlir/passes/psoff_passes.h"
#include "util/bump_allocator.h"

#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;

  registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
                  mlir::psoff::PSOFFDialect, mlir::spirv::SPIRVDialect>();

  registry.addExtension(+[](mlir::MLIRContext* ctx) {
    ctx->loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect, mlir::spirv::SPIRVDialect,
                     mlir::psoff::PSOFFDialect>();
  });

  compiler::util::BumpAllocator allocator;
  mlir::registerPass([&allocator] { return mlir::psoff::createRegisterSSAPass(allocator); });
  mlir::registerRemoveDeadValues();
  mlir::registerCSE();
  mlir::registerCanonicalizer();

  return failed(mlir::MlirOptMain(argc, argv, "psOff compiler test\n", registry));
}