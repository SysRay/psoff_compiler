#include "mlir/custom.h"

#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;

  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::psoff::PSOFFDialect>();

  //mlir::registerAllPasses();
  //my::registerMyPasses(); // todo

  return failed(mlir::MlirOptMain(argc, argv, "psOff compiler test\n", registry));
}