
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def PSOFFDialect : Dialect {
  let name = "psoff";
  let summary = "psoff Operations";

  let description = [{
    First step. Create graph and then structure it. Connect operations afterwards
  }];

  let cppNamespace = "::mlir::psoff";
  let hasConstantMaterializer = 1;
}

def PSOFF_Branch : Op<PSOFFDialect,"branch"> {
  let summary = "Branch to a target using a variable";
  let description = [{
    Branches to a target using a variable as destination.
  }];
  let arguments = (ins I64:$dst);
  let results = (outs);
}