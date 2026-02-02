
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

def PSOFF_Branch : Op<PSOFFDialect,"BranchOp"> {
  let summary = "Branch to a target using a variable";
  let description = [{
    Branches to a target using a variable as destination.
  }];
  let arguments = (ins I64Attr:$dst);
  let results = (outs);
}

def PSOFF_CBranch : Op<PSOFFDialect,"CBranchOp"> {
  let summary = "Branch to a target using a variable";
  let description = [{
    Branches to a target using a variable as destination.
  }];
  let arguments = (ins
    I1:$cond,
    I64Attr:$ifBranch,
    I64Attr:$elseBranch
  );
  let results = (outs);
}