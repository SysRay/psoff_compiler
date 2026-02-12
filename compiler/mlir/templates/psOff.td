
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

def RegisterSSAPass : Pass<"registerSSA", "mlir::func::FuncOp"> {
  let summary = "Connect load and stores";
  let description = [{
   Rewrite non-SSA load and store patterns to SSA.
  }];

  let constructor = "mlir::createRegisterSSAPass()";
}

def PSOFF_IndirectBranch : Op<PSOFFDialect,"IndirectBranch"> {
  let summary = "Branch to a target using a variable";
  let description = [{
    Branches to a target using a variable as destination.
  }];
  let arguments = (ins I64:$dst);
  let results = (outs);
}

def PSOFF_LoadOp : Op<PSOFFDialect, "load", [Pure]> {
  let summary = "Load a value into the register identified by an unsigned ID";
  let description = [{
    Loads a value from memory into the register specified by a compile-time constant unsigned integer ID.
    The `id` operand is an unsigned index (e.g., `ui32` or `index`) identifying the target register.
    The result type is inferred from the memory location being loaded (typically matching the register’s element type).
  }];

  let arguments = (ins IndexAttr:$id);
  let results = (outs AnyType:$val);

  let hasCustomAssemblyFormat = 1;
}

def PSOFF_StoreOp : Op<PSOFFDialect, "store", [Pure]> {
  let summary = "Store a value from an register";
  let description = [{
    Stores a value from the specified register into memory.
  }];

  let arguments = (ins IndexAttr:$id, AnyType:$val);
  let results = (outs);

  let hasCustomAssemblyFormat = 1;
}

def PSOFF_Create64bOp : Op<PSOFFDialect, "create64b", [Pure]> {
  let summary = "Create i64 from two i32 values (low, high)";
  let description = [{
    Combines two i32 values into one i64: result = (high << 32) | low.
  }];

  let arguments = (ins
    I32:$lo,
    I32:$hi
  );

  let results = (outs
    I64:$result
  );

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def PSOFF_Split64bOp : Op<PSOFFDialect, "split64b", [Pure]> {
  let summary = "Split i64 into two i32 values (low, high)";
  let description = [{
    Extracts the lower and upper 32 bits of a 64-bit integer.
  }];

  let arguments = (ins
    I64:$input
  );

  let results = (outs
    I32:$lo,
    I32:$hi
  );

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}