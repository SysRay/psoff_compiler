
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

def PSOFF_IndirectBranch : Op<PSOFFDialect,"IndirectBranch"> {
  let summary = "Branch to a target using a variable";
  let description = [{
    Branches to a target using a variable as destination.
  }];
  let arguments = (ins I64:$dst);
  let results = (outs);
}

def PSOFF_LoadOp : Op<PSOFFDialect, "LoadOp", [MemoryEffects<[MemRead]>]> {
  let summary = "Load a value into the register identified by an unsigned ID";
  let description = [{
    Loads a value from memory into the register specified by a compile-time constant unsigned integer ID.
    The `id` operand is an unsigned index (e.g., `ui32` or `index`) identifying the target register.
    The result type is inferred from the memory location being loaded (typically matching the register’s element type).
  }];

  let arguments = (ins IndexAttr:$id);
  let results = (outs AnyType:$val);

  let traits = [ConstantLike];
  let hasCustomAssemblyFormat = 1;
}

def PSOFF_StoreOp : Op<PSOFFDialect, "StoreOp", [MemoryEffects<[MemWrite]>]> {
  let summary = "Store a value from an register";
  let description = [{
    Stores a value from the specified register into memory.
  }];

  let arguments = (ins IndexAttr:$id, AnyType:$val);
  let results = (outs);

  let hasCustomAssemblyFormat = 1;
}