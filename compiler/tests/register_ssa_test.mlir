// RUN: mlir-opt --registerSSA --remove-dead-values %s | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%[[ARG0:.*]]: f32) -> i1
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CMP:.*]] = arith.cmpf one, %[[CST]], %[[ARG0]] : f32
// CHECK: return %[[CMP]] : i1

func.func @main(%arg0: f32) -> i1 {
  psoff.store s[4] = %arg0 : f32
  %cst = arith.constant 0.000000e+00 : f32
  %2 = psoff.load s[4] : f32
  %3 = arith.cmpf one, %cst, %2 : f32
  return %3 : i1
}