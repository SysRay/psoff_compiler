// RUN: mlir-opt --registerSSA --remove-dead-values %s | FileCheck %s

// CHECK-LABEL: func.func @simple
// CHECK-SAME: (%[[ARG0:.*]]: f32) -> i1
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CMP:.*]] = arith.cmpf one, %[[CST]], %[[ARG0]] : f32
// CHECK: return %[[CMP]] : i1
func.func @simple(%arg0: f32) -> i1 {
  psoff.store s[4] = %arg0 : f32
  %cst = arith.constant 0.000000e+00 : f32
  %2 = psoff.load s[4] : f32
  %3 = arith.cmpf one, %cst, %2 : f32
  return %3 : i1
}

// CHECK-LABEL: func.func @combined
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i64
// CHECK:         %[[VAL0:.*]] = "psoff.create64b"(%[[ARG1]], %[[ARG0]]) : (i32, i32) -> i64
// CHECK:         %[[VAL1:.*]] = "psoff.create64b"(%[[ARG0]], %[[ARG1]]) : (i32, i32) -> i64
// CHECK:         %[[RES:.*]] = arith.addi %[[VAL0]], %[[VAL1]] : i64
func.func @combined(%arg0: i32, %arg1: i32) -> i64 {
  psoff.store s[4] = %arg0 : i32
  psoff.store s[5] = %arg1 : i32
  psoff.store s[6] = %arg1 : i32
  psoff.store s[7] = %arg0 : i32

  %2 = psoff.load s[4] : i64
  %3 = psoff.load s[6] : i64

  %res = arith.addi %2, %3 : i64
  return %res : i64
}

// CHECK-LABEL: func.func @combined64b
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i64
// CHECK: %[[VAL0:.*]] = "psoff.create64b"(%[[ARG1]], %[[ARG0]]) : (i32, i32) -> i64
// CHECK: %[[VAL1:.*]] = "psoff.create64b"(%[[ARG0]], %[[ARG1]]) : (i32, i32) -> i64
// CHECK: %[[RES:.*]] = arith.addi %[[VAL0]], %[[VAL1]] : i64
func.func @combined64b(%arg0: i32, %arg1: i32) -> i64 {
  psoff.store s[4] = %arg0 : i32
  psoff.store s[5] = %arg1 : i32
  psoff.store s[6] = %arg1 : i32
  psoff.store s[7] = %arg0 : i32

  %2 = psoff.load s[4] : i64
  %3 = psoff.load s[6] : i64

  %res = arith.addi %2, %3 : i64
  return %res : i64
}

// CHECK-LABEL: func.func @split
// CHECK-SAME: (%[[ARG0:.*]]: i64)
// CHECK:   %[[LO:.*]], %[[HI:.*]] = "psoff.split64b"(%[[ARG0]]) : (i64) -> (i32, i32)
// CHECK:   %[[VAL0:.*]] = arith.addi %[[LO]], %[[HI]] : i32
func.func @split(%arg0: i64) -> i32 {
  psoff.store s[4] = %arg0 : i64

  %2 = psoff.load s[4] : i32
  %3 = psoff.load s[5] : i32

  %res = arith.addi %2, %3 : i32
  return %res : i32
}

// CHECK-LABEL: func.func @splitUpperLower
// CHECK-SAME: (%[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64)
// CHECK:   %[[VAL0:.*]], %[[VAL1:.*]] = "psoff.split64b"(%[[ARG0:.*]]) : (i64) -> (i32, i32)
// CHECK:   %[[VAL2:.*]], %[[VAL3:.*]] = "psoff.split64b"(%[[ARG1:.*]]) : (i64) -> (i32, i32)
// CHECK:   %[[VAL4:.*]] = arith.addi %[[VAL1]], %[[VAL2]] : i32
func.func @splitUpperLower(%arg0: i64, %arg1: i64) -> i32 {
  psoff.store s[4] = %arg0 : i64
  psoff.store s[6] = %arg1 : i64

  %2 = psoff.load s[5] : i32
  %3 = psoff.load s[6] : i32

  %res = arith.addi %2, %3 : i32
  return %res : i32
}

// CHECK-LABEL: func.func @simpleIf
// CHECK-SAME: (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: i1)
// CHECK: %[[VAL0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[VAL1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[VAL2:.*]]:2 = scf.if %[[ARG1]] -> (f32, f32) {
// CHECK:   scf.yield %[[ARG0]], %[[VAL1]] : f32, f32
// CHECK: } else {
// CHECK:   scf.yield %[[VAL0]], %[[VAL0]] : f32, f32
// CHECK: }
// CHECK: %[[RES:.*]] = arith.cmpf one, %[[VAL2]]#0, %[[VAL2]]#1 : f32
func.func @simpleIf(%arg0: f32, %arg1: i1) -> i1 {
  psoff.store s[0] = %arg1 : i1
  %1 = psoff.load s[0] : i1
  %cst = arith.constant 0.000000e+00 : f32
  %cst1 = arith.constant 1.000000e+00 : f32
  psoff.store s[4] = %cst : f32
  psoff.store v[2] = %cst1 : f32

  scf.if %1 {
    psoff.store s[4] = %arg0 : f32
  } else {
    %2 = psoff.load s[4] : f32
    psoff.store v[2] = %2 : f32
  }

  %3 = psoff.load s[4] : f32
  %4 = psoff.load v[2] : f32
  %5 = arith.cmpf one, %3, %4 : f32
  return %5 : i1
}

func.func @simpleIfDiffTypes(%arg0: i64, %arg1: i1) -> i64 {
  %cst = arith.constant 0 : i32
  %cst1 = arith.constant 1 : i32

  psoff.store s[4] = %cst : i32
  psoff.store s[5] = %cst : i32
  psoff.store v[2] = %cst1 : i32
  psoff.store v[3] = %cst1 : i32

  psoff.store s[0] = %arg1 : i1
  %1 = psoff.load s[0] : i1

  scf.if %1 {
    psoff.store s[4] = %arg0 : i64
  } else {
    %2 = psoff.load s[5] : i32
    psoff.store v[2] = %2 : i32
  }

  %3 = psoff.load s[4] : i64
  %4 = psoff.load v[2] : i64
  %res = arith.addi %3, %4 : i64
  return %res : i64
}