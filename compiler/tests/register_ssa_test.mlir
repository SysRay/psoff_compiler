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
// CHECK:         return %[[RES]] : i64
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
// CHECK:   return %[[VAL0]] : i32
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
// CHECK:   return %[[VAL4]] : i32
func.func @splitUpperLower(%arg0: i64, %arg1: i64) -> i32 {
  psoff.store s[4] = %arg0 : i64
  psoff.store s[6] = %arg1 : i64

  %2 = psoff.load s[5] : i32
  %3 = psoff.load s[6] : i32

  %res = arith.addi %2, %3 : i32
  return %res : i32
}