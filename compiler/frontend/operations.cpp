#include "operations.h"

#include "mlir/custom.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>

namespace compiler::frontend::op {
void MoveOp::create(Parser* parser, OpDst dst, OpSrc src, OperandType_t type) {
  parser->storeRegister(dst, parser->loadRegister(src, type));
}

void MoveOp::create(Parser* parser, OpDst dst, uint64_t) {
  // parser->storeRegister(dst.kind, parser->loadRegister(src.kind, type));
}

void CMoveOp::create(Parser* parser, OpDst dst, OpSrc predicate, OpSrc src0, OpSrc src1, OperandType_t type) {
  auto const pred = parser->loadRegister(OpSrc(eOperandKind::SCC()), parser->types().i1());
  auto const v0   = parser->loadRegister(src0, type);
  auto const v1   = parser->loadRegister(src1, type);

  auto res = parser->create<mlir::arith::SelectOp>(pred, v0, v1);
  parser->storeRegister(dst, res);
}

void NotOp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OperandType_t type) {
  auto const v0  = parser->loadRegister(src0, type);
  auto       res = parser->create<mlir::spirv::NotOp>(v0);
  parser->storeRegister(dst, res);

  auto carryOut = parser->create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::ne, res, parser->loadRegister(OpSrc(eOperandKind::createImm(0)), type));
  parser->storeRegister(carry, carryOut);
}

void BrevOp::create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type) {
  auto const v0  = parser->loadRegister(src0, type);
  auto       res = parser->create<mlir::spirv::BitReverseOp>(v0);
  parser->storeRegister(dst, res);
}

void BitCountOp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OperandType_t type) {
  auto const v0  = parser->loadRegister(src0, type);
  auto       res = parser->create<mlir::spirv::BitCountOp>(parser->types().i32(), v0);
  parser->storeRegister(dst, res);

  auto carryOut = parser->create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::ne, res, parser->loadRegister(OpSrc(eOperandKind::createImm(0)), type));
  parser->storeRegister(carry, carryOut);
}

void FindFirstLsbBitOp::create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type) {
  auto const v0  = parser->loadRegister(src0, type);
  auto       res = parser->create<mlir::spirv::GLFindILsbOp>(parser->types().i32(), v0);
  parser->storeRegister(dst, res);
}

void FindFirstUMsbBitOp::create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type) {
  auto const v0 = parser->loadRegister(src0, type);

  mlir::Value res;
  if (type.getIntOrFloatBitWidth() <= 32)
    res = parser->create<mlir::spirv::GLFindUMsbOp>(parser->types().i32(), v0);
  else
    throw std::runtime_error("todo FindFirstMsbBitOp 64 bit");
  parser->storeRegister(dst, res);
}

void FindFirstSMsbBitOp::create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t type) {
  auto const v0 = parser->loadRegister(src0, type);

  mlir::Value res;
  if (type.getIntOrFloatBitWidth() <= 32)
    res = parser->create<mlir::spirv::GLFindSMsbOp>(parser->types().i32(), v0);
  else
    throw std::runtime_error("todo FindFirstMsbBitOp 64 bit");
  parser->storeRegister(dst, res);
}

void SignExtOp::create(Parser* parser, OpDst dst, OperandType_t dstType, OpSrc src0, OperandType_t srcType) {
  auto value0 = parser->loadRegister(src0, srcType);
  auto res    = parser->create<mlir::arith::ExtSIOp>(dstType, value0);
  parser->storeRegister(dst, res);
}

void AbsIOp::create(Parser* parser, OpDst dst, OpSrc src0, OperandType_t srcType) {
  auto value0 = parser->loadRegister(src0, srcType);
  auto res    = parser->create<mlir::spirv::GLSAbsOp>(value0);
  parser->storeRegister(dst, res);
}

void BitSetOp::create(Parser* parser, OpDst dst, OpSrc offset, OperandType_t type) {
  auto opOffset = parser->loadRegister(offset, parser->types().i32());
  auto value0   = parser->loadRegister(OpSrc(dst.kind), type);

  auto bitMask = parser->create<mlir::arith::ConstantIntOp>(type, type.getIntOrFloatBitWidth() > 32 ? 0b111111 : 0b11111);
  auto index   = parser->create<mlir::spirv::BitwiseAndOp>(parser->types().i32(), opOffset, bitMask);

  auto bitValue = parser->create<mlir::arith::ConstantIntOp>(type, 1);
  auto result   = parser->create<mlir::spirv::BitFieldInsertOp>(parser->types().i32(), value0, bitValue, index, bitValue);

  auto res = parser->create<mlir::spirv::GLSAbsOp>(value0);
  parser->storeRegister(dst, res);
}

void BitClearOp::create(Parser* parser, OpDst dst, OpSrc offset, OperandType_t type) {
  auto opOffset = parser->loadRegister(offset, parser->types().i32());
  auto value0   = parser->loadRegister(OpSrc(dst.kind), type);

  auto bitMask = parser->create<mlir::arith::ConstantIntOp>(type, type.getIntOrFloatBitWidth() > 32 ? 0b111111 : 0b11111);
  auto index   = parser->create<mlir::spirv::BitwiseAndOp>(parser->types().i32(), opOffset, bitMask);

  auto bitValue = parser->create<mlir::arith::ConstantIntOp>(type, 0);
  auto bitWidth = parser->create<mlir::arith::ConstantIntOp>(type, 1);
  auto result   = parser->create<mlir::spirv::BitFieldInsertOp>(parser->types().i32(), value0, bitValue, index, bitWidth);

  auto res = parser->create<mlir::spirv::GLSAbsOp>(value0);
  parser->storeRegister(dst, res);
}

void BranchOp::create(Parser* parser, OpSrc src) {
  auto target = parser->loadRegister(src, parser->types().i64());
  auto res    = parser->create<mlir::psoff::IndirectBranch>(target);
}

void SaveExecOp::create(Parser* parser, OpDst dst, BitOp bitop, OpSrc src0, OpSrc src1, OperandType_t type) {
  auto mask = parser->loadRegister(src0, type);
  auto exec = parser->loadRegister(src1, type);

  mlir::Value res;
  switch (bitop) {
    case BitOp::eAND: res = parser->create<mlir::arith::AndIOp>(type, mask, exec); break;
    case BitOp::eOR: res = parser->create<mlir::arith::OrIOp>(type, mask, exec); break;
    case BitOp::eXOR: res = parser->create<mlir::arith::XOrIOp>(type, mask, exec); break;
  }

  parser->storeRegister(dst, exec); // save exec
  parser->storeRegister(OpDst(eOperandKind::EXEC()), res);
}
} // namespace compiler::frontend::op