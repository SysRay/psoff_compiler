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

  auto res = parser->create<mlir::spirv::SelectOp>(pred, v0, v1);
  parser->storeRegister(dst, res);
}

void NotOp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OperandType_t type) {
  auto const v0  = parser->loadRegister(src0, type);
  auto       res = parser->create<mlir::spirv::NotOp>(v0);
  parser->storeRegister(dst, res);

  auto zero     = mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder());
  auto carryOut = parser->create<mlir::spirv::INotEqualOp>(res, zero);
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

  auto zero     = mlir::spirv::ConstantOp::getZero(parser->types().i32(), parser->getLoc(), parser->getBuilder());
  auto carryOut = parser->create<mlir::spirv::INotEqualOp>(res, zero);
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
  auto res    = parser->create<mlir::spirv::BitFieldSExtractOp>(dstType, value0);
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

  auto bitValue = mlir::spirv::ConstantOp::getOne(type, parser->getLoc(), parser->getBuilder());
  auto result   = parser->create<mlir::spirv::BitFieldInsertOp>(parser->types().i32(), value0, bitValue, index, bitValue);

  auto res = parser->create<mlir::spirv::GLSAbsOp>(value0);
  parser->storeRegister(dst, res);
}

void BitClearOp::create(Parser* parser, OpDst dst, OpSrc offset, OperandType_t type) {
  auto opOffset = parser->loadRegister(offset, parser->types().i32());
  auto value0   = parser->loadRegister(OpSrc(dst.kind), type);

  auto bitMask = parser->create<mlir::arith::ConstantIntOp>(type, type.getIntOrFloatBitWidth() > 32 ? 0b111111 : 0b11111);
  auto index   = parser->create<mlir::spirv::BitwiseAndOp>(parser->types().i32(), opOffset, bitMask);

  auto bitValue = mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder());
  auto bitWidth = mlir::spirv::ConstantOp::getOne(type, parser->getLoc(), parser->getBuilder());
  auto result   = parser->create<mlir::spirv::BitFieldInsertOp>(parser->types().i32(), value0, bitValue, index, bitWidth);

  auto res = parser->create<mlir::spirv::GLSAbsOp>(value0);
  parser->storeRegister(dst, res);
}

void AddUIOp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type) {
  auto value0 = parser->loadRegister(src0, type);
  auto value1 = parser->loadRegister(src1, type);

  auto results     = parser->create<mlir::spirv::IAddCarryOp>(value0, value1);
  auto resultValue = parser->create<mlir::spirv::CompositeExtractOp>(results, llvm::ArrayRef(0));
  auto carryValue  = parser->create<mlir::spirv::CompositeExtractOp>(results, llvm::ArrayRef(1));

  auto zero        = mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder());
  auto carryResult = parser->create<mlir::spirv::INotEqualOp>(carryValue, zero);

  parser->storeRegister(dst, resultValue);
  parser->storeRegister(carry, carryResult);
}

void AddUIOp::create(Parser* parser, OpDst dst, OpDst carryOut, OpSrc src0, OpSrc src1, OpSrc carryIn, OperandType_t type) {
  auto value0       = parser->loadRegister(src0, type);
  auto value1       = parser->loadRegister(src1, type);
  auto carryInvalue = parser->loadRegister(carryIn, type);

  auto results      = parser->create<mlir::spirv::IAddCarryOp>(value0, value1);
  auto resultValue1 = parser->create<mlir::spirv::CompositeExtractOp>(results, llvm::ArrayRef(0));
  auto carryValue1  = parser->create<mlir::spirv::CompositeExtractOp>(results, llvm::ArrayRef(1));

  auto results2     = parser->create<mlir::spirv::IAddCarryOp>(resultValue1, carryInvalue);
  auto resultValue2 = parser->create<mlir::spirv::CompositeExtractOp>(results2, llvm::ArrayRef(0));
  auto carryValue2  = parser->create<mlir::spirv::CompositeExtractOp>(results2, llvm::ArrayRef(1));

  auto carryValue  = parser->create<mlir::spirv::BitwiseOrOp>(carryValue1, carryValue2);
  auto zero        = mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder());
  auto carryResult = parser->create<mlir::spirv::INotEqualOp>(carryValue, zero);

  parser->storeRegister(dst, resultValue2);
  parser->storeRegister(carryOut, carryResult);
}

void AddSIOp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type) {
  auto value0 = parser->loadRegister(src0, type);
  auto value1 = parser->loadRegister(src1, type);
  auto res    = parser->create<mlir::spirv::IAddOp>(value0, value1);
  parser->storeRegister(dst, res);
  parser->storeRegister(carry, mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder())); // todo
}

void SubUIOp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type) {
  auto value0 = parser->loadRegister(src0, type);
  auto value1 = parser->loadRegister(src1, type);

  auto results     = parser->create<mlir::spirv::ISubBorrowOp>(value0, value1);
  auto resultValue = parser->create<mlir::spirv::CompositeExtractOp>(results, llvm::ArrayRef(0));
  auto carryValue  = parser->create<mlir::spirv::CompositeExtractOp>(results, llvm::ArrayRef(1));

  auto zero        = mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder());
  auto carryResult = parser->create<mlir::spirv::INotEqualOp>(carryValue, zero);

  parser->storeRegister(dst, resultValue);
  parser->storeRegister(carry, carryResult);
}

void SubUIOp::create(Parser* parser, OpDst dst, OpDst carryOut, OpSrc src0, OpSrc src1, OpSrc carryIn, OperandType_t type) {
  auto value0       = parser->loadRegister(src0, type);
  auto value1       = parser->loadRegister(src1, type);
  auto carryInvalue = parser->loadRegister(carryIn, type);

  auto results1     = parser->create<mlir::spirv::ISubBorrowOp>(value0, value1);
  auto resultValue1 = parser->create<mlir::spirv::CompositeExtractOp>(results1, llvm::ArrayRef(0));
  auto carryValue1  = parser->create<mlir::spirv::CompositeExtractOp>(results1, llvm::ArrayRef(1));

  auto results2     = parser->create<mlir::spirv::ISubBorrowOp>(results1, carryInvalue);
  auto resultValue2 = parser->create<mlir::spirv::CompositeExtractOp>(results2, llvm::ArrayRef(0));
  auto carryValue2  = parser->create<mlir::spirv::CompositeExtractOp>(results2, llvm::ArrayRef(1));

  auto carryValue  = parser->create<mlir::spirv::BitwiseOrOp>(carryValue1, carryValue2);
  auto zero        = mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder());
  auto carryResult = parser->create<mlir::spirv::INotEqualOp>(carryValue, zero);

  parser->storeRegister(dst, results2);
  parser->storeRegister(carryOut, carryResult);
}

void SubSIOp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type) {
  auto value0 = parser->loadRegister(src0, type);
  auto value1 = parser->loadRegister(src1, type);
  auto res    = parser->create<mlir::spirv::ISubOp>(value0, value1);
  parser->storeRegister(dst, res);
  parser->storeRegister(carry, mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder())); // todo
}

void MulSIOp::create(Parser* parser, OpDst dst, OpSrc src0, OpSrc src1, OperandType_t type) {
  auto value0 = parser->loadRegister(src0, type);
  auto value1 = parser->loadRegister(src1, type);
  auto res    = parser->create<mlir::spirv::IMulOp>(value0, value1);
  parser->storeRegister(dst, res);
}

void MinUIOp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type) {
  auto value0 = parser->loadRegister(src0, type);
  auto value1 = parser->loadRegister(src1, type);

  auto pred = parser->create<mlir::spirv::ULessThanOp>(value0, value1);
  auto res  = parser->create<mlir::spirv::SelectOp>(pred, value0, value1);

  parser->storeRegister(dst, res);
  parser->storeRegister(carry, pred);
}

void MaxUIOp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type) {
  auto value0 = parser->loadRegister(src0, type);
  auto value1 = parser->loadRegister(src1, type);

  auto pred = parser->create<mlir::spirv::UGreaterThanOp>(value0, value1);
  auto res  = parser->create<mlir::spirv::SelectOp>(pred, value0, value1);

  parser->storeRegister(dst, res);
  parser->storeRegister(carry, pred);
}

void MinSIOp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type) {
  auto value0 = parser->loadRegister(src0, type);
  auto value1 = parser->loadRegister(src1, type);

  auto pred = parser->create<mlir::spirv::SLessThanOp>(value0, value1);
  auto res  = parser->create<mlir::spirv::SelectOp>(pred, value0, value1);

  parser->storeRegister(dst, res);
  parser->storeRegister(carry, pred);
}

void MaxSIOp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type) {
  auto value0 = parser->loadRegister(src0, type);
  auto value1 = parser->loadRegister(src1, type);

  auto pred = parser->create<mlir::spirv::SGreaterThanOp>(value0, value1);
  auto res  = parser->create<mlir::spirv::SelectOp>(pred, value0, value1);

  parser->storeRegister(dst, res);
  parser->storeRegister(carry, pred);
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
    case BitOp::eAND: res = parser->create<mlir::spirv::BitwiseAndOp>(type, mask, exec); break;
    case BitOp::eOR: res = parser->create<mlir::spirv::BitwiseOrOp>(type, mask, exec); break;
    case BitOp::eXOR: res = parser->create<mlir::spirv::BitwiseXorOp>(type, mask, exec); break;
  }

  parser->storeRegister(dst, exec);
  parser->storeRegister(OpDst(eOperandKind::EXEC()), res);
}

void CmpIOp::create(Parser* parser, eCmpIPredicate predOp, OpDst dst, OpSrc src0, OpSrc src1, OperandType_t type) {
  auto value0 = parser->loadRegister(src0, type);
  auto value1 = parser->loadRegister(src1, type);

  mlir::Value res;
  switch (predOp) {
    case eCmpIPredicate::AlwaysFalse: res = mlir::spirv::ConstantOp::getZero(parser->types().i1(), parser->getLoc(), parser->getBuilder()); break;
    case eCmpIPredicate::eq: res = parser->create<mlir::spirv::IEqualOp>(value0, value1); break;
    case eCmpIPredicate::ne: res = parser->create<mlir::spirv::INotEqualOp>(value0, value1); break;
    case eCmpIPredicate::slt: res = parser->create<mlir::spirv::SLessThanOp>(value0, value1); break;
    case eCmpIPredicate::sle: res = parser->create<mlir::spirv::SLessThanEqualOp>(value0, value1); break;
    case eCmpIPredicate::sgt: res = parser->create<mlir::spirv::SGreaterThanOp>(value0, value1); break;
    case eCmpIPredicate::sge: res = parser->create<mlir::spirv::SGreaterThanEqualOp>(value0, value1); break;
    case eCmpIPredicate::ult: res = parser->create<mlir::spirv::ULessThanOp>(value0, value1); break;
    case eCmpIPredicate::ule: res = parser->create<mlir::spirv::ULessThanEqualOp>(value0, value1); break;
    case eCmpIPredicate::ugt: res = parser->create<mlir::spirv::UGreaterThanOp>(value0, value1); break;
    case eCmpIPredicate::uge: res = parser->create<mlir::spirv::UGreaterThanEqualOp>(value0, value1); break;
    case eCmpIPredicate::AlwaysTrue: res = mlir::spirv::ConstantOp::getOne(parser->types().i1(), parser->getLoc(), parser->getBuilder()); break;
  }

  parser->storeRegister(dst, res);
}

void IsBitSetOp::create(Parser* parser, OpDst dst, OpSrc src, OpSrc index, OperandType_t type) {
  auto value0 = parser->loadRegister(src, type);
  auto value1 = parser->loadRegister(index, type);

  auto bitMask    = parser->create<mlir::arith::ConstantIntOp>(type, type.getIntOrFloatBitWidth() > 32 ? 0b111111 : 0b11111);
  auto indexValue = parser->create<mlir::spirv::BitwiseAndOp>(parser->types().i32(), value1, bitMask);

  auto bitWidth   = mlir::spirv::ConstantOp::getOne(type, parser->getLoc(), parser->getBuilder());
  auto bitExtract = parser->create<mlir::spirv::BitFieldUExtractOp>(parser->types().i32(), value0, indexValue, bitWidth);

  auto zero = mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder());
  auto res  = parser->create<mlir::spirv::INotEqualOp>(bitExtract, zero);

  parser->storeRegister(dst, res);
}

void IsBitClearOp::create(Parser* parser, OpDst dst, OpSrc src, OpSrc index, OperandType_t type) {
  auto value0 = parser->loadRegister(src, type);
  auto value1 = parser->loadRegister(index, type);

  auto bitMask    = parser->create<mlir::arith::ConstantIntOp>(type, type.getIntOrFloatBitWidth() > 32 ? 0b111111 : 0b11111);
  auto indexValue = parser->create<mlir::spirv::BitwiseAndOp>(parser->types().i32(), value1, bitMask);

  auto bitWidth   = mlir::spirv::ConstantOp::getOne(type, parser->getLoc(), parser->getBuilder());
  auto bitExtract = parser->create<mlir::spirv::BitFieldUExtractOp>(parser->types().i32(), value0, indexValue, bitWidth);

  auto zero = mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder());
  auto res  = parser->create<mlir::spirv::IEqualOp>(bitExtract, zero);

  parser->storeRegister(dst, res);
}

void AndIOp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type) {
  auto value0 = parser->loadRegister(src0, type);
  auto value1 = parser->loadRegister(src1, type);

  mlir::Value res = parser->create<mlir::spirv::BitwiseAndOp>(type, value0, value1);
  res             = parser->storeRegister(dst, res);

  auto zero       = mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder());
  auto carryValue = parser->create<mlir::spirv::INotEqualOp>(res, zero);
  parser->storeRegister(carry, carryValue);
}

void OrIOp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type) {
  auto value0 = parser->loadRegister(src0, type);
  auto value1 = parser->loadRegister(src1, type);

  mlir::Value res = parser->create<mlir::spirv::BitwiseOrOp>(type, value0, value1);
  res             = parser->storeRegister(dst, res);

  auto zero       = mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder());
  auto carryValue = parser->create<mlir::spirv::INotEqualOp>(res, zero);
  parser->storeRegister(carry, carryValue);
}

void XorIOp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type) {
  auto value0 = parser->loadRegister(src0, type);
  auto value1 = parser->loadRegister(src1, type);

  mlir::Value res = parser->create<mlir::spirv::BitwiseXorOp>(type, value0, value1);
  res             = parser->storeRegister(dst, res);

  auto zero       = mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder());
  auto carryValue = parser->create<mlir::spirv::INotEqualOp>(res, zero);
  parser->storeRegister(carry, carryValue);
}

void LSHLOp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type) {
  auto value0     = parser->loadRegister(src0, type);
  auto shiftValue = parser->loadRegister(src1, parser->types().i32());

  auto bitMask = parser->create<mlir::arith::ConstantIntOp>(parser->types().i32(), type.getIntOrFloatBitWidth() > 32 ? 0b111111 : 0b11111);
  shiftValue   = parser->create<mlir::spirv::BitwiseAndOp>(parser->types().i32(), shiftValue, bitMask);

  mlir::Value res = parser->create<mlir::spirv::ShiftLeftLogicalOp>(type, value0, shiftValue);
  res             = parser->storeRegister(dst, res);

  auto zero       = mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder());
  auto carryValue = parser->create<mlir::spirv::INotEqualOp>(res, zero);
  parser->storeRegister(carry, carryValue);
}

void LSHROp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type) {
  auto value0     = parser->loadRegister(src0, type);
  auto shiftValue = parser->loadRegister(src1, parser->types().i32());

  auto bitMask = parser->create<mlir::arith::ConstantIntOp>(parser->types().i32(), type.getIntOrFloatBitWidth() > 32 ? 0b111111 : 0b11111);
  shiftValue   = parser->create<mlir::spirv::BitwiseAndOp>(parser->types().i32(), shiftValue, bitMask);

  mlir::Value res = parser->create<mlir::spirv::ShiftRightLogicalOp>(type, value0, shiftValue);
  res             = parser->storeRegister(dst, res);

  auto zero       = mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder());
  auto carryValue = parser->create<mlir::spirv::INotEqualOp>(res, zero);
  parser->storeRegister(carry, carryValue);
}

void ASHROp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type) {
  auto value0     = parser->loadRegister(src0, type);
  auto shiftValue = parser->loadRegister(src1, parser->types().i32());

  auto bitMask = parser->create<mlir::arith::ConstantIntOp>(parser->types().i32(), type.getIntOrFloatBitWidth() > 32 ? 0b111111 : 0b11111);
  shiftValue   = parser->create<mlir::spirv::BitwiseAndOp>(parser->types().i32(), shiftValue, bitMask);

  mlir::Value res = parser->create<mlir::spirv::ShiftRightArithmeticOp>(type, value0, shiftValue);
  res             = parser->storeRegister(dst, res);

  auto zero       = mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder());
  auto carryValue = parser->create<mlir::spirv::INotEqualOp>(res, zero);
  parser->storeRegister(carry, carryValue);
}

void BitfieldMaskOp::create(Parser* parser, OpDst dst, OpSrc width, OpSrc offset, OperandType_t type) {
  auto widthValue  = parser->loadRegister(width, parser->types().i32());
  auto offsetValue = parser->loadRegister(offset, parser->types().i32());

  auto bitMask = parser->create<mlir::arith::ConstantIntOp>(parser->types().i32(), type.getIntOrFloatBitWidth() > 32 ? 0b111111 : 0b11111);
  widthValue   = parser->create<mlir::spirv::BitwiseAndOp>(parser->types().i32(), widthValue, bitMask);
  offsetValue  = parser->create<mlir::spirv::BitwiseAndOp>(parser->types().i32(), offsetValue, bitMask);

  auto        zero  = mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder());
  auto        value = parser->create<mlir::arith::ConstantIntOp>(type, -1);
  mlir::Value res   = parser->create<mlir::spirv::BitFieldInsertOp>(type, zero, value, widthValue, offsetValue);
  res               = parser->storeRegister(dst, res);
}

void BitfieldExtractUIOp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src, OpSrc packed, OperandType_t type) {
  auto srcValue    = parser->loadRegister(src, type);
  auto packedValue = parser->loadRegister(packed, parser->types().i32());

  auto zero        = mlir::spirv::ConstantOp::getZero(parser->types().i32(), parser->getLoc(), parser->getBuilder());
  auto bitwidth    = parser->create<mlir::arith::ConstantIntOp>(parser->types().i32(), type.getIntOrFloatBitWidth() > 32 ? 6 : 5);
  auto widthOffset = parser->create<mlir::arith::ConstantIntOp>(parser->types().i32(), 16);
  auto offset      = parser->create<mlir::spirv::BitFieldUExtractOp>(parser->types().i32(), packedValue, bitwidth, zero);
  auto width       = parser->create<mlir::spirv::BitFieldUExtractOp>(parser->types().i32(), packedValue, bitwidth, widthOffset);

  mlir::Value res = parser->create<mlir::spirv::BitFieldUExtractOp>(type, srcValue, width, offset);
  res             = parser->storeRegister(dst, res);

  auto carryValue = parser->create<mlir::spirv::INotEqualOp>(res, mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder()));
  parser->storeRegister(carry, carryValue);
}

void BitfieldExtractSIOp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src, OpSrc packed, OperandType_t type) {
  auto srcValue    = parser->loadRegister(src, type);
  auto packedValue = parser->loadRegister(packed, parser->types().i32());

  auto zero        = mlir::spirv::ConstantOp::getZero(parser->types().i32(), parser->getLoc(), parser->getBuilder());
  auto bitwidth    = parser->create<mlir::arith::ConstantIntOp>(parser->types().i32(), type.getIntOrFloatBitWidth() > 32 ? 6 : 5);
  auto widthOffset = parser->create<mlir::arith::ConstantIntOp>(parser->types().i32(), 16);
  auto offset      = parser->create<mlir::spirv::BitFieldUExtractOp>(parser->types().i32(), packedValue, bitwidth, zero);
  auto width       = parser->create<mlir::spirv::BitFieldUExtractOp>(parser->types().i32(), packedValue, bitwidth, widthOffset);

  mlir::Value res = parser->create<mlir::spirv::BitFieldSExtractOp>(type, srcValue, width, offset);
  res             = parser->storeRegister(dst, res);

  auto carryValue = parser->create<mlir::spirv::INotEqualOp>(res, mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder()));
  parser->storeRegister(carry, carryValue);
}

void BitfieldInsertOp::create(Parser* parser, OpDst dst, OpSrc src, OpSrc value, OpSrc width, OpSrc offset, OperandType_t type) {
  auto srcValue    = parser->loadRegister(src, type);
  auto insertValue = parser->loadRegister(value, type);
  auto widthValue  = parser->loadRegister(width, parser->types().i32());
  auto offsetValue = parser->loadRegister(offset, parser->types().i32());

  mlir::Value res = parser->create<mlir::spirv::BitFieldInsertOp>(type, srcValue, insertValue, widthValue, offsetValue);
  res             = parser->storeRegister(dst, res);
}

void AbsDiffIOp::create(Parser* parser, OpDst dst, OpDst carry, OpSrc src0, OpSrc src1, OperandType_t type) {
  auto value0 = parser->loadRegister(src0, type);
  auto value1 = parser->loadRegister(src1, type);

  mlir::Value res = parser->create<mlir::spirv::ISubOp>(type, value0, value1);
  res             = parser->storeRegister(dst, res);

  auto zero       = mlir::spirv::ConstantOp::getZero(type, parser->getLoc(), parser->getBuilder());
  auto carryValue = parser->create<mlir::spirv::INotEqualOp>(res, zero);
  parser->storeRegister(carry, carryValue);
}

} // namespace compiler::frontend::op