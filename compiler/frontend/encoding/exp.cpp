#include "../gfx/encoding_types.h"
#include "../parser.h"
#include "opcodes_table.h"

#include <bitset>
#include <format>
#include <stdexcept>

// mlir
#include "mlir/custom.h"

#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>

namespace compiler::frontend {

static mlir::Value getData(Parser& parser, EXP inst) {
  std::bitset<4> const enabled      = inst.template get<EXP::Field::EN>();
  bool const           isCompressed = inst.template get<EXP::Field::COMPR>();

  auto src0 = OpSrc(eOperandKind::VGPR(inst.template get<EXP::Field::VSRC0>()));
  auto src1 = OpSrc(eOperandKind::VGPR(inst.template get<EXP::Field::VSRC1>()));

  auto zero = mlir::spirv::ConstantOp::getZero(parser.types().f32(), parser.getLoc(), parser.getBuilder());

  std::array<mlir::Value, 4> items = {zero, zero, zero, zero};

  if (isCompressed) {
    if (enabled[0]) {
      auto vec = parser.create<mlir::spirv::GLUnpackHalf2x16Op>(parser.types().vec2xf32(), parser.loadRegister(src0, parser.types().i32()));
      items[0] = parser.create<mlir::spirv::CompositeExtractOp>(vec, llvm::ArrayRef(0));
      items[1] = parser.create<mlir::spirv::CompositeExtractOp>(vec, llvm::ArrayRef(1));
    }

    if (enabled[1]) {
      auto vec = parser.create<mlir::spirv::GLUnpackHalf2x16Op>(parser.types().vec2xf32(), parser.loadRegister(src1, parser.types().i32()));
      items[2] = parser.create<mlir::spirv::CompositeExtractOp>(vec, llvm::ArrayRef(0));
      items[3] = parser.create<mlir::spirv::CompositeExtractOp>(vec, llvm::ArrayRef(1));
    }
  } else {
    auto src2 = OpSrc(eOperandKind::VGPR(inst.template get<EXP::Field::VSRC2>()));
    auto src3 = OpSrc(eOperandKind::VGPR(inst.template get<EXP::Field::VSRC2>()));

    if (enabled[0]) items[0] = parser.loadRegister(src0, parser.types().f32());
    if (enabled[1]) items[1] = parser.loadRegister(src1, parser.types().f32());
    if (enabled[2]) items[2] = parser.loadRegister(src2, parser.types().f32());
    if (enabled[3]) items[3] = parser.loadRegister(src3, parser.types().f32());
  }

  return parser.create<mlir::spirv::CompositeConstructOp>(parser.types().vec4xf32(), mlir::ValueRange(items));
}

uint8_t Parser::handleExp(CodeBlock& cb, pc_t pc, uint32_t const* pCode) {
  auto inst = EXP(getU64(pCode));

  std::bitset<4> const enable       = inst.template get<EXP::Field::EN>();
  uint8_t const        target       = inst.template get<EXP::Field::TGT>();
  bool const           isCompressed = inst.template get<EXP::Field::COMPR>();
  bool const           isLast       = inst.template get<EXP::Field::DONE>();

  auto src0 = eOperandKind::VGPR(inst.template get<EXP::Field::VSRC0>());
  auto src1 = eOperandKind::VGPR(inst.template get<EXP::Field::VSRC1>());
  auto src2 = eOperandKind::VGPR(inst.template get<EXP::Field::VSRC2>());
  auto src3 = eOperandKind::VGPR(inst.template get<EXP::Field::VSRC2>());

  if (target >= 0x0 && target <= 0x7) { // Attachments
    bool const useExecMask = inst.template get<EXP::Field::VM>();

    auto execValue =
        useExecMask ? loadRegister(OpSrc(eOperandKind::EXEC()), types().i1()) : mlir::spirv::ConstantOp::getOne(types().i1(), getLoc(), getBuilder());
    create<mlir::psoff::WriteOutputOp>(target, execValue, getData(*this, inst));
  } else if (target == 0x8 && enable.any()) { // Write to depth

  } else if (target >= 0xc && target <= 0xf) { // Vertex output
    auto const pos = target - 0xc;

  } else if (target >= 0x20 && target <= 0x3F) { //  output params
    auto const pos = target - 0x20;

    auto execValue = mlir::spirv::ConstantOp::getOne(types().i1(), getLoc(), getBuilder());
    create<mlir::psoff::WriteOutputOp>(pos, execValue, getData(*this, inst));
  }

  return sizeof(uint64_t);
}

} // namespace compiler::frontend