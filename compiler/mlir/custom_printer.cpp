#include "../frontend/gfx/register_types.h"
#include "custom.h"

namespace mlir::psoff {

static bool is64BitType(mlir::Type ty) {
  return ty.getIntOrFloatBitWidth() == 64;
}

static void printOperandKind(mlir::OpAsmPrinter& p, compiler::frontend::eOperandKind kind, mlir::Type type) {
  auto const is64bit = is64BitType(type);

  if (kind.isConstI()) {
    p << kind.getConstI();
  } else if (kind.isConstF()) {
    p << kind.getConstF();
  } else if (kind.isSGPR()) {
    p << "s" << "[" << kind.getSGPR() << "]";
  } else if (kind.isVGPR()) {
    p << "s" << "[" << kind.getVGPR() << "]";
  } else {
    using namespace compiler::frontend;
    if (is64bit) {
      switch (kind.base()) {
        case eOperandKind::eBase::VccLo: p << "VCC"; break;
        case eOperandKind::eBase::CUSTOM_UNSET: p << "NOT_SET"; break;
        case eOperandKind::eBase::ExecLo: p << "EXEC"; break;
        case eOperandKind::eBase::Literal: p << "LITERAL"; break;
        default: p << "UNK" << (uint16_t)kind.base(); break;
      }
    } else {
      switch (kind.base()) {
        case eOperandKind::eBase::VccLo: p << "VCC_LO"; break;
        case eOperandKind::eBase::VccHi: p << "VCC_HI"; break;
        case eOperandKind::eBase::M0: p << "M0"; break;
        case eOperandKind::eBase::CustomVskip: p << "VSKIP"; break;
        case eOperandKind::eBase::ExecLo: p << "EXEC_LO"; break;
        case eOperandKind::eBase::ExecHi: p << "EXEC_HI"; break;
        case eOperandKind::eBase::INV_2PI: p << "INV_2PI"; break;
        case eOperandKind::eBase::SDWA: p << "SDWA"; break;
        case eOperandKind::eBase::DPP: p << "DPP"; break;
        case eOperandKind::eBase::VccZ: p << "VCCZ"; break;
        case eOperandKind::eBase::ExecZ: p << "EXECZ"; break;
        case eOperandKind::eBase::Scc: p << "SCC"; break;
        case eOperandKind::eBase::LdsDirect: p << "DIRECT"; break;
        case eOperandKind::eBase::Literal: p << "LITERAL"; break;
        default: p << "UNK" << (uint16_t)kind.base(); break;
      }
    }
  }
}

compiler::frontend::eOperandKind parseOperandKind(std::string_view kindToken, uint32_t index) {
  using namespace compiler::frontend;

  eOperandKind kind = eOperandKind::SGPR(0);
  if (kindToken == "VCC")
    kind = eOperandKind::VCC();
  else if (kindToken == "EXEC")
    kind = eOperandKind::EXEC();
  else if (kindToken == "LITERAL")
    kind = eOperandKind::Literal();
  else if (kindToken == "VCC_LO")
    kind = eOperandKind((eOperandKind_t)eOperandKind::eBase::VccLo);
  else if (kindToken == "VCC_HI")
    kind = eOperandKind((eOperandKind_t)eOperandKind::eBase::VccHi);
  else if (kindToken == "M0")
    kind = eOperandKind((eOperandKind_t)eOperandKind::eBase::M0);
  else if (kindToken == "VSKIP")
    kind = eOperandKind((eOperandKind_t)eOperandKind::eBase::CustomVskip);
  else if (kindToken == "EXEC_LO")
    kind = eOperandKind((eOperandKind_t)eOperandKind::eBase::ExecLo);
  else if (kindToken == "EXEC_HI")
    kind = eOperandKind((eOperandKind_t)eOperandKind::eBase::ExecHi);
  else if (kindToken == "INV_2PI")
    kind = eOperandKind((eOperandKind_t)eOperandKind::eBase::INV_2PI);
  else if (kindToken == "SDWA")
    kind = eOperandKind((eOperandKind_t)eOperandKind::eBase::SDWA);
  else if (kindToken == "DPP")
    kind = eOperandKind((eOperandKind_t)eOperandKind::eBase::DPP);
  else if (kindToken == "VCCZ")
    kind = eOperandKind((eOperandKind_t)eOperandKind::eBase::VccZ);
  else if (kindToken == "EXECZ")
    kind = eOperandKind((eOperandKind_t)eOperandKind::eBase::ExecZ);
  else if (kindToken == "SCC")
    kind = eOperandKind((eOperandKind_t)eOperandKind::eBase::Scc);
  else if (kindToken == "DIRECT")
    kind = eOperandKind((eOperandKind_t)eOperandKind::eBase::LdsDirect);
  else if (kindToken.starts_with("s")) {
    kind = eOperandKind::SGPR(index);
  } else if (kindToken.starts_with("v")) {
    kind = eOperandKind::VGPR(index);
  } else {
    kind = eOperandKind::Unset();
  }

  return kind;
}

void LoadOp::print(mlir::OpAsmPrinter& p) {
  auto const kind = compiler::frontend::eOperandKind((compiler::frontend::eOperandKind_t)getId().getZExtValue());

  p << " ";
  printOperandKind(p, kind, getType());
  p << " : ";
  p.printType(getType());
}

mlir::ParseResult LoadOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result) {
  using namespace compiler::frontend;

  std::string kindToken;
  if (parser.parseKeywordOrString(&kindToken)) return mlir::failure();

  uint32_t index;
  if (parser.parseOptionalLSquare()) {
    if (parser.parseInteger(index) || parser.parseOptionalRSquare()) return failure();
  }

  if (parser.parseColon()) return mlir::failure();

  mlir::Type valType;
  if (parser.parseType(valType)) return mlir::failure();

  eOperandKind kind = parseOperandKind(kindToken, index);
  if (kind.raw() == eOperandKind::Unset().raw()) {
    return parser.emitError(parser.getCurrentLocation()) << "unknown operand literal: " << kindToken;
  }

  auto idAttr = parser.getBuilder().getI32IntegerAttr(static_cast<uint32_t>(kind.raw()));
  result.addAttribute(getIdAttrName(result.name), idAttr);
  result.addTypes(valType);

  return mlir::success();
}

void StoreOp::print(mlir::OpAsmPrinter& p) {
  auto const kind = compiler::frontend::eOperandKind((compiler::frontend::eOperandKind_t)getId().getZExtValue());

  p << " ";
  printOperandKind(p, kind, getVal().getType());
  p << " = ";
  p.printOperand(getVal());
  p << " : ";
  p.printType(getVal().getType());
}

mlir::ParseResult StoreOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result) {
  using namespace compiler::frontend;

  std::string kindToken;
  if (parser.parseKeywordOrString(&kindToken)) return mlir::failure();

  uint32_t index;
  if (parser.parseOptionalLSquare()) {
    if (parser.parseInteger(index) || parser.parseOptionalRSquare()) return failure();
  }
  eOperandKind kind = parseOperandKind(kindToken, index);
  if (kind.raw() == eOperandKind::Unset().raw()) {
    return parser.emitError(parser.getCurrentLocation()) << "unknown operand literal: " << kindToken;
  }

  if (parser.parseEqual()) return mlir::failure();

  mlir::OpAsmParser::UnresolvedOperand valOperand;
  if (parser.parseOperand(valOperand)) return mlir::failure();

  if (parser.parseColon()) return mlir::failure();

  mlir::Type valType;
  if (parser.parseType(valType)) return mlir::failure();
  if (parser.resolveOperand(valOperand, valType, result.operands)) return mlir::failure();

  auto idAttr = parser.getBuilder().getI32IntegerAttr(static_cast<uint32_t>(kind.raw()));
  result.addAttribute(getIdAttrName(result.name), idAttr);

  return mlir::success();
}

} // namespace mlir::psoff