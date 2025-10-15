#include "debug_strings.h"

#include "ir/ir.h"
#include "ir_types.h"
#include "parser/shader_types.h"

#include <iostream>

namespace compiler::frontend::debug {

static void printKind(std::ostream& os, eOperandKind kind, uint16_t numdw, bool is64bit) {
  if (kind.isConstI()) {
    os << kind.getConstI();
  } else if (kind.isConstF()) {
    os << kind.getConstF();
  } else if (kind.isSGPR()) {
    auto const start = kind.getSGPR();
    if (numdw == 1)
      os << "s" << std::dec << start;
    else {
      auto const end = start + numdw - 1;
      os << "s[" << std::dec << start << ":" << end << "]";
    }
  } else if (kind.isVGPR()) {
    auto const start = kind.getVGPR();
    if (numdw == 1)
      os << "v" << std::dec << start;
    else {
      auto const end = start + numdw - 1;
      os << "v[" << std::dec << start << ":" << end << "]";
    }
  } else {
    if (is64bit) {
      switch (kind.base()) {
        case eOperandKind::eBase::VccLo: os << "VCC"; break;
        case eOperandKind::eBase::CustomTemp0Lo: os << "TEMP0"; break;
        case eOperandKind::eBase::CustomTemp1Lo: os << "TEMP1"; break;
        case eOperandKind::eBase::ExecLo: os << "EXEC"; break;
        case eOperandKind::eBase::Literal: os << "LITERAL"; break;
        default: os << "UNK" << std::dec << (uint16_t)kind.base(); break;
      }
    } else {
      switch (kind.base()) {
        case eOperandKind::eBase::VccLo: os << "VCC_LO"; break;
        case eOperandKind::eBase::VccHi: os << "VCC_HI"; break;
        case eOperandKind::eBase::CustomTemp0Lo: os << "TEMP0_LO"; break;
        case eOperandKind::eBase::CustomTemp0Hi: os << "TEMP0_HI"; break;
        case eOperandKind::eBase::CustomTemp1Lo: os << "TEMP1_LO"; break;
        case eOperandKind::eBase::CustomTemp1Hi: os << "TEMP1_HI"; break;
        case eOperandKind::eBase::M0: os << "M0"; break;
        case eOperandKind::eBase::CustomVskip: os << "VSKIP"; break;
        case eOperandKind::eBase::ExecLo: os << "EXEC_LO"; break;
        case eOperandKind::eBase::ExecHi: os << "EXEC_HI"; break;
        case eOperandKind::eBase::INV_2PI: os << "INV_2PI"; break;
        case eOperandKind::eBase::SDWA: os << "SDWA"; break;
        case eOperandKind::eBase::DPP: os << "DPP"; break;
        case eOperandKind::eBase::VccZ: os << "VCCZ"; break;
        case eOperandKind::eBase::ExecZ: os << "EXECZ"; break;
        case eOperandKind::eBase::Scc: os << "SCC"; break;
        case eOperandKind::eBase::LdsDirect: os << "DIRECT"; break;
        case eOperandKind::eBase::Literal: os << "LITERAL"; break;
        default: os << "UNK" << std::dec << (uint16_t)kind.base(); break;
      }
    }
  }
}

void printType(std::ostream& os, ir::OperandType type) {
  if (type.is_array()) os << "arr";
  if (type.is_vector()) os << "vec";
  if (type.cols() > 1) {
    os << type.cols() << "x";
  }
  switch (type.base()) {
    case ir::OperandType::eBase::i1: os << "i1"; break;
    case ir::OperandType::eBase::i8: os << "i8"; break;
    case ir::OperandType::eBase::i16: os << "i16"; break;
    case ir::OperandType::eBase::i32: os << "i32"; break;
    case ir::OperandType::eBase::i64: os << "i64"; break;
    case ir::OperandType::eBase::f32: os << "f32"; break;
    case ir::OperandType::eBase::f64: os << "f64"; break;
  }
}

static uint16_t getNumDw(ir::OperandType type) {
  switch (type.base()) {
    case ir::OperandType::eBase::i1: return 1 * type.cols();
    case ir::OperandType::eBase::i8: return 1 * type.cols();
    case ir::OperandType::eBase::i16: return 1 * type.cols();
    case ir::OperandType::eBase::i32: return 1 * type.cols();
    case ir::OperandType::eBase::i64: return 2 * type.cols();
    case ir::OperandType::eBase::f32: return 1 * type.cols();
    case ir::OperandType::eBase::f64: return 2 * type.cols();
  }
}

void printOperandDst(std::ostream& os, ir::Operand const& op) {
  auto const flags = OperandFlagsDst(op.flags);
  auto       kind  = eOperandKind::import(op.kind);

  if (flags.getNegate()) { // custom type
    os << "-(";
  }
  if (flags.getClamp()) {
    os << "clamp{";
  }
  if (flags.hasMultiply()) {
    os << flags.getMultiply() << "*";
  }
  printKind(os, kind, getNumDw(op.type), op.type.is_64bit());

  if (flags.getClamp()) {
    os << "}";
  }
  if (flags.getNegate()) { // custom type
    os << ")";
  }
}

void printOperandSrc(std::ostream& os, ir::Operand const& op) {
  auto const flags = OperandFlagsSrc(op.flags);
  auto       kind  = eOperandKind::import(op.kind);

  if (flags.getNegate()) {
    os << "-(";
  }
  if (flags.getAbsolute()) {
    os << "abs(";
  }

  printKind(os, kind, getNumDw(op.type), op.type.is_64bit());

  if (flags.getAbsolute()) {
    os << ")";
  }
  if (flags.getNegate()) {
    os << ")";
  }
}

} // namespace compiler::frontend::debug