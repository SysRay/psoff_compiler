#pragma once
#include "ir/config.h"

namespace compiler::frontend::liverpool {
enum class OperandKind : OperandKind_t {
  SGPR  = 0,
  VccLo = 106,
  VccHi,
  M0     = 124,
  ExecLo = 126,
  ExecHi,
  ConstZero,
  ConstInt     = 129,
  ConstUInt    = 193,
  ConstFloat   = 240,
  VccZ         = 251,
  ExecZ        = 252,
  Scc          = 253,
  LdsDirect    = 254,
  LiteralConst = 255,
  VGPR,
};
}