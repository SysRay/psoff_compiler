#pragma once

#include "config.h"

namespace compiler {
enum class CmpIPredicate : InstructionUserData_t {
  AlwaysFalse = 0,
  eq          = 1,
  ne          = 2,
  slt         = 3,
  sle         = 4,
  sgt         = 5,
  sge         = 6,
  ult         = 7,
  ule         = 8,
  ugt         = 9,
  uge         = 10,
  AlwaysTrue  = 11,
};

enum class CmpFPredicate : InstructionUserData_t {
  AlwaysFalse = 0,
  OEQ         = 1,
  OGT         = 2,
  OGE         = 3,
  OLT         = 4,
  OLE         = 5,
  ONE         = 6,
  ORD         = 7,
  UEQ         = 8,
  UGT         = 9,
  UGE         = 10,
  ULT         = 11,
  ULE         = 12,
  UNE         = 13,
  UNO         = 14,
  AlwaysTrue  = 15,
};

} // namespace compiler