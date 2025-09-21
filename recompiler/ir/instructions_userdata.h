#pragma once

#include "config.h"

namespace compiler {
enum class CmpIPredicate : InstructionUserData_t {
  eq  = 0,
  ne  = 1,
  slt = 2,
  sle = 3,
  sgt = 4,
  sge = 5,
  ult = 6,
  ule = 7,
  ugt = 8,
  uge = 9,
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