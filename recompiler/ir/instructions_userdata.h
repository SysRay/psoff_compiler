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
}