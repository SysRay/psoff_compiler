#pragma once

#include "frontend/frontend.h"
#include "ir/instructions.h"
#include "ir/instructions_userdata.h"

namespace compiler::frontend::translate {

struct OperandSrc {
  eOperandKind    kind;
  OperandFlagsSrc flags;
};

struct OperandDst {
  eOperandKind    kind;
  OperandFlagsDst flags;
};

namespace create {
ir::InstCore literalOp(uint32_t);
ir::InstCore constantOp(eOperandKind dst, uint64_t, ir::OperandType type);
ir::InstCore constantOp(eOperandKind dst, int16_t, ir::OperandType type);
ir::InstCore moveOp(eOperandKind dst, eOperandKind src, ir::OperandType type);
ir::InstCore selectOp(eOperandKind dst, eOperandKind predicate, eOperandKind srcTrue, eOperandKind srcFalse, ir::OperandType type);
ir::InstCore notOp(eOperandKind dst, eOperandKind src, ir::OperandType type);
ir::InstCore absOp(eOperandKind dst, eOperandKind src, ir::OperandType type);
ir::InstCore bitReverseOp(eOperandKind dst, eOperandKind src, ir::OperandType type);
ir::InstCore bitCountOp(eOperandKind dst, eOperandKind src, ir::OperandType type, bool value);
ir::InstCore bitFieldMaskOp(eOperandKind dst, eOperandKind size, eOperandKind offset, ir::OperandType type);
ir::InstCore bitAndOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type);
ir::InstCore bitAndNOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type);
ir::InstCore bitNandOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type);
ir::InstCore bitOrOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type);
ir::InstCore bitOrNOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type);
ir::InstCore bitNorOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type);
ir::InstCore bitXorOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type);
ir::InstCore bitXnorOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type);
ir::InstCore findILsbOp(eOperandKind dst, eOperandKind src, ir::OperandType type, bool value);
ir::InstCore findUMsbOp(eOperandKind dst, eOperandKind src, ir::OperandType type);
ir::InstCore findSMsbOp(eOperandKind dst, eOperandKind src, ir::OperandType type);
ir::InstCore signExtendI32Op(eOperandKind dst, eOperandKind src, ir::OperandType type);
ir::InstCore bitsetOp(eOperandKind dst, eOperandKind src, eOperandKind offset, eOperandKind value, ir::OperandType type);
ir::InstCore bitFieldInsertOp(eOperandKind dst, eOperandKind value, eOperandKind offset, eOperandKind count, ir::OperandType type);
ir::InstCore bitUIExtractOp(eOperandKind dst, eOperandKind base, eOperandKind offset, eOperandKind count, ir::OperandType type);
ir::InstCore bitSIExtractOp(eOperandKind dst, eOperandKind base, eOperandKind offset, eOperandKind count, ir::OperandType type);
ir::InstCore bitUIExtractOp(eOperandKind dst, eOperandKind base, eOperandKind compact, ir::OperandType type);
ir::InstCore bitSIExtractOp(eOperandKind dst, eOperandKind base, eOperandKind compact, ir::OperandType type);
ir::InstCore bitCmpOp(eOperandKind dst, eOperandKind base, ir::OperandType type, eOperandKind index, bool value);
ir::InstCore cmpIOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type, CmpIPredicate op);

ir::InstCore shiftLUIOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type);
ir::InstCore shiftRUIOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type);
ir::InstCore shiftRSIOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type);

// // arith
ir::InstCore mulIOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type);
ir::InstCore addIOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type);
ir::InstCore addcIOp(eOperandKind dst, eOperandKind carryOut, eOperandKind src0, eOperandKind src1, eOperandKind carryIn, ir::OperandType type);
ir::InstCore subIOp(eOperandKind dst, eOperandKind src0, eOperandKind src1, ir::OperandType type);
ir::InstCore subbIOp(eOperandKind dst, eOperandKind carryOut, eOperandKind src0, eOperandKind src1, eOperandKind carryIn, ir::OperandType type);
// // Flow control
ir::InstCore jumpAbsOp(eOperandKind addr);
ir::InstCore jumpAbsOp(uint64_t addr);
ir::InstCore cjumpAbsOp(eOperandKind predicate, bool invert, eOperandKind addr);
} // namespace create
} // namespace compiler::frontend::translate