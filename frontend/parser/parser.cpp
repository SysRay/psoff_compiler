#include "parser.h"

#include "opcodes_table.h"
#include "translate/encodings.h"
#include "translate/translate.h"

namespace compiler::frontend::parser {
enum eEncodingMask : uint32_t {
  eEncodingMask_9b = 0xff800000,
  eEncodingMask_7b = 0xfe000000,
  eEncodingMask_6b = 0xfc000000,
  eEncodingMask_5b = 0xf8000000,
  eEncodingMask_4b = 0xf0000000,
  eEncodingMask_2b = 0xc0000000,
  eEncodingMask_1b = 0x80000000,
};

constexpr uint32_t getEncodingBits(eEncoding encoding) {
  switch (encoding) {
    case eEncoding::SOP1: return 0xbe800000;
    case eEncoding::SOP2: return 0x80000000;
    case eEncoding::SOPP: return 0xbf800000;
    case eEncoding::SOPC: return 0xbf000000;
    case eEncoding::EXP: return 0xc4000000;
    case eEncoding::VINTRP: return 0xd4000000;
    case eEncoding::DS: return 0xd8000000;
    case eEncoding::MUBUF: return 0xe0000000;
    case eEncoding::MTBUF: return 0xe8000000;
    case eEncoding::MIMG: return 0xf0000000;
    case eEncoding::SMRD: return 0xc0000000;
    case eEncoding::SOPK: return 0xb0000000;
    case eEncoding::VOP1: return 0x7e000000;
    case eEncoding::VOP2: return 0x00000000;
    case eEncoding::VOP3: return 0xd0000000;
    case eEncoding::VOPC: return 0x7c000000;
    default: return 0;
  }
}

static eEncoding getEncoding(code_t code) {
  switch (code & eEncodingMask_9b) {
    case getEncodingBits(eEncoding::SOP1): return eEncoding::SOP1;
    case getEncodingBits(eEncoding::SOPP): return eEncoding::SOPP;
    case getEncodingBits(eEncoding::SOPC): return eEncoding::SOPC;
    default: break;
  }
  switch (code & eEncodingMask_7b) {
    case getEncodingBits(eEncoding::VOP1): return eEncoding::VOP1;
    case getEncodingBits(eEncoding::VOPC): return eEncoding::VOPC;
    default: break;
  }
  switch (code & eEncodingMask_6b) {
    case getEncodingBits(eEncoding::VOP3): return eEncoding::VOP3;
    case getEncodingBits(eEncoding::EXP): return eEncoding::EXP;
    case getEncodingBits(eEncoding::VINTRP): return eEncoding::VINTRP;
    case getEncodingBits(eEncoding::DS): return eEncoding::DS;
    case getEncodingBits(eEncoding::MUBUF): return eEncoding::MUBUF;
    case getEncodingBits(eEncoding::MTBUF): return eEncoding::MTBUF;
    case getEncodingBits(eEncoding::MIMG): return eEncoding::MIMG;
    default: break;
  }
  switch (code & eEncodingMask_5b) {
    case getEncodingBits(eEncoding::SMRD): return eEncoding::SMRD;
    default: break;
  }
  switch (code & eEncodingMask_4b) {
    case getEncodingBits(eEncoding::SOPK): return eEncoding::SOPK;
    default: break;
  }
  switch (code & eEncodingMask_2b) {
    case getEncodingBits(eEncoding::SOP2): return eEncoding::SOP2;
    default: break;
  }
  switch (code & eEncodingMask_1b) {
    case getEncodingBits(eEncoding::VOP2): return eEncoding::VOP2;
    default: break;
  }
  return eEncoding::UNK;
}

ir::InstCore parseInstruction(Builder& builder, pc_t pc, code_p_t* pCode) {
  using namespace translate;
  switch (getEncoding(**pCode)) {
    case eEncoding::SOP1: return handleSop1(builder, pc, pCode);
    case eEncoding::SOP2: return handleSop2(builder, pc, pCode);
    case eEncoding::SOPP: return handleSopp(builder, pc, pCode);
    case eEncoding::SOPC: return handleSopc(builder, pc, pCode);
    case eEncoding::EXP: return handleExp(builder, pc, pCode);
    case eEncoding::VINTRP: return handleVintrp(builder, pc, pCode);
    case eEncoding::DS: return handleDs(builder, pc, pCode);
    case eEncoding::MUBUF: return handleMubuf(builder, pc, pCode);
    case eEncoding::MTBUF: return handleMtbuf(builder, pc, pCode);
    case eEncoding::MIMG: return handleMimg(builder, pc, pCode);
    case eEncoding::SMRD: return handleSmrd(builder, pc, pCode);
    case eEncoding::SOPK: return handleSopk(builder, pc, pCode);
    case eEncoding::VOP1: return handleVop1(builder, pc, pCode, false);
    case eEncoding::VOP2: return handleVop2(builder, pc, pCode, false);
    case eEncoding::VOP3: {
      auto const header = VOP3(*(codeE_p_t)*pCode);
      auto const op     = header.get<VOP3::Field::OP>();
      if (op >= 0x180) return handleVop1(builder, pc, pCode, true);
      if (op >= 0x140) return handleVop3(builder, pc, pCode);
      if (op >= 0x100) return handleVop2(builder, pc, pCode, true);
      return handleVopc(builder, pc, pCode, true);
    }
    case eEncoding::VOPC: return handleVopc(builder, pc, pCode, false);
    default: return {};
  }
}
} // namespace compiler::frontend::parser