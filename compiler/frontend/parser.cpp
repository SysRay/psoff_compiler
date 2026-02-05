#include "parser.h"

#include "builder.h"
#include "encoding/opcodes_table.h"
#include "gfx/encoding_types.h"
#include "logging.h"

#include <limits>

namespace compiler::frontend {

enum eEncodingMask : uint32_t {
  eEncodingMask_9b = 0xff800000,
  eEncodingMask_7b = 0xfe000000,
  eEncodingMask_6b = 0xfc000000,
  eEncodingMask_5b = 0xf8000000,
  eEncodingMask_4b = 0xf0000000,
  eEncodingMask_2b = 0xc0000000,
  eEncodingMask_1b = 0x80000000,
};

static constexpr uint32_t getEncodingBits(eEncoding encoding) {
  switch (encoding) {
    case eEncoding::SOP1: return 0xbe800000;
    case eEncoding::SOP2: return 0x80000000;
    case eEncoding::SOPP: return 0xbf800000;
    case eEncoding::SOPC: return 0xbf000000;
    case eEncoding::EXP: return 0xf8000000;
    case eEncoding::VINTRP: return 0xc8000000;
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

static eEncoding getEncoding(uint32_t code) {
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

CodeBlock* Parser::getOrCreateBlock(pc_t pc) {
  auto block = _blocks.get_allocator().new_object<CodeBlock>(pc);

  // Sorted insert
  auto it = std::upper_bound(_blocks.begin(), _blocks.end(), pc, [](pc_t rhs, const std::pair<pc_t, CodeBlock*>& lhs) { return lhs.first > rhs; });

  bool addtask = true;
  if (it != _blocks.begin()) {
    auto prevBlock = std::prev(it)->second;

    if (prevBlock->isParsed && pc <= prevBlock->pc_end) {
      LOG(eLOG_TYPE::DEBUG, " Parse| split block to pc:0x{:x}:0x{:x}  0x{:x}:0x{:x}:", prevBlock->pc_start, pc, pc, prevBlock->pc_end);

      prevBlock->pc_end = pc;
      addtask           = false;
    } else {
    }
  }

  _blocks.insert(it, std::make_pair(pc, block));
  // -

  if (addtask) _tasks.push_back(block);

  return block;
}

void Parser::process() {
  while (!_tasks.empty()) {
    auto& curBlock = *_tasks.back();
    _tasks.pop_back();

    [[unlikely]] if (curBlock.isParsed)
      continue;

    auto pc = curBlock.pc_start;

    { // adjust end
      auto it = std::upper_bound(_blocks.begin(), _blocks.end(), pc, [](pc_t rhs, const std::pair<pc_t, CodeBlock*>& lhs) { return lhs.first > rhs; });
      if (it != _blocks.end()) {
        curBlock.pc_end = it->second->pc_start;
      }
    }

    auto hostMemory = _builder.getHostMapping(pc);
    if (hostMemory == nullptr) throw std::runtime_error("missing mapping");

    LOG(eLOG_TYPE::DEBUG, "Parse| -> pc:0x{:x} module:0x{:x}", pc, hostMemory->pc);

    curBlock.isParsed = true;
    while (pc < curBlock.pc_end) {
      auto handle = [&] {
        auto pCode = (uint32_t const*)(hostMemory->host + pc);
        switch (getEncoding(*pCode)) {
          case eEncoding::SOP1: return handleSop1(curBlock, pc, pCode);
          case eEncoding::SOP2: return handleSop2(curBlock, pc, pCode);
          case eEncoding::SOPP: return handleSopp(curBlock, pc, pCode);
          case eEncoding::SOPC: return handleSopc(curBlock, pc, pCode);
          case eEncoding::EXP: return handleExp(curBlock, pc, pCode);
          case eEncoding::VINTRP: return handleVintrp(curBlock, pc, pCode);
          case eEncoding::DS: return handleDs(curBlock, pc, pCode);
          case eEncoding::MUBUF: return handleMubuf(curBlock, pc, pCode);
          case eEncoding::MTBUF: return handleMtbuf(curBlock, pc, pCode);
          case eEncoding::MIMG: return handleMimg(curBlock, pc, pCode);
          case eEncoding::SMRD: return handleSmrd(curBlock, pc, pCode);
          case eEncoding::SOPK: return handleSopk(curBlock, pc, pCode);
          case eEncoding::VOP1: return handleVop1(curBlock, pc, pCode, false);
          case eEncoding::VOP2: return handleVop2(curBlock, pc, pCode, false);
          case eEncoding::VOP3: {
            auto const header = VOP3(*pCode);
            auto const op     = header.get<VOP3::Field::OP>();
            if (op >= OpcodeOffset_VOP1_VOP3) return handleVop1(curBlock, pc, pCode, true);
            if (op >= 0x140) return handleVop3(curBlock, pc, pCode);
            if (op >= OpcodeOffset_VOP2_VOP3) return handleVop2(curBlock, pc, pCode, true);
            return handleVopc(curBlock, pc, pCode, true);
          }
          case eEncoding::VOPC: return handleVopc(curBlock, pc, pCode, false);
          default: throw std::runtime_error(std::format("wrong encoding 0x{:x}", *pCode));
        }
      };

      pc += handle();
    }
    curBlock.pc_end     = pc;
    hostMemory->size_dw = std::max(hostMemory->size_dw, (uint32_t((pc - hostMemory->pc) / sizeof(uint32_t)))); // update shader size

    // todo handle falltrough (no terminator)

    LOG(eLOG_TYPE::DEBUG, "Parse| <- pc:0x{:x}-0x{:x}  binary size:0x{:x} bytes", curBlock.pc_start, curBlock.pc_end, sizeof(uint32_t) * hostMemory->size_dw);
  }
}

} // namespace compiler::frontend