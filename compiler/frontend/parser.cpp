#include "parser.h"

#include "compiler_ctx.h"
#include "encoding/opcodes_table.h"
#include "gfx/encoding_types.h"
#include "logging.h"

#include <limits>

// mlir
#include "mlir/custom.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

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

Parser::Parser(CompilerCtx& ctx, std::pmr::memory_resource* resource)
    : _compilerCtx(ctx),
      _blocks(resource),
      _tasks(resource),
      _mlirBuilder(_compilerCtx.getContext()),
      _defaultLocation(mlir::UnknownLoc::get(_compilerCtx.getContext())) {
  ;
}

Parser::~Parser() {
  auto allocator = _blocks.get_allocator();
  for (auto const& item: _blocks) {
    allocator.delete_object<CodeBlock>(item.second);
  }
}

auto upperBoundOpByPC(mlir::Block* block, uint64_t targetPC) {
  auto& ops     = block->getOperations();
  auto  beginIt = ops.begin();
  auto  endIt   = ops.end();

  auto candidate = beginIt;

  while (beginIt != endIt) {
    auto midIt = beginIt;
    std::advance(midIt, std::distance(beginIt, endIt) / 2);
    mlir::Operation& op = *midIt;

    uint64_t opPC = 0;
    if (auto loc = llvm::dyn_cast<mlir::OpaqueLoc>(op.getLoc())) {
      opPC = reinterpret_cast<uint64_t>(loc.getUnderlyingLocation());
    } else {
      // If no location, scan forward for next valid operation
      auto scanIt = std::next(midIt);
      while (scanIt != endIt) {
        if (auto loc = llvm::dyn_cast<mlir::OpaqueLoc>((*scanIt).getLoc())) {
          opPC = reinterpret_cast<uint64_t>(loc.getAsOpaquePointer());
          break;
        }
        ++scanIt;
      }

      if (opPC == 0) opPC = UINT64_MAX; // If still zero, treat as infinity to move left
    }

    if (opPC <= targetPC) {
      // Go right
      beginIt = std::next(midIt);
    } else {
      // Go left
      candidate = midIt;
      endIt     = midIt;
    }
  }

  return candidate; // first op with pc > targetPC, or nullptr if none
}

CodeBlock* Parser::getOrCreateBlock(pc_t pc, mlir::Region* region) {
  // Sorted insert
  auto it = std::upper_bound(_blocks.begin(), _blocks.end(), pc, [](pc_t rhs, const std::pair<pc_t, CodeBlock*>& lhs) { return lhs.first > rhs; });

  if (it != _blocks.begin()) {
    auto prev = std::prev(it);
    if (prev->first == pc) return prev->second; // Return matchin block

    // Check if block needs splitting
    auto prevBlock = prev->second;
    if (prevBlock->isParsed && pc <= prevBlock->pc_end) {
      LOG(eLOG_TYPE::DEBUG, " Parse| split block to pc:0x{:x}:0x{:x}  0x{:x}:0x{:x}:", prevBlock->pc_start, pc, pc, prevBlock->pc_end);

      auto block = prevBlock->mlirBlock;

      auto itOp = upperBoundOpByPC(block, pc);

      auto pNewBlock = _blocks.get_allocator().new_object<CodeBlock>(pc, _blocks.get_allocator().resource());
      _blocks.insert(it, std::make_pair(pc, pNewBlock));

      mlir::OpBuilder::InsertionGuard guard(_mlirBuilder);

      pNewBlock->mlirBlock = block->splitBlock(itOp);
      pNewBlock->pc_end    = prevBlock->pc_end;

      _mlirBuilder.setInsertionPointToEnd(block);
      _mlirBuilder.create<mlir::cf::BranchOp>(_defaultLocation, pNewBlock->mlirBlock);
      prevBlock->pc_end = pc;
      return pNewBlock;
    }
  }

  auto pBlock = _blocks.get_allocator().new_object<CodeBlock>(pc, _blocks.get_allocator().resource());
  _blocks.insert(it, std::make_pair(pc, pBlock));

  mlir::OpBuilder::InsertionGuard guard(_mlirBuilder);
  pBlock->mlirBlock = _mlirBuilder.createBlock(region);

  _tasks.push_back(pBlock);
  return pBlock;
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

    auto hostMemory = _compilerCtx.getHostMapping(pc);
    if (hostMemory == nullptr) throw std::runtime_error("missing mapping");

    LOG(eLOG_TYPE::DEBUG, "Parse| -> pc:0x{:x} module:0x{:x}", pc, hostMemory->pc);

    _mlirBuilder.setInsertionPointToEnd(curBlock.mlirBlock);
    curBlock.isParsed = true;

    size_t curOperationIndex = 0;
    while (pc < curBlock.pc_end) {
      auto handle = [&] {
        _curCode = (uint32_t const*)(hostMemory->host + pc);
        switch (getEncoding(*_curCode)) {
          case eEncoding::SOP1: return handleSop1(curBlock, pc, _curCode);
          case eEncoding::SOP2: return handleSop2(curBlock, pc, _curCode);
          case eEncoding::SOPP: return handleSopp(curBlock, pc, _curCode);
          case eEncoding::SOPC: return handleSopc(curBlock, pc, _curCode);
          case eEncoding::EXP: return handleExp(curBlock, pc, _curCode);
          case eEncoding::VINTRP: return handleVintrp(curBlock, pc, _curCode);
          case eEncoding::DS: return handleDs(curBlock, pc, _curCode);
          case eEncoding::MUBUF: return handleMubuf(curBlock, pc, _curCode);
          case eEncoding::MTBUF: return handleMtbuf(curBlock, pc, _curCode);
          case eEncoding::MIMG: return handleMimg(curBlock, pc, _curCode);
          case eEncoding::SMRD: return handleSmrd(curBlock, pc, _curCode);
          case eEncoding::SOPK: return handleSopk(curBlock, pc, _curCode);
          case eEncoding::VOP1: return handleVop1(curBlock, pc, _curCode, false);
          case eEncoding::VOP2: return handleVop2(curBlock, pc, _curCode, false);
          case eEncoding::VOP3: {
            auto const header = VOP3(*_curCode);
            auto const op     = header.get<VOP3::Field::OP>();
            if (op >= OpcodeOffset_VOP1_VOP3) return handleVop1(curBlock, pc, _curCode, true);
            if (op >= 0x140) return handleVop3(curBlock, pc, _curCode);
            if (op >= OpcodeOffset_VOP2_VOP3) return handleVop2(curBlock, pc, _curCode, true);
            return handleVopc(curBlock, pc, _curCode, true);
          }
          case eEncoding::VOPC: return handleVopc(curBlock, pc, _curCode, false);
          default: throw std::runtime_error(std::format("wrong encoding 0x{:x}", *_curCode));
        }
      };

      pc += handle();

      // handle pc mapping
      auto& ops = curBlock.mlirBlock->getOperations();
      if (curOperationIndex != ops.size()) {
        curOperationIndex = ops.size();

        ops.back().setLoc(mlir::OpaqueLoc::get(pc, _compilerCtx.getContext()));
      }
      // -
    }
    curBlock.pc_end     = pc;
    hostMemory->size_dw = std::max(hostMemory->size_dw, (uint32_t((pc - hostMemory->pc) / sizeof(uint32_t)))); // update shader size

    if (curBlock.mlirBlock->empty() || !curBlock.mlirBlock->getTerminator()) {
      // Note: handle Falltrough
      auto target = getOrCreateBlock(pc, curBlock.mlirBlock->getParent());
      _mlirBuilder.create<mlir::cf::BranchOp>(_defaultLocation, target->mlirBlock);
    }

    LOG(eLOG_TYPE::DEBUG, "Parse| <- pc:0x{:x}-0x{:x}  binary size:0x{:x} bytes", curBlock.pc_start, curBlock.pc_end, sizeof(uint32_t) * hostMemory->size_dw);
  }
}

OperandTypeCache const& Parser::types() const {
  return _compilerCtx.types();
}

mlir::Value Parser::loadRegister(eOperandKind src, mlir::Type type) {
  if (src.isLiteral()) {
    if (type.getIntOrFloatBitWidth() == 64 && type.isFloat()) {
      // Note Literal double constants are placed in high 32-bits of double
      return _mlirBuilder.create<mlir::arith::ConstantFloatOp>(_defaultLocation, (mlir::FloatType)type,
                                                               llvm::APFloat(std::bit_cast<double>((uint64_t)(*(_curCode + 1)) << 32u)));
    } else {
      // Note: signed ops need to sign extend. make it i32 and extend later
      return _mlirBuilder.create<mlir::arith::ConstantIntOp>(_defaultLocation, types().i32(), *(_curCode + 1));
    }
  } else if (src.isConstI()) {
    return _mlirBuilder.create<mlir::arith::ConstantIntOp>(_defaultLocation, type, src.getConstI());
  } else if (src.isConstF()) {
    return _mlirBuilder.create<mlir::arith::ConstantFloatOp>(_defaultLocation, (mlir::FloatType)type, llvm::APFloat(src.getConstF()));
  } else {
    return _mlirBuilder.create<mlir::psoff::LoadOp>(_defaultLocation, type, _mlirBuilder.getIndexAttr((uint32_t)src.base()));
  }
}

void Parser::storeRegister(eOperandKind dst, mlir::Value value) {
  _mlirBuilder.create<mlir::psoff::StoreOp>(_defaultLocation, _mlirBuilder.getIndexAttr((uint32_t)dst.base()), value);
}
} // namespace compiler::frontend