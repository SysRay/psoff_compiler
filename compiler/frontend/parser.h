#pragma once
#include "gfx/register_types.h"
#include "operand_types.h"
#include "types.h"
#include "util/common.h"

#include <limits>
#include <memory_resource>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <stdint.h>
#include <vector>

namespace compiler {
class CompilerCtx;
}

namespace mlir::func {
class FuncOp;
}

namespace compiler::frontend {

struct OperationId;
using OperationId_t = id_t<OperationId, uint32_t>;

struct CodeBlock {
  CLASS_NO_COPY(CodeBlock);

  pc_t         pc_start, pc_end;
  mlir::Block* mlirBlock;

  bool isParsed = false;

  CodeBlock(pc_t start, std::pmr::memory_resource* resource): pc_start(start), pc_end(start + std::numeric_limits<uint32_t>::max()) {}
};

class Parser {
  uint8_t handleSop1(CodeBlock& cb, pc_t pc, uint32_t const* pCode);
  uint8_t handleSop2(CodeBlock& cb, pc_t pc, uint32_t const* pCode);
  uint8_t handleSopc(CodeBlock& cb, pc_t pc, uint32_t const* pCode);
  uint8_t handleSopk(CodeBlock& cb, pc_t pc, uint32_t const* pCode);
  uint8_t handleSopp(CodeBlock& cb, pc_t pc, uint32_t const* pCode);
  uint8_t handleSmrd(CodeBlock& cb, pc_t pc, uint32_t const* pCode);
  uint8_t handleVop1(CodeBlock& cb, pc_t pc, uint32_t const* pCode, bool extended);
  uint8_t handleVop2(CodeBlock& cb, pc_t pc, uint32_t const* pCode, bool extended);
  uint8_t handleVop3(CodeBlock& cb, pc_t pc, uint32_t const* pCode);
  uint8_t handleVopc(CodeBlock& cb, pc_t pc, uint32_t const* pCode, bool extended);
  uint8_t handleVintrp(CodeBlock& cb, pc_t pc, uint32_t const* pCode);
  uint8_t handleExp(CodeBlock& cb, pc_t pc, uint32_t const* pCode);
  uint8_t handleMubuf(CodeBlock& cb, pc_t pc, uint32_t const* pCode);
  uint8_t handleMtbuf(CodeBlock& cb, pc_t pc, uint32_t const* pCode);
  uint8_t handleMimg(CodeBlock& cb, pc_t pc, uint32_t const* pCode);
  uint8_t handleDs(CodeBlock& cb, pc_t pc, uint32_t const* pCode);

  mlir::Value loadRegister(eOperandKind op, mlir::Type type);
  void        storeRegister(eOperandKind dst, mlir::Value value);

  OperandTypeCache const& types() const;

  public:
  Parser(CompilerCtx& builder, std::pmr::memory_resource* resource);

  ~Parser();

  void process();

  CodeBlock* getOrCreateBlock(pc_t pc, mlir::Region* region);

  private:
  std::pmr::vector<std::pair<pc_t, CodeBlock*>> _blocks;
  std::pmr::vector<CodeBlock*>                  _tasks;

  CompilerCtx& _compilerCtx;

  mlir::OpBuilder _mlirBuilder;
  mlir::Location  _defaultLocation;

  uint32_t const* _curCode = nullptr;
};

} // namespace compiler::frontend