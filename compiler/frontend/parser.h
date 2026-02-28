#pragma once
#include "../util/flags.h"
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

enum OpFlags : uint8_t {
  eNegate       = (1 << 0), ///< SET MSB
  eAbsolute     = (1 << 1),
  eNot          = (1 << 2), ///< Invert
  eClampToOne   = (1 << 3), ///< +-1.0
  eClampFMinMax = (1 << 4), ///< float min max
  eClampFZero   = (1 << 5), ///< +zero
};

using OperandType_t = mlir::Type;
using IRResult      = mlir::Value;

struct OpSrc {
  PACK(struct {
    eOperandKind         kind;
    util::Flags<OpFlags> flags;

    union {
      uint32_t uimm;
      float    fimm;
    };
  });

  inline void setNot() { flags ^= OpFlags::eNot; }

  constexpr explicit OpSrc() {}

  constexpr explicit OpSrc(eOperandKind kind, util::Flags<OpFlags> flags = {}): kind(kind), flags(flags) {}

  constexpr explicit OpSrc(eOperandKind kind, bool negate, bool absolute): kind(kind) {
    if (negate) flags |= OpFlags::eNegate;
    if (absolute) flags |= OpFlags::eAbsolute;
  }

  constexpr explicit OpSrc(uint32_t imm, util::Flags<OpFlags> flags = {}): kind(eOperandKind::Unset()), flags(flags), uimm(imm) {}

  constexpr explicit OpSrc(int32_t imm, util::Flags<OpFlags> flags = {}): kind(eOperandKind::Unset()), flags(flags), uimm(std::bit_cast<uint32_t>(imm)) {}

  constexpr explicit OpSrc(float imm, util::Flags<OpFlags> flags = {}): kind(eOperandKind::Unset()), flags(flags), fimm(imm) {}

  constexpr OpSrc& operator=(OpSrc const& other) = default;
};

static_assert(sizeof(OpSrc) <= sizeof(uint64_t));

struct OpDst {
  PACK(struct {
    eOperandKind         kind;
    util::Flags<OpFlags> flags;
    uint8_t              omod = 0;
  });

  constexpr explicit OpDst() {}

  constexpr explicit OpDst(eOperandKind kind, util::Flags<OpFlags> flags = {}): kind(kind) {}

  constexpr explicit OpDst(eOperandKind kind, uint8_t omod, bool clamp): kind(kind), omod(omod) {
    if (clamp) flags |= OpFlags::eClampToOne;
  }

  constexpr OpDst& operator=(OpDst const& other) = default;
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

  template <typename Op, typename... Args>
  inline auto parse(Args&&... args) {
    return Op::create(this, std::forward<Args>(args)...);
  }

  public:
  Parser(CompilerCtx& builder, std::pmr::memory_resource* resource);

  ~Parser();

  void process();

  CodeBlock* getOrCreateBlock(pc_t pc, mlir::Region* region);

  OperandTypeCache const& types() const;
  mlir::Value             loadRegister(OpSrc op, mlir::Type type);
  mlir::Value             storeRegister(OpDst dst, mlir::Value value);

  template <typename Op, typename... Args>
  inline auto create(Args&&... args) {
    return _mlirBuilder.create<Op>(_defaultLocation, std::forward<Args>(args)...);
  }

  inline auto& getLoc() { return _defaultLocation; }

  inline auto& getBuilder() { return _mlirBuilder; }

  private:
  std::pmr::vector<std::pair<pc_t, CodeBlock*>> _blocks;
  std::pmr::vector<CodeBlock*>                  _tasks;

  CompilerCtx& _compilerCtx;

  mlir::OpBuilder _mlirBuilder;
  mlir::Location  _defaultLocation;

  uint32_t const* _curCode = nullptr;
};

} // namespace compiler::frontend