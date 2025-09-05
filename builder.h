#pragma once

#include "ir/ir.h"

#include <memory_resource>

namespace compiler {
constexpr auto operator""_MB(size_t x)
    -> size_t {
  return 1024L * 1024L * x;
}

constexpr size_t MEMORY_SIZE = 1_MB;

class Builder {
  public:
  Builder(size_t numInstructions = 0): _buffer(std::make_unique_for_overwrite<uint8_t[]>(MEMORY_SIZE)) {
    if(numInstructions != 0) {
      numInstructions = 2048;
    }
    _instructions.reserve(numInstructions);
  }

  auto& createInstruction(ir::InstCore const& instr) {
    return _instructions.emplace_back(instr);
  }

  private:
  std::unique_ptr<uint8_t[]>          _buffer;
  std::pmr::monotonic_buffer_resource _pool {_buffer.get(), MEMORY_SIZE};

  std::pmr::vector<ir::InstCore> _instructions {&_pool};
};

} // namespace compiler