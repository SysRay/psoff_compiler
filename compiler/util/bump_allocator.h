
#pragma once

#include "llvm/Support/PerThreadBumpPtrAllocator.h"

#include <memory_resource>

namespace compiler::util {
class BumpAllocator: public std::pmr::memory_resource {
  public:
  explicit BumpAllocator() {}

  protected:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override { return allocator.Allocate(bytes, alignment); }

  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {}

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    const BumpAllocator* otherAlloc = dynamic_cast<const BumpAllocator*>(&other);
    return otherAlloc && &otherAlloc->allocator == &allocator;
  }

  private:
  llvm::BumpPtrAllocator allocator;
};
} // namespace compiler::util