#pragma once
#include <cassert>
#include <cstddef>
#include <memory_resource>
#include <vector>

namespace compiler::util {

class checkpoint_resource;

template <class T>
struct checkpoint_t {
  std::size_t offset;
  T*          obj;

  checkpoint_t(std::size_t offset, T* obj): offset(offset), obj(obj) {}

  inline ~checkpoint_t() { obj->rollback(*this); }
};

class checkpoint_resource: public std::pmr::memory_resource {
  public:
  checkpoint_resource(void* buffer, std::size_t size): buffer_(static_cast<std::byte*>(buffer)), size_(size), offset_(0) {}

  checkpoint_t<checkpoint_resource> checkpoint() { return checkpoint_t(offset_, this); }

  void rollback(checkpoint_t<checkpoint_resource> const& cp) { offset_ = cp.offset; }

  protected:
  void* do_allocate(std::size_t bytes, std::size_t align) final {
    std::size_t current    = reinterpret_cast<std::size_t>(buffer_) + offset_;
    std::size_t aligned    = (current + (align - 1)) & ~(align - 1);
    std::size_t new_offset = aligned - reinterpret_cast<std::size_t>(buffer_) + bytes;

    if (new_offset > size_) { // overflow
      throw std::bad_alloc();
    }

    offset_ = new_offset;
    return reinterpret_cast<void*>(aligned);
  }

  void do_deallocate(void*, std::size_t, std::size_t) final {
    // no-op (bump allocator)
  }

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept final { return this == &other; }

  private:
  std::byte*  buffer_;
  std::size_t size_;
  std::size_t offset_;
};

} // namespace compiler::util