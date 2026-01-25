#pragma once

#include <Metal/Metal.hpp>
#include <array>
#include <cstddef>
#include <unordered_map>

namespace NomadPathTracer {

class GpuMemoryTracker {
public:
  enum class Category {
    Scratch = 0,
    Geometry = 1,
    Textures = 2,
    Restir = 3,
    RendererBuffers = 4,
    Heaps = 5,
    Staging = 6,
    Other = 7,
    Count = 8
  };

  struct Totals {
    size_t scratch = 0;
    size_t geometry = 0;
    size_t textures = 0;
    size_t restir = 0;
    size_t rendererBuffers = 0;
    size_t heaps = 0;
    size_t staging = 0;
    size_t other = 0;
    size_t totalTracked = 0;
  };

  void recordResource(Category category, const MTL::Resource *resource);
  void recordAllocation(Category category, const void *key, size_t bytes);
  void releaseResource(const MTL::Resource *resource);
  void releaseAllocation(const void *key);

  size_t bytes(Category category) const;
  size_t totalTrackedBytes() const;
  Totals totals() const;

private:
  struct AllocationInfo {
    Category category = Category::Other;
    size_t bytes = 0;
  };

  std::unordered_map<const void *, AllocationInfo> _allocations;
  std::array<size_t, static_cast<size_t>(Category::Count)> _bytes{};
};

} // namespace NomadPathTracer
