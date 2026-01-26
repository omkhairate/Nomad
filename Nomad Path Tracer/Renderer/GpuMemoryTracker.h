#pragma once

#include <array>
#include <cstddef>
#include <mutex>
#include <unordered_map>

namespace NomadPathTracer {

class GpuMemoryTracker {
public:
  enum class Category {
    Scratch,
    Geometry,
    Textures,
    RendererBuffers,
    HeapsAS,
    Staging,
    Count
  };

  struct Snapshot {
    std::array<size_t, static_cast<size_t>(Category::Count)> bytes{};
    size_t totalTracked = 0;
  };

  void trackResource(const void *resource, size_t bytes, Category category);
  void releaseResource(const void *resource);
  Snapshot snapshot() const;
  size_t bytesForCategory(Category category) const;

private:
  struct Entry {
    size_t bytes = 0;
    Category category = Category::Scratch;
  };

  std::unordered_map<const void *, Entry> _entries;
  std::array<size_t, static_cast<size_t>(Category::Count)> _totals{};
  mutable std::mutex _mutex;
};

} // namespace NomadPathTracer
