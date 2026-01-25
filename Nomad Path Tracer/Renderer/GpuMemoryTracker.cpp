#include "GpuMemoryTracker.h"

namespace NomadPathTracer {

namespace {
size_t categoryIndex(GpuMemoryTracker::Category category) {
  return static_cast<size_t>(category);
}
} // namespace

void GpuMemoryTracker::recordResource(Category category,
                                      const MTL::Resource *resource) {
  if (!resource)
    return;
  size_t bytes = static_cast<size_t>(resource->allocatedSize());
  recordAllocation(category, resource, bytes);
}

void GpuMemoryTracker::recordAllocation(Category category, const void *key,
                                        size_t bytes) {
  if (!key)
    return;

  if (bytes == 0) {
    releaseAllocation(key);
    return;
  }

  auto it = _allocations.find(key);
  if (it != _allocations.end()) {
    size_t index = categoryIndex(it->second.category);
    if (_bytes[index] >= it->second.bytes)
      _bytes[index] -= it->second.bytes;
    else
      _bytes[index] = 0;
  }

  _allocations[key] = {category, bytes};
  _bytes[categoryIndex(category)] += bytes;
}

void GpuMemoryTracker::releaseResource(const MTL::Resource *resource) {
  releaseAllocation(resource);
}

void GpuMemoryTracker::releaseAllocation(const void *key) {
  if (!key)
    return;

  auto it = _allocations.find(key);
  if (it == _allocations.end())
    return;

  size_t index = categoryIndex(it->second.category);
  if (_bytes[index] >= it->second.bytes)
    _bytes[index] -= it->second.bytes;
  else
    _bytes[index] = 0;

  _allocations.erase(it);
}

size_t GpuMemoryTracker::bytes(Category category) const {
  return _bytes[categoryIndex(category)];
}

size_t GpuMemoryTracker::totalTrackedBytes() const {
  size_t total = 0;
  for (size_t bytes : _bytes)
    total += bytes;
  return total;
}

GpuMemoryTracker::Totals GpuMemoryTracker::totals() const {
  Totals totals{};
  totals.scratch = bytes(Category::Scratch);
  totals.geometry = bytes(Category::Geometry);
  totals.textures = bytes(Category::Textures);
  totals.restir = bytes(Category::Restir);
  totals.rendererBuffers = bytes(Category::RendererBuffers);
  totals.heaps = bytes(Category::Heaps);
  totals.staging = bytes(Category::Staging);
  totals.other = bytes(Category::Other);
  totals.totalTracked = totalTrackedBytes();
  return totals;
}

} // namespace NomadPathTracer
