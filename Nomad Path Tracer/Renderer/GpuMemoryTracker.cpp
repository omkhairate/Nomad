#include "GpuMemoryTracker.h"

namespace NomadPathTracer {

void GpuMemoryTracker::trackResource(const void *resource, size_t bytes,
                                     Category category) {
  if (!resource)
    return;

  std::lock_guard<std::mutex> lock(_mutex);
  auto it = _entries.find(resource);
  if (it != _entries.end()) {
    Entry &entry = it->second;
    size_t oldIndex = static_cast<size_t>(entry.category);
    if (_totals[oldIndex] >= entry.bytes)
      _totals[oldIndex] -= entry.bytes;
    else
      _totals[oldIndex] = 0;
    if (bytes == 0) {
      _entries.erase(it);
      return;
    }
    entry.bytes = bytes;
    entry.category = category;
  } else {
    if (bytes == 0)
      return;
    _entries.emplace(resource, Entry{bytes, category});
  }

  size_t newIndex = static_cast<size_t>(category);
  _totals[newIndex] += bytes;
}

void GpuMemoryTracker::releaseResource(const void *resource) {
  if (!resource)
    return;

  std::lock_guard<std::mutex> lock(_mutex);
  auto it = _entries.find(resource);
  if (it == _entries.end())
    return;

  size_t index = static_cast<size_t>(it->second.category);
  if (_totals[index] >= it->second.bytes)
    _totals[index] -= it->second.bytes;
  else
    _totals[index] = 0;
  _entries.erase(it);
}

GpuMemoryTracker::Snapshot GpuMemoryTracker::snapshot() const {
  std::lock_guard<std::mutex> lock(_mutex);
  Snapshot snap;
  snap.bytes = _totals;
  snap.totalTracked = 0;
  for (size_t bytes : _totals)
    snap.totalTracked += bytes;
  return snap;
}

size_t GpuMemoryTracker::bytesForCategory(Category category) const {
  std::lock_guard<std::mutex> lock(_mutex);
  return _totals[static_cast<size_t>(category)];
}

} // namespace NomadPathTracer
