#ifndef ALWAYS_RESIDENT_CACHE_H
#define ALWAYS_RESIDENT_CACHE_H

#include <cstddef>

namespace NomadPathTracer {

struct AlwaysResidentCache {
  bool dirty = true;
  size_t lastPrimitiveCount = 0;
  size_t lastObjectCount = 0;

  void reset() {
    dirty = true;
    lastPrimitiveCount = 0;
    lastObjectCount = 0;
  }

  void markDirty() { dirty = true; }

  bool needsUpdate(bool forceAllToggles, size_t primitiveCount,
                   size_t objectCount, bool hasRecentChanges) const {
    if (forceAllToggles)
      return true;
    if (dirty)
      return true;
    if (primitiveCount != lastPrimitiveCount ||
        objectCount != lastObjectCount)
      return true;
    if (hasRecentChanges)
      return true;
    return false;
  }

  void markUpdated(size_t primitiveCount, size_t objectCount) {
    dirty = false;
    lastPrimitiveCount = primitiveCount;
    lastObjectCount = objectCount;
  }
};

} // namespace NomadPathTracer

#endif // ALWAYS_RESIDENT_CACHE_H

