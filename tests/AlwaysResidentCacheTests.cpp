#include "MetalCpp Path Tracer/Renderer/AlwaysResidentCache.h"

#include <cassert>

using MetalCppPathTracer::AlwaysResidentCache;

int main() {
  AlwaysResidentCache cache;

  assert(cache.needsUpdate(false, 0, 0, false));

  cache.markUpdated(5, 3);
  assert(!cache.needsUpdate(false, 5, 3, false));

  assert(cache.needsUpdate(false, 6, 3, false));
  cache.markUpdated(6, 3);

  assert(cache.needsUpdate(false, 6, 2, false));
  cache.markUpdated(6, 2);

  cache.markDirty();
  assert(cache.needsUpdate(false, 6, 2, false));
  cache.markUpdated(6, 2);

  assert(cache.needsUpdate(false, 6, 2, true));
  cache.markUpdated(6, 2);

  assert(cache.needsUpdate(true, 6, 2, false));

  return 0;
}

