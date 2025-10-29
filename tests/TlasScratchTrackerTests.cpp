#include "MetalCpp Path Tracer/Renderer/TlasScratchTracker.h"

#include <cassert>

using MetalCppPathTracer::TlasScratchTracker;

int main() {
  TlasScratchTracker tracker;

  tracker.noteAllocation(8192);
  tracker.registerRebuild(5, 8192, 2048);
  assert(tracker.hasPendingResize());
  assert(!tracker.ready(4));
  assert(tracker.ready(5));
  assert(tracker.targetRefitSize() == 2048);
  assert(tracker.retiredBytes() == 8192 - 2048);

  tracker.noteAllocation(2048);
  tracker.finalizeResize();
  assert(!tracker.hasPendingResize());

  tracker.registerRebuild(0, 2048, 2048);
  assert(!tracker.hasPendingResize());

  tracker.noteAllocation(0);
  tracker.registerRebuild(7, 0, 0);
  assert(!tracker.hasPendingResize());
  assert(!tracker.ready(7));

  return 0;
}
