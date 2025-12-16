#include "Nomad Path Tracer/Renderer/AlwaysResidentCache.h"

#include <cassert>

using NomadPathTracer::AlwaysResidentCache;

namespace {

enum class FakeResidencyStrategy { AlwaysResident, Streaming };

struct FakeSceneObject {};

struct FakeBlasInstanceRecord {
  int primitiveCount = 1;
};

struct FakeResidentResources {
  enum class State { Cold, Streaming, Resident };

  State state = State::Resident;
  bool geometryValid = true;
  bool transitionedToCold = false;

  void transitionToStreaming() { state = State::Streaming; }

  void transitionToCold(FakeBlasInstanceRecord &record) {
    state = State::Cold;
    geometryValid = false;
    transitionedToCold = true;
    record.primitiveCount = 0;
  }
};

struct FakeRenderer {
  FakeResidencyStrategy strategy = FakeResidencyStrategy::Streaming;
  bool buildResult = true;
  bool transitionToColdCalled = false;

  void cancelPendingResidentEviction(size_t, FakeResidentResources &) {}

  bool buildObjectBlas(size_t, const FakeSceneObject &,
                       FakeResidentResources &) {
    return buildResult;
  }

  void transitionResidentToCold(size_t, FakeResidentResources &resources,
                                FakeBlasInstanceRecord &record) {
    transitionToColdCalled = true;
    resources.transitionToCold(record);
  }

  bool isAlwaysResidentStrategy() const {
    return strategy == FakeResidencyStrategy::AlwaysResident;
  }
};

bool ensureResident(FakeRenderer &renderer, size_t objectIndex,
                    const FakeSceneObject &object,
                    FakeResidentResources &resources,
                    FakeBlasInstanceRecord &instanceRecord,
                    bool forceRebuild) {
  renderer.cancelPendingResidentEviction(objectIndex, resources);

  auto previousState = resources.state;
  auto previousGeometryValid = resources.geometryValid;

  if (resources.state == FakeResidentResources::State::Resident && !forceRebuild)
    return true;

  resources.transitionToStreaming();
  resources.geometryValid = false;

  if (!renderer.buildObjectBlas(objectIndex, object, resources)) {
    if (renderer.isAlwaysResidentStrategy()) {
      resources.state = previousState;
      resources.geometryValid = previousGeometryValid;
      return false;
    }
    renderer.transitionResidentToCold(objectIndex, resources, instanceRecord);
    return false;
  }

  resources.state = FakeResidentResources::State::Resident;
  resources.geometryValid = true;
  return true;
}

} // namespace

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
  cache.markUpdated(6, 2);

  {
    FakeRenderer renderer;
    renderer.strategy = FakeResidencyStrategy::AlwaysResident;
    renderer.buildResult = false;
    FakeResidentResources resources;
    FakeSceneObject object;
    FakeBlasInstanceRecord record;

    bool result = ensureResident(renderer, 0, object, resources, record, true);
    assert(!result);
    assert(resources.state == FakeResidentResources::State::Resident);
    assert(resources.geometryValid);
    assert(!resources.transitionedToCold);
    assert(!renderer.transitionToColdCalled);
    assert(record.primitiveCount == 1);
  }

  {
    FakeRenderer renderer;
    renderer.strategy = FakeResidencyStrategy::Streaming;
    renderer.buildResult = false;
    FakeResidentResources resources;
    FakeSceneObject object;
    FakeBlasInstanceRecord record;

    bool result = ensureResident(renderer, 0, object, resources, record, true);
    assert(!result);
    assert(resources.state == FakeResidentResources::State::Cold);
    assert(!resources.geometryValid);
    assert(resources.transitionedToCold);
    assert(renderer.transitionToColdCalled);
    assert(record.primitiveCount == 0);
  }

  return 0;
}
