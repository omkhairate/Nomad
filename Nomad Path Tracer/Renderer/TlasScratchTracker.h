#pragma once

#include <cstddef>
#include <cstdint>

namespace NomadPathTracer {

class TlasScratchTracker {
public:
  void reset() {
    _currentCapacity = 0;
    _pendingRefitSize = 0;
    _pendingRetiredBytes = 0;
    _pendingEventValue = 0;
    _pendingResize = false;
  }

  void noteAllocation(std::size_t capacity) { _currentCapacity = capacity; }

  void registerRebuild(uint64_t eventValue, std::size_t buildScratchSize,
                       std::size_t refitScratchSize) {
    if (refitScratchSize >= buildScratchSize) {
      _pendingResize = false;
      _pendingEventValue = eventValue;
      _pendingRefitSize = refitScratchSize;
      _pendingRetiredBytes = 0;
      return;
    }

    _pendingResize = true;
    _pendingEventValue = eventValue;
    _pendingRefitSize = refitScratchSize;

    std::size_t basis = _currentCapacity > 0 ? _currentCapacity : buildScratchSize;
    if (basis > refitScratchSize)
      _pendingRetiredBytes = basis - refitScratchSize;
    else
      _pendingRetiredBytes = buildScratchSize - refitScratchSize;
  }

  bool hasPendingResize() const { return _pendingResize; }

  bool ready(uint64_t completedEventValue) const {
    if (!_pendingResize)
      return false;
    if (_pendingEventValue == 0)
      return true;
    return completedEventValue >= _pendingEventValue;
  }

  std::size_t targetRefitSize() const { return _pendingRefitSize; }

  std::size_t retiredBytes() const { return _pendingRetiredBytes; }

  void finalizeResize() {
    _pendingRefitSize = 0;
    _pendingRetiredBytes = 0;
    _pendingEventValue = 0;
    _pendingResize = false;
  }

private:
  std::size_t _currentCapacity = 0;
  std::size_t _pendingRefitSize = 0;
  std::size_t _pendingRetiredBytes = 0;
  uint64_t _pendingEventValue = 0;
  bool _pendingResize = false;
};

} // namespace NomadPathTracer

