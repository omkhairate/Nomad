#ifndef METALCPP_PATH_TRACER_INCREMENTAL_UPDATE_UTILS_H
#define METALCPP_PATH_TRACER_INCREMENTAL_UPDATE_UTILS_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace MetalCppPathTracer {

struct RangeUpdate {
  size_t offset = 0; // Byte offset within the destination buffer.
  size_t length = 0; // Number of bytes covered by the update.
};

inline std::vector<RangeUpdate>
computeChangedRanges(const void *previous, const void *current,
                     size_t elementSize, size_t elementCount) {
  std::vector<RangeUpdate> ranges;
  if (!previous || !current || elementSize == 0 || elementCount == 0)
    return ranges;

  const auto *prevBytes = static_cast<const uint8_t *>(previous);
  const auto *currBytes = static_cast<const uint8_t *>(current);

  size_t activeStart = static_cast<size_t>(-1);
  for (size_t index = 0; index < elementCount; ++index) {
    const uint8_t *prevPtr = prevBytes + index * elementSize;
    const uint8_t *currPtr = currBytes + index * elementSize;
    bool changed = std::memcmp(prevPtr, currPtr, elementSize) != 0;
    if (changed) {
      if (activeStart == static_cast<size_t>(-1))
        activeStart = index;
    } else if (activeStart != static_cast<size_t>(-1)) {
      size_t offset = activeStart * elementSize;
      size_t length = (index - activeStart) * elementSize;
      if (length > 0)
        ranges.push_back({offset, length});
      activeStart = static_cast<size_t>(-1);
    }
  }

  if (activeStart != static_cast<size_t>(-1)) {
    size_t offset = activeStart * elementSize;
    size_t length = (elementCount - activeStart) * elementSize;
    if (length > 0)
      ranges.push_back({offset, length});
  }

  return ranges;
}

} // namespace MetalCppPathTracer

#endif // METALCPP_PATH_TRACER_INCREMENTAL_UPDATE_UTILS_H

