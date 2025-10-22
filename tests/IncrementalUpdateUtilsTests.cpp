#include "MetalCpp Path Tracer/Renderer/IncrementalUpdateUtils.h"

#include <cassert>
#include <vector>

using MetalCppPathTracer::computeChangedRanges;

int main() {
  std::vector<int> baseline{1, 2, 3, 4, 5};
  std::vector<int> candidate = baseline;

  auto ranges =
      computeChangedRanges(baseline.data(), candidate.data(), sizeof(int),
                           baseline.size());
  assert(ranges.empty());

  candidate = baseline;
  candidate[1] = 42;
  candidate[2] = 84;
  ranges = computeChangedRanges(baseline.data(), candidate.data(), sizeof(int),
                                baseline.size());
  assert(ranges.size() == 1);
  assert(ranges[0].offset == sizeof(int) * 1);
  assert(ranges[0].length == sizeof(int) * 2);

  candidate = baseline;
  candidate[0] = -7;
  candidate[3] = 19;
  ranges = computeChangedRanges(baseline.data(), candidate.data(), sizeof(int),
                                baseline.size());
  assert(ranges.size() == 2);
  assert(ranges[0].offset == 0);
  assert(ranges[0].length == sizeof(int));
  assert(ranges[1].offset == sizeof(int) * 3);
  assert(ranges[1].length == sizeof(int));

  candidate = baseline;
  candidate.back() = 99;
  ranges = computeChangedRanges(baseline.data(), candidate.data(), sizeof(int),
                                baseline.size());
  assert(ranges.size() == 1);
  assert(ranges[0].offset == sizeof(int) * (baseline.size() - 1));
  assert(ranges[0].length == sizeof(int));

  ranges =
      computeChangedRanges(baseline.data(), candidate.data(), 0, baseline.size());
  assert(ranges.empty());

  return 0;
}

