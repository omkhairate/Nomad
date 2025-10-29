#include "MetalCpp Path Tracer/Renderer/ResidencyBudget.h"

#include <cassert>
#include <cmath>
#include <cstddef>

using MetalCppPathTracer::ResidencyBudget;

int main() {
  ResidencyBudget budget;

  assert(budget.totalScratchBytes() == 0);
  assert(budget.scratchMemoryMB() == 0.0);
  assert(budget.residencyMemoryMB(5.0) == 5.0);

  const std::size_t tlasBytes = 2 * 1024 * 1024;
  const std::size_t blasAvailable = 1 * 1024 * 1024;
  const std::size_t blasInUse = 512 * 1024;

  budget.setTlasScratchBytes(tlasBytes);
  budget.setBlasScratchBytes(blasAvailable, blasInUse);

  double scratchMB = budget.scratchMemoryMB();
  assert(std::abs(scratchMB - 3.5) < 1e-6);

  double adjusted = budget.residencyMemoryMB(10.0);
  assert(std::abs(adjusted - 6.5) < 1e-6);

  double clamped = budget.residencyMemoryMB(2.0);
  assert(clamped == 0.0);

  budget.setBlasScratchBytes(0, 0);
  budget.setTlasScratchBytes(0);
  assert(budget.totalScratchBytes() == 0);
  assert(budget.residencyMemoryMB(1.0) == 1.0);

  return 0;
}

