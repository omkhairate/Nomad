#pragma once

#include <cstddef>

namespace NomadPathTracer {

class ResidencyBudget {
public:
  void setTlasScratchBytes(std::size_t bytes) { _tlasScratchBytes = bytes; }

  void setBlasScratchBytes(std::size_t availableBytes, std::size_t inUseBytes) {
    _blasScratchAvailableBytes = availableBytes;
    _blasScratchInUseBytes = inUseBytes;
  }

  std::size_t totalScratchBytes() const {
    return _tlasScratchBytes + _blasScratchAvailableBytes + _blasScratchInUseBytes;
  }

  double scratchMemoryMB() const { return bytesToMB(totalScratchBytes()); }

  double residencyMemoryMB(double totalMemoryMB) const {
    double adjusted = totalMemoryMB - scratchMemoryMB();
    return adjusted > 0.0 ? adjusted : 0.0;
  }

private:
  static double bytesToMB(std::size_t bytes) {
    constexpr double kBytesPerMB = 1024.0 * 1024.0;
    return static_cast<double>(bytes) / kBytesPerMB;
  }

  std::size_t _tlasScratchBytes = 0;
  std::size_t _blasScratchAvailableBytes = 0;
  std::size_t _blasScratchInUseBytes = 0;
};

} // namespace NomadPathTracer

