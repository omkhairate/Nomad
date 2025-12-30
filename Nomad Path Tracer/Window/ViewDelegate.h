#ifndef VIEW_DELEGATE_H
#define VIEW_DELEGATE_H

#include <Metal/Metal.hpp>
#include <MetalKit/MetalKit.hpp>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <string>
#include <simd/simd.h>

#include "Renderer.h"

namespace NomadPathTracer {

class ViewDelegate : public MTK::ViewDelegate {
public:
  ViewDelegate(MTL::Device *pDevice);
  virtual ~ViewDelegate() override;
  virtual void drawInMTKView(MTK::View *pView) override;
  virtual void drawableSizeWillChange(MTK::View *pView, CGSize size) override;
  void setMaxRayDepth(uint32_t depth);
  uint32_t maxRayDepth() const;
  void setCameraPosition(simd::float3 position);
  simd::float3 activeCameraPosition() const;
  bool hasCameraKeyframes() const;

private:
  Renderer *_pRenderer;
  std::size_t _frameCount = 0;
  std::size_t _maxFrames = 0;
  std::chrono::steady_clock::time_point _lastTime;
  std::string _dumpPath;
  std::ofstream _gpuMemLog;
  std::ofstream _perfLog;
};

}; // namespace NomadPathTracer

#endif //  VIEW_DELEGATE_H
