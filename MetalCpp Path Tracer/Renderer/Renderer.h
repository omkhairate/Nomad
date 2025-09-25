#ifndef RENDERER_H
#define RENDERER_H

#include <Metal/Metal.hpp>
#include <MetalKit/MetalKit.hpp>
#include <cstdint>
#include <chrono>
#include <simd/simd.h>
#include <vector>

#include "Scene.h"

namespace MetalCppPathTracer {

class Renderer {
public:
  Renderer(MTL::Device *pDevice);
  ~Renderer();

  void updateVisibleScene();
  void buildShaders();
  void buildBuffers();
  void buildTextures();

  void recalculateViewport();
  bool updateCamera();

  void updateUniforms();
  void draw(MTK::View *pView);
  void drawableSizeWillChange(MTK::View *pView, CGSize size);

  bool hasKeyframes() const;
  // Toggle visibility of an individual primitive without rebuilding TLAS.
  void setPrimitiveActive(size_t index, bool active);

  // Dump acceleration structure to a JSON file for debugging.
  void dumpAccelerationStructure(const std::string &path);

  // Return the amount of GPU memory currently allocated by the device in MB.
  double currentGPUMemoryMB() const;

  // Expose last recorded performance metrics.
  double lastCPUTime() const;
  double lastGPUTime() const;
  double lastRaysPerSecond() const;
  size_t activeNodeCount() const;
  size_t totalNodeCount() const;

  std::vector<std::pair<simd::float3, float>> _allSpheres;

  struct Chunk {
    std::vector<std::pair<simd::float4, simd::float4>>
        spheres; // (transform, material)
    simd::int3 chunkCoords;
  };

private:
  MTL::Device *_pDevice = nullptr;
  MTL::CommandQueue *_pCommandQueue = nullptr;
  MTL::RenderPipelineState *_pPSO = nullptr;

  // Core scene and geometry data
  Scene *_pScene = nullptr;

  // Buffers
  MTL::Buffer *_pSphereBuffer = nullptr;
  MTL::Buffer *_pSphereMaterialBuffer = nullptr;
  MTL::Buffer *_pTriangleVertexBuffer = nullptr;
  MTL::Buffer *_pTriangleIndexBuffer = nullptr;
  MTL::Buffer *_pUniformsBuffer = nullptr;
  MTL::Buffer *_pBVHBuffer = nullptr;
  MTL::Buffer *_pPrimitiveIndexBuffer = nullptr;
  MTL::Buffer *_pTLASBuffer = nullptr;
  MTL::Buffer *_pActiveBuffer = nullptr;
  size_t _blasNodeCount = 0;
  size_t _tlasNodeCount = 0;
  size_t _activeNodeCount = 0;
  size_t _totalNodeCount = 0;
  // Accumulation framebuffers
  MTL::Texture *_accumulationTargets[2] = {nullptr, nullptr};

  struct BoundingSphere {
    simd::float3 center;
    float radius;
  };

  std::vector<Primitive> _allPrimitives;
  std::vector<bool> _activePrimitive;
  std::vector<BoundingSphere> _primitiveBounds;

  bool isInView(const BoundingSphere &b);
  void rebuildAccelerationStructures();
  void updateLODByDistance();

  // Performance metrics
  void beginFrameMetrics();
  void completeFrameMetrics(MTL::CommandBuffer *pCmd);
  std::chrono::high_resolution_clock::time_point _cpuStart;
  double _lastCPUTime = 0.0;
  double _lastGPUTime = 0.0;
  double _lastRaysPerSecond = 0.0;
  size_t _lastRayCount = 0;

  size_t _animationFrame = 0;
};

} // namespace MetalCppPathTracer

#endif // RENDERER_H
