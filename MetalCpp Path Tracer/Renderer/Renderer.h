#ifndef RENDERER_H
#define RENDERER_H

#include <Metal/Metal.hpp>
#include <MetalKit/MetalKit.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <simd/simd.h>
#include <thread>
#include <vector>

#include "Scene.h"

namespace MetalCppPathTracer {

class Renderer {
public:
  Renderer(MTL::Device *pDevice);
  ~Renderer();

  void updateVisibleScene();
  void buildShaders();
  void buildTextures();

  void recalculateViewport();
  bool updateCamera();

  void updateUniforms();
  void draw(MTK::View *pView);
  void drawableSizeWillChange(MTK::View *pView, CGSize size);

  bool hasKeyframes() const;
  bool setPrimitiveActive(size_t index, bool active);

  void dumpAccelerationStructure(const std::string &path);

  double currentGPUMemoryMB() const;

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
  void buildBuffers(const Scene &scene);
  void rebuildAccelerationStructures(const Scene &scene);
  void rebuildResidentResources();
  void processPendingResources();
  struct PreparedResources;
  PreparedResources buildResourcesForMask(const std::vector<uint8_t> &mask);
  void commitPreparedResources(const PreparedResources &resources);
  void enqueueLODRebuild(const std::vector<uint8_t> &mask);
  void rebuildWorkerLoop();
  struct BoundingSphere {
    simd::float3 center;
    float radius;
  };
  bool isInView(const BoundingSphere &b);
  void updateLODByDistance();
  void beginFrameMetrics();
  void completeFrameMetrics(MTL::CommandBuffer *pCmd);

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
  MTL::Buffer *_pLightIndexBuffer = nullptr;
  MTL::Buffer *_pLightCdfBuffer = nullptr;
  size_t _blasNodeCount = 0;
  size_t _tlasNodeCount = 0;
  size_t _activeNodeCount = 0;
  size_t _totalNodeCount = 0;
  // Accumulation framebuffers
  MTL::Texture *_accumulationTargets[2] = {nullptr, nullptr};

  std::vector<Primitive> _allPrimitives;
  std::vector<bool> _activePrimitive;
  std::vector<BoundingSphere> _primitiveBounds;
  std::vector<SceneObject> _allSceneObjects;

  struct PreparedResources {
    std::vector<simd::float4> transforms;
    std::vector<simd::float4> materials;
    std::vector<simd::float4> bvhData;
    std::vector<simd::float4> tlasData;
    std::vector<int> primitiveIndices;
    std::vector<uint8_t> activeMask;
    std::vector<simd::float3> triangleVertices;
    std::vector<simd::uint3> triangleIndices;
    std::vector<uint32_t> lightIndices;
    std::vector<float> lightCdf;
    size_t primitiveCount = 0;
    size_t triangleCount = 0;
    size_t blasNodeCount = 0;
    size_t tlasNodeCount = 0;
    size_t activeNodeCount = 0;
    size_t lightCount = 0;
    float lightTotalWeight = 0.0f;
  };

  struct ResourceBuffer {
    PreparedResources data;
    size_t version = 0;
    bool ready = false;
  };

  std::mutex _primitiveMutex;
  std::atomic<bool> _lodDirty{false};
  std::vector<uint8_t> _pendingActiveMask;
  size_t _lodRequestedVersion = 0;
  size_t _lodAppliedVersion = 0;
  size_t _lodBuildingVersion = 0;
  bool _rebuildInProgress = false;
  ResourceBuffer _resourceBuffers[2];
  int _activeResourceIndex = 0;
  int _readyResourceIndex = -1;
  int _inFlightResourceIndex = -1;
  std::thread _rebuildThread;
  std::mutex _workerMutex;
  std::condition_variable _workerCv;
  bool _workerExit = false;

  size_t _residentPrimitiveCount = 0;
  size_t _residentTriangleCount = 0;
  size_t _lightCount = 0;
  float _lightTotalWeight = 0.0f;

  std::chrono::high_resolution_clock::time_point _cpuStart;
  double _lastCPUTime = 0.0;
  double _lastGPUTime = 0.0;
  double _lastRaysPerSecond = 0.0;
  size_t _lastRayCount = 0;

  size_t _animationFrame = 0;
};

} // namespace MetalCppPathTracer

#endif // RENDERER_H
