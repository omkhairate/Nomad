#ifndef RENDERER_H
#define RENDERER_H

#include <Metal/Metal.hpp>
#include <MetalKit/MetalKit.hpp>
#include <chrono>
#include <cstdint>
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
  void buildTextures();

  void recalculateViewport();
  bool updateCamera();

  void updateUniforms();
  void draw(MTK::View *pView);
  void drawableSizeWillChange(MTK::View *pView, CGSize size);
  void setDeltaTime(double deltaSeconds);

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
  void rebuildResidentResources(bool forceFullRebuild);
  void ensureBufferCapacity(MTL::Buffer *&buffer, size_t requiredBytes,
                            size_t &currentCapacity,
                            bool allowShrink = false,
                            MTL::ResourceStorageMode storageMode =
                                MTL::ResourceStorageModeManaged);
  struct BoundingSphere {
    simd::float3 center;
    float radius;
  };
  bool isInView(const BoundingSphere &b);
  void updateResidency(bool forceAllToggles = false,
                       bool forceFullRebuild = false);
  bool updateLODByDistance(bool forceAllToggles);
  bool updateEnergyImportance(bool forceAllToggles);
  bool updateRayHitBudget(bool forceAllToggles);
  bool updateScreenSpaceFootprint(bool forceAllToggles);
  void flushResidencyChanges(bool forceFullRebuild);
  void beginFrameMetrics();
  void completeFrameMetrics(MTL::CommandBuffer *pCmd);
  void processRayHitCounters();

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
  MTL::Buffer *_pPrimitiveRemapBuffer = nullptr;
  MTL::Buffer *_pPrimitiveHitBufferGPU = nullptr;
  MTL::Buffer *_pPrimitiveHitReadback = nullptr;
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
  std::vector<uint32_t> _primitiveCooldown;
  std::vector<int32_t> _primitiveToResidentIndex;
  std::vector<BoundingSphere> _primitiveBounds;
  std::vector<SceneObject> _allSceneObjects;
  std::vector<float> _primitiveImportance;
  std::vector<size_t> _energySortedIndices;
  std::vector<float> _primitiveHitScores;
  std::vector<uint32_t> _primitiveHitLastFrame;
  std::vector<size_t> _rayHitSortedIndices;
  std::vector<float> _primitiveScreenCoverage;
  std::vector<size_t> _screenCoverageSortedIndices;
  float _totalPrimitiveImportance = 0.0f;

  std::vector<Primitive> _residentPrimitives;
  std::vector<uint32_t> _residentRemap;
  std::vector<size_t> _recentlyActivated;
  std::vector<size_t> _recentlyDeactivated;

  uint32_t _rayHitRebuildCooldown = 0;

  size_t _residentPrimitiveCount = 0;
  size_t _residentTriangleCount = 0;
  size_t _lightCount = 0;
  float _lightTotalWeight = 0.0f;

  size_t _maxPrimitiveCount = 0;
  size_t _maxTriangleVertexCount = 0;
  size_t _maxTriangleIndexCount = 0;
  size_t _maxBlasNodeCount = 0;
  size_t _maxTlasNodeCount = 0;

  size_t _sphereBufferCapacity = 0;
  size_t _sphereMaterialBufferCapacity = 0;
  size_t _triangleVertexBufferCapacity = 0;
  size_t _triangleIndexBufferCapacity = 0;
  size_t _bvhBufferCapacity = 0;
  size_t _tlasBufferCapacity = 0;
  size_t _primitiveIndexBufferCapacity = 0;
  size_t _activeBufferCapacity = 0;
  size_t _lightIndexBufferCapacity = 0;
  size_t _lightCdfBufferCapacity = 0;
  size_t _primitiveRemapBufferCapacity = 0;
  size_t _primitiveHitBufferCapacity = 0;
  size_t _primitiveHitReadbackCapacity = 0;

  std::chrono::high_resolution_clock::time_point _cpuStart;
  double _lastCPUTime = 0.0;
  double _lastGPUTime = 0.0;
  double _lastRaysPerSecond = 0.0;
  size_t _lastRayCount = 0;

  double _deltaTimeSeconds = 0.0;

  size_t _animationFrame = 0;

  ResidencyParameters _residencyConfig;
};

} // namespace MetalCppPathTracer

#endif // RENDERER_H
