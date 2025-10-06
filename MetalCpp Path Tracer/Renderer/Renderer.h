#ifndef RENDERER_H
#define RENDERER_H

#include <Metal/Metal.hpp>
#include <MetalKit/MetalKit.hpp>
#include <chrono>
#include <cstdint>
#include <simd/simd.h>
#include <vector>

#include "GpuHeapResources.h"
#include "Scene.h"

namespace MetalCppPathTracer {

struct BlasInstanceRecord {
  int32_t blasRootIndex;
  uint32_t primitiveBase;
  uint32_t primitiveCount;
  uint32_t primitiveIndexBase;
};

struct GeometryHandle {
  uint64_t vertexBufferAddress = 0;
  uint64_t indexBufferAddress = 0;
  uint32_t vertexStride = 0;
  uint32_t indexStride = 0;
  uint32_t vertexCount = 0;
  uint32_t indexCount = 0;
  uint32_t instanceSlot = 0;
  uint32_t padding = 0;
};

class Renderer;

struct ResidentObjectGpuResources {
  enum class ResidencyState { Cold, Streaming, Resident };

  bool ensureResident(Renderer &renderer, size_t objectIndex,
                      const SceneObject &object,
                      BlasInstanceRecord &instanceRecord,
                      bool forceRebuild);
  void transitionToStreaming(MTL::CommandBuffer *pending = nullptr);
  void transitionToCold(BlasInstanceRecord &instanceRecord);
  void clearPendingCommand();
  bool isResident() const { return state == ResidencyState::Resident; }

  GpuHeapResources resources;
  size_t byteSize = 0;
  size_t triangleCount = 0;
  size_t vertexCount = 0;
  size_t vertexBufferOffset = 0;
  size_t indexBufferOffset = 0;
  bool geometryValid = false;
  ResidencyState state = ResidencyState::Cold;
  std::chrono::steady_clock::time_point lastStateChange{};
  MTL::CommandBuffer *pendingCommand = nullptr;
};

class Renderer {
  friend struct ResidentObjectGpuResources;

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
  size_t residentNodeCount() const;
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
                            MTL::ResourceOptions storageMode =
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
  bool updateAlwaysResident(bool forceAllToggles);
  void flushResidencyChanges(bool forceFullRebuild);
  void beginFrameMetrics();
  void completeFrameMetrics(MTL::CommandBuffer *pCmd);
  void processRayHitCounters();
  bool buildObjectBlas(size_t objectIndex, const SceneObject &object,
                       ResidentObjectGpuResources &residentResources);
  bool ensureDummyBlas();
  void updateTopLevelAccelerationStructure(
      const std::vector<MTL::AccelerationStructureInstanceDescriptor>
          &descriptors,
      const std::vector<MTL::AccelerationStructure *> &structures);
  void updateAdaptiveSamplingMaps(MTL::CommandBuffer *pCmd);
  bool resetAccumulationTargets(MTL::CommandBuffer *cmd);

  MTL::Device *_pDevice = nullptr;
  MTL::CommandQueue *_pCommandQueue = nullptr;
  MTL::RenderPipelineState *_pPSO = nullptr;
  MTL::ComputePipelineState *_pPathTracePSO = nullptr;
  MTL::ComputePipelineState *_pAdaptiveSamplingPSO = nullptr;

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
  MTL::Buffer *_pInstanceBuffer = nullptr;
  MTL::Buffer *_pTlasInstanceDescriptorBuffer = nullptr;
  MTL::Buffer *_pGeometryHandleBuffer = nullptr;
  size_t _blasNodeCount = 0;
  size_t _tlasNodeCount = 0;
  size_t _activeNodeCount = 0;
  size_t _residentNodeCount = 0;
  size_t _totalNodeCount = 0;
  struct ManagedTextureSlot {
    NS::UInteger width = 0;
    NS::UInteger height = 0;
    MTL::PixelFormat pixelFormat = MTL::PixelFormat::PixelFormatInvalid;
    MTL::TextureUsage usage = MTL::TextureUsage(0);
    MTL::TextureType textureType = MTL::TextureType::TextureType2D;
    MTL::StorageMode storageMode = MTL::StorageMode::StorageModePrivate;
    bool descriptorValid = false;
    bool stagingValid = false;
    MTL::Texture *texture = nullptr;
    MTL::Buffer *stagingBuffer = nullptr;
    size_t stagingCapacity = 0;
  };

  // Accumulation framebuffers
  ManagedTextureSlot _accumulationSlots[2];
  ManagedTextureSlot _sampleCountSlot;
  ManagedTextureSlot _sampleImportanceSlot;

  std::vector<Primitive> _allPrimitives;
  std::vector<bool> _activePrimitive;
  std::vector<uint32_t> _primitiveCooldown;
  std::vector<int32_t> _primitiveToResidentIndex;
  std::vector<size_t> _primitiveToObject;
  std::vector<BoundingSphere> _primitiveBounds;
  std::vector<SceneObject> _allSceneObjects;
  std::vector<BoundingSphere> _objectBounds;
  std::vector<bool> _objectActive;
  std::vector<uint32_t> _objectCooldown;
  std::vector<float> _objectImportance;
  std::vector<size_t> _objectEnergySortedIndices;
  std::vector<float> _primitiveImportance;
  std::vector<size_t> _energySortedIndices;
  std::vector<float> _primitiveHitScores;
  std::vector<float> _primitiveHitScoresSnapshot;
  std::vector<uint32_t> _primitiveHitLastFrame;
  std::vector<size_t> _rayHitSortedIndices;
  std::vector<float> _primitiveScreenCoverage;
  std::vector<size_t> _screenCoverageSortedIndices;
  float _totalPrimitiveImportance = 0.0f;
  float _totalObjectImportance = 0.0f;
  double _textureResidencyMemoryCapMB = 2048.0;

  std::vector<BlasInstanceRecord> _instanceRecords;
  std::vector<Primitive> _residentPrimitives;
  std::vector<uint32_t> _residentRemap;
  std::vector<size_t> _recentlyActivated;
  std::vector<size_t> _recentlyDeactivated;

  std::vector<ResidentObjectGpuResources> _residentObjectGpuResources;

  GpuHeapResources _tlasHeap;
  GpuHeapResources _dummyBlasResources;
  MTL::AccelerationStructure *_pTlasStructure = nullptr;
  MTL::AccelerationStructure *_pDummyBlas = nullptr;
  std::vector<MTL::AccelerationStructureInstanceDescriptor>
      _cachedInstanceDescriptors;
  std::vector<MTL::AccelerationStructure *> _cachedInstancedAccelerationStructures;

  uint32_t _rayHitRebuildCooldown = 0;

  size_t _residentPrimitiveCount = 0;
  size_t _cachedTotalPrimitiveCount = 0;
  size_t _residentTriangleCount = 0;
  size_t _activePrimitiveCount = 0;
  size_t _activeTriangleCount = 0;
  size_t _lightCount = 0;
  float _lightTotalWeight = 0.0f;

  bool _residentBuffersInitialized = false;
  bool _residentCompacted = false;
  bool _useAccelerationStructureBindings = false;
  uint32_t _compactionCooldown = 0;
  std::vector<uint8_t> _cpuActiveMask;
  std::vector<simd::float4> _cachedPrimitiveData;
  std::vector<simd::float4> _cachedMaterialData;
  std::vector<int> _cachedPrimitiveIndices;
  std::vector<simd::float4> _cachedBVHNodes;
  std::vector<simd::float4> _cachedTLASNodes;
  std::vector<simd::float3> _cachedTriangleVertices;
  std::vector<simd::uint3> _cachedTriangleIndices;
  std::vector<uint32_t> _cachedLightIndices;
  std::vector<float> _cachedLightCdf;

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
  size_t _instanceBufferCapacity = 0;
  size_t _geometryHandleBufferCapacity = 0;
  size_t _primitiveHitReadbackCapacity = 0;

  std::chrono::high_resolution_clock::time_point _cpuStart;
  double _lastCPUTime = 0.0;
  double _lastGPUTime = 0.0;
  double _lastRaysPerSecond = 0.0;
  size_t _lastRayCount = 0;

  double _deltaTimeSeconds = 0.0;

  size_t _animationFrame = 0;

  ResidencyParameters _residencyConfig;

  uint32_t _minSamplesPerPixel = 1;
  uint32_t _maxSamplesPerPixel = 4;
  bool _needsAccumulationReset = true;
  bool _accumulationTargetsNeedClear = false;
  MTL::Buffer *_pTextureClearBuffer = nullptr;
  size_t _textureClearBufferCapacity = 0;

  size_t setObjectActive(size_t objectIndex, bool active);
  void configureTextureSlot(ManagedTextureSlot &slot, NS::UInteger width,
                            NS::UInteger height, MTL::PixelFormat format,
                            MTL::TextureUsage usage);
  size_t textureByteSize(const ManagedTextureSlot &slot) const;
  MTL::Texture *requestResidentTexture(ManagedTextureSlot &slot,
                                       MTL::CommandBuffer *cmd,
                                       MTL::BlitCommandEncoder *&blit);
  bool evictTextureSlot(ManagedTextureSlot &slot, MTL::CommandBuffer *cmd,
                        MTL::BlitCommandEncoder *&blit);
  void releaseTextureSlot(ManagedTextureSlot &slot);
  const char *textureSlotLabel(const ManagedTextureSlot &slot) const;
  void updateTextureResidency(MTL::CommandBuffer *cmd);
};

} // namespace MetalCppPathTracer

#endif // RENDERER_H
