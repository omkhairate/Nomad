#ifndef RENDERER_H
#define RENDERER_H

#include <Metal/Metal.hpp>
#include <MetalKit/MetalKit.hpp>
#include <cstdint>
#include <chrono>
#include <limits>
#include <mutex>
#include <simd/simd.h>
#include <utility>
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

  void updateUniforms(bool cameraChanged);
  void draw(MTK::View *pView);
  void drawableSizeWillChange(MTK::View *pView, CGSize size);

  bool hasKeyframes() const;
  // Toggle visibility of an individual primitive without rebuilding TLAS.
  void setPrimitiveActive(size_t index, bool active);

  // Dump acceleration structure to a JSON file for debugging.
  void dumpAccelerationStructure(const std::string &path);

  struct FrameMetrics {
    double cpuTime = 0.0;
    double gpuTime = 0.0;
    double raysPerSecond = 0.0;
    size_t activeNodes = 0;
    size_t offloadedNodes = 0;
    size_t offloadedInstances = 0;
    double gpuMemoryMB = 0.0;
  };

  // Return the amount of GPU memory currently allocated by the device in MB.
  double currentGPUMemoryMB() const;

  // Retrieve the most recently completed frame metrics, if available.
  bool popCompletedFrameMetrics(FrameMetrics &outMetrics);

  // Expose last recorded performance metrics.
  double lastCPUTime() const;
  double lastGPUTime() const;
  double lastRaysPerSecond() const;
  size_t activeNodeCount() const;
  size_t totalNodeCount() const;
  size_t offloadedNodeCount() const;
  size_t offloadedInstanceCount() const;

private:
  MTL::Device *_pDevice = nullptr;
  MTL::CommandQueue *_pCommandQueue = nullptr;
  MTL::RenderPipelineState *_pPSO = nullptr;

  // Core scene and geometry data
  Scene *_pScene = nullptr;

  // Buffers
  MTL::Buffer *_pTriangleVertexBuffer = nullptr;
  MTL::Buffer *_pTriangleIndexBuffer = nullptr;
  MTL::Buffer *_pUniformsBuffer = nullptr;
  MTL::Buffer *_pTLASBuffer = nullptr;
  MTL::Buffer *_pTLASInstanceIndexBuffer = nullptr;
  MTL::Buffer *_pInstanceMetaBuffer = nullptr;
  MTL::Buffer *_pInstanceArgBuffer = nullptr;
  MTL::ArgumentEncoder *_pInstanceArgEncoder = nullptr;
  MTL::ArgumentEncoder *_pInstanceArgElementEncoder = nullptr;
  size_t _instanceArgStride = 0;
  size_t _blasNodeCount = 0;
  size_t _tlasNodeCount = 0;
  size_t _activeNodeCount = 0;
  size_t _totalNodeCount = 0;
  size_t _cpuBlasNodeCount = 0;
  size_t _residentInstanceCount = 0;
  size_t _currentBlasNodeCount = 0;
  size_t _recommendedBudget = std::numeric_limits<size_t>::max();
  size_t _minInstanceFootprint = 0;
  size_t _recentVisibleFootprint = 0;
  bool _manualBudget = false;
  double _targetFrameTime = 1.0 / 30.0;
  // Accumulation framebuffers
  MTL::Texture *_accumulationTargets[2] = {nullptr, nullptr};

  // GPU memory budgeting
  size_t _gpuMemoryBudget = std::numeric_limits<size_t>::max();
  bool ensureBudget(size_t bytes);
  void adjustBudgetForPerformance();

  struct BoundingSphere {
    simd::float3 center;
    float radius;
  };

  std::vector<Primitive> _allPrimitives;

  enum class ResidencyState {
    NotResident,
    StreamingIn,
    Resident,
    StreamingOut,
  };

  struct PendingStreamIn {
    size_t instanceIndex = 0;
    MTL::CommandBuffer *commandBuffer = nullptr;
    std::vector<MTL::Buffer *> stagingBuffers;
  };

  struct InstanceGPU {
    MTL::Buffer *blasNodes = nullptr;
    MTL::Buffer *primitives = nullptr;
    MTL::Buffer *materials = nullptr;
    MTL::Buffer *primitiveIndices = nullptr;
    size_t blasNodesBytes = 0;
    size_t primitiveBytes = 0;
    size_t materialBytes = 0;
    size_t primitiveIndicesBytes = 0;
    uint32_t primitiveCount = 0;
    uint32_t blasNodeCount = 0;
    uint32_t rootNodeIndex = 0;
  };

  struct GPULightData {
    simd::float4 meta;
    simd::float4 emission;
    simd::float4 cdf;
  };

  struct InstanceRecord {
    size_t primitiveIndex = 0;
    uint32_t meshId = kInvalidMeshId;
    BoundingSphere bounds;
    simd::float3 aabbMin;
    simd::float3 aabbMax;
    std::vector<simd::float4> primitiveData;
    std::vector<simd::float4> materialData;
    std::vector<int> primitiveIndices;
    std::vector<simd::float4> blasNodes;
    std::vector<simd::float3> primitiveBoundsMin;
    std::vector<simd::float3> primitiveBoundsMax;
    uint32_t cpuPrimitiveCount = 0;
    uint32_t cpuBlasNodeCount = 0;
    ResidencyState state = ResidencyState::NotResident;
    InstanceGPU gpu;
  };

  std::vector<InstanceRecord> _instances;
  std::vector<size_t> _instancesPendingRelease;
  std::vector<PendingStreamIn> _pendingStreamIns;
  std::vector<GPULightData> _lightTable;

  size_t _trackedAllocatedBytes = 0;
  size_t _uniformsBufferSize = 0;
  size_t _tlasBufferSize = 0;
  size_t _tlasInstanceIndexBufferSize = 0;
  size_t _instanceMetaBufferSize = 0;
  size_t _instanceArgBufferSize = 0;
  size_t _lightBufferSize = 0;
  size_t _accumulationTargetSizes[2] = {0, 0};

  bool streamInInstance(size_t index);
  void streamOutInstance(size_t index);
  void releaseInstanceResources(size_t index);
  void updateInstanceArgument(size_t index);
  void updateInstanceMetadata(size_t index);
  void rebuildLightTable();
  void markLightTableDirty();
  void rebuildTLAS();
  void refreshActiveNodeCount();
  std::pair<size_t, size_t> calculateOffloadedResidency() const;
  void initializeInstances();
  void processPendingReleases();
  void processPendingStreamIns();
  size_t instanceFootprintBytes(const InstanceRecord &inst) const;
  bool createPrivateBuffer(const void *data, size_t size,
                           MTL::Buffer **outBuffer,
                           size_t &sizeTracker);
  bool createPrivateBuffer(const std::vector<simd::float4> &data,
                           MTL::Buffer **outBuffer,
                           size_t &sizeTracker);
  bool createPrivateBuffer(const std::vector<int> &data,
                           MTL::Buffer **outBuffer,
                           size_t &sizeTracker);

  void trackAllocation(size_t bytes);
  void trackDeallocation(size_t bytes);
  void releaseBuffer(MTL::Buffer *&buffer, size_t &sizeTracker);
  void releaseTexture(MTL::Texture *&texture, size_t &sizeTracker);

  bool _pendingAccumulationReset = false;
  bool _lightTableDirty = true;

  MTL::Buffer *_pLightBuffer = nullptr;

  void updateLODByDistance();

  // Performance metrics
  void beginFrameMetrics();
  void completeFrameMetrics(MTL::CommandBuffer *pCmd);
  std::chrono::high_resolution_clock::time_point _cpuStart;
  double _lastCPUTime = 0.0;
  double _lastGPUTime = 0.0;
  double _lastRaysPerSecond = 0.0;
  size_t _lastRayCount = 0;
  size_t _lastOffloadedNodeCount = 0;
  size_t _lastOffloadedInstanceCount = 0;
  FrameMetrics _pendingFrameMetrics;
  bool _hasPendingMetrics = false;
  mutable std::mutex _metricsMutex;

  size_t _animationFrame = 0;
};

} // namespace MetalCppPathTracer

#endif // RENDERER_H
