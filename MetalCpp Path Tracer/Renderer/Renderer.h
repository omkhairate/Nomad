#ifndef RENDERER_H
#define RENDERER_H

#include <Metal/Metal.hpp>
#include <MetalKit/MetalKit.hpp>
#include <cstdint>
#include <chrono>
#include <limits>
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

  // Set a maximum GPU memory budget in megabytes. A value of 0 disables the
  // budget and allows unlimited allocations.
  void setGPUMemoryBudgetMB(double mb);

  // Expose last recorded performance metrics.
  double lastCPUTime() const;
  double lastGPUTime() const;
  double lastRaysPerSecond() const;
  size_t activeNodeCount() const;
  size_t totalNodeCount() const;

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
  MTL::Buffer *_pInstanceMetaBuffer = nullptr;
  MTL::Buffer *_pInstanceArgBuffer = nullptr;
  MTL::ArgumentEncoder *_pInstanceArgEncoder = nullptr;
  MTL::ArgumentEncoder *_pInstanceArgElementEncoder = nullptr;
  size_t _instanceArgStride = 0;
  size_t _blasNodeCount = 0;
  size_t _tlasNodeCount = 0;
  size_t _activeNodeCount = 0;
  size_t _totalNodeCount = 0;
  size_t _residentInstanceCount = 0;
  size_t _currentBlasNodeCount = 0;
  // Accumulation framebuffers
  MTL::Texture *_accumulationTargets[2] = {nullptr, nullptr};

  // GPU memory budgeting
  size_t _gpuMemoryBudget = std::numeric_limits<size_t>::max();
  bool ensureBudget(size_t bytes) const;

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

  struct InstanceGPU {
    MTL::Buffer *blasNodes = nullptr;
    MTL::Buffer *primitives = nullptr;
    MTL::Buffer *materials = nullptr;
    MTL::Buffer *primitiveIndices = nullptr;
    uint32_t primitiveCount = 0;
    uint32_t blasNodeCount = 0;
    uint32_t rootNodeIndex = 0;
  };

  struct InstanceRecord {
    size_t primitiveIndex = 0;
    BoundingSphere bounds;
    simd::float3 aabbMin;
    simd::float3 aabbMax;
    std::vector<simd::float4> primitiveData;
    std::vector<simd::float4> materialData;
    std::vector<int> primitiveIndices;
    std::vector<simd::float4> blasNodes;
    uint32_t cpuPrimitiveCount = 0;
    uint32_t cpuBlasNodeCount = 0;
    ResidencyState state = ResidencyState::NotResident;
    InstanceGPU gpu;
  };

  std::vector<InstanceRecord> _instances;
  std::vector<size_t> _instancesPendingRelease;

  bool streamInInstance(size_t index);
  void streamOutInstance(size_t index);
  void releaseInstanceResources(size_t index);
  void updateInstanceArgument(size_t index);
  void updateInstanceMetadata(size_t index);
  void rebuildTLAS();
  void initializeInstances();
  void processPendingReleases();
  size_t instanceFootprintBytes(const InstanceRecord &inst) const;
  bool createPrivateBuffer(const void *data, size_t size,
                           MTL::Buffer **outBuffer);
  bool createPrivateBuffer(const std::vector<simd::float4> &data,
                           MTL::Buffer **outBuffer);
  bool createPrivateBuffer(const std::vector<int> &data,
                           MTL::Buffer **outBuffer);

  bool _pendingAccumulationReset = false;

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
