#ifndef RENDERER_H
#define RENDERER_H

#include <Metal/Metal.hpp>
#include <MetalKit/MetalKit.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <simd/simd.h>
#include <string>
#include <vector>

#include "AlwaysResidentCache.h"
#include "DeferredPurgeQueue.h"
#include "GpuHeapResources.h"
#include "GpuMemoryTracker.h"
#include "ResidencyBudget.h"
#include "TlasScratchTracker.h"
#include "Camera.h"
#include "Scene.h"

namespace NomadPathTracer {

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

struct TextureInfo {
  uint32_t offset = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t flags = 0;
};

class Renderer;

struct ResidentObjectGpuResources {
  enum class ResidencyState { Cold, Streaming, Resident };

  bool ensureResident(Renderer &renderer, size_t objectIndex,
                      const SceneObject &object,
                      BlasInstanceRecord &instanceRecord,
                      bool forceRebuild);
  void transitionToStreaming(MTL::CommandBuffer *pending = nullptr);
  bool transitionToCold(BlasInstanceRecord &instanceRecord);
  void clearPendingCommand();
  bool hasPendingCommands();
  bool waitForPendingCommands(std::chrono::milliseconds timeout);
  bool isResident() const { return state == ResidencyState::Resident; }

  ResidentObjectGpuResources() = default;
  ResidentObjectGpuResources(const ResidentObjectGpuResources &) = delete;
  ResidentObjectGpuResources &operator=(const ResidentObjectGpuResources &) = delete;
  ResidentObjectGpuResources(ResidentObjectGpuResources &&other) noexcept;
  ResidentObjectGpuResources &
  operator=(ResidentObjectGpuResources &&other) noexcept;

  GpuHeapResources resources;
  size_t byteSize = 0;
  size_t triangleCount = 0;
  size_t vertexCount = 0;
  size_t vertexBufferOffset = 0;
  size_t indexBufferOffset = 0;
  bool geometryValid = false;
  ResidencyState state = ResidencyState::Cold;
  std::chrono::steady_clock::time_point lastStateChange{};
  struct PendingCommand {
    MTL::CommandBuffer *command = nullptr;
    bool completed = false;
    bool error = false;
  };

  struct PendingCommandState {
    std::mutex mutex;
    std::vector<PendingCommand> commands;
  };

  std::shared_ptr<PendingCommandState> pendingCommandState =
      std::make_shared<PendingCommandState>();
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
  bool updateCameraStates();

  void updateUniforms(bool cameraChanged);
  void draw(MTK::View *pView);
  void drawableSizeWillChange(MTK::View *pView, CGSize size);
  void setDeltaTime(double deltaSeconds);
  void setBenchmarkMode(bool enabled);
  bool benchmarkModeEnabled() const { return _benchmarkEnabled; }
  void setFrameCaptureEnabled(bool enabled);
  void setFrameCaptureInterval(size_t interval);
  void setMaxRayDepth(uint32_t depth);
  uint32_t maxRayDepth() const;

  bool hasKeyframes() const;
  bool setPrimitiveActive(size_t index, bool active);

  void dumpAccelerationStructure(const std::string &path);

  double currentGPUMemoryMB() const;
  double scratchMemoryMB() const;
  double residentGeometryMemoryMB() const;
  double residentTextureMemoryMB() const;

  double lastCPUTime() const;
  double lastGPUTime() const;
  double lastRaysPerSecond() const;
  size_t activeNodeCount() const;
  size_t residentNodeCount() const;
  size_t totalNodeCount() const;
  bool isAlwaysResidentStrategy() const;

  std::vector<std::pair<simd::float3, float>> _allSpheres;

  Camera::State _primaryCameraState{};
  Camera::State _observerCameraState{};
  bool _observerActive = false;

  struct Chunk {
    std::vector<std::pair<simd::float4, simd::float4>>
        spheres; // (transform, material)
    simd::int3 chunkCoords;
  };

private:
  struct BenchmarkSample;
  struct FrameCaptureRequest;
  struct ManagedTextureSlot;
  void recalculateNodeCounters(const std::vector<bool> &residentMask);
  void rebuildResidentResources(bool forceFullRebuild);
  void ensureBufferCapacity(MTL::Buffer *&buffer, size_t requiredBytes,
                            size_t &currentCapacity,
                            bool allowShrink = false,
                            MTL::ResourceOptions storageMode =
                                MTL::ResourceStorageModeManaged,
                            GpuMemoryTracker::Category category =
                                GpuMemoryTracker::Category::RendererBuffers,
                            const char *label = nullptr,
                            const char *resizeContext = nullptr);
  MTL::Buffer *allocateBuffer(NS::UInteger size,
                              MTL::ResourceOptions storageMode,
                              GpuMemoryTracker::Category category,
                              const char *label = nullptr);
  MTL::Texture *allocateTexture(MTL::TextureDescriptor *descriptor,
                                GpuMemoryTracker::Category category,
                                const char *label = nullptr);
  void trackResource(MTL::Resource *resource,
                     GpuMemoryTracker::Category category);
  void releaseTrackedResource(MTL::Resource *resource);
  GpuMemoryTracker::Category textureCategoryForSlot(
      const ManagedTextureSlot &slot) const;
  void rebuildEnvironmentTexture();
  void releaseEnvironmentTexture();
  struct BoundingSphere {
    simd::float3 center;
    float radius;
  };
  bool isInView(const BoundingSphere &b);
  void updateResidency(bool forceAllToggles = false,
                       bool forceFullRebuild = false);
  bool updateLODByDistance(bool forceAllToggles);
  bool updateEnergyImportance(bool forceAllToggles);
  bool updateUnifiedResidency(bool forceAllToggles);
  bool updateRayHitBudget(bool forceAllToggles);
  bool updateProbabilisticResidency(bool forceAllToggles);
  void resetProbabilisticResidencyState();
  bool updateScreenSpaceFootprint(bool forceAllToggles);
  bool updatePrimitiveScreenCoverageForFrame();
  bool updateEnvironmentHitResidency(bool forceAllToggles);
  bool updatePredictiveResidency(bool forceAllToggles);
  bool updateAlwaysResident(bool forceAllToggles);
  void prewarmAlwaysResidentResources();
  void flushResidencyChanges(bool forceFullRebuild);
  void beginFrameMetrics();
  void completeFrameMetrics(MTL::CommandBuffer *pCmd);
  size_t residentGeometryMemoryBytes() const;
  size_t geometryResidencyCapBytes() const;
  double effectiveTotalGpuMemoryCapMB() const;
  size_t totalGpuMemoryCapBytes() const;
  size_t minimumResidentFootprintBytes() const;
  size_t essentialGeometryMemoryBytes() const;
  size_t historyFootprintBytes() const;
  bool checkTotalMemoryCap(size_t requestedBytes, size_t existingBytes,
                           const char *context);
  void recordGeometryResidencyHardCapDenied(size_t objectIndex,
                                            size_t requestedBytes,
                                            size_t existingBytes,
                                            size_t residentBytes,
                                            size_t capBytes,
                                            const char *context);
  bool checkGeometryResidencyCap(size_t objectIndex, size_t requestedBytes,
                                 size_t existingBytes,
                                 const char *context);
  std::vector<bool> buildResidentMaskFromGpuResources() const;
  void trackFrameCommandBuffer(MTL::CommandBuffer *commandBuffer);
  void recordPathTraceCommandTime(double gpuMs, size_t tileCount);
  size_t updatePathTraceTilesPerCommandBudget();
  bool waitForPendingFrameCommands(
      std::chrono::milliseconds timeout,
      std::chrono::steady_clock::time_point *waitSnapshot = nullptr);
  std::vector<float> computeUnifiedImportance(float &outTotalScore);
  bool flushRayHitCopy();
  bool rayHitCopyReady() const;
  void processRayHitCounters();
  bool buildObjectBlas(size_t objectIndex, const SceneObject &object,
                       ResidentObjectGpuResources &residentResources);
  bool ensureDummyBlas();
  void updateTopLevelAccelerationStructure(
      const std::vector<MTL::AccelerationStructureInstanceDescriptor>
          &descriptors,
      const std::vector<MTL::AccelerationStructure *> &structures);
  struct SceneAccelerationBuildResult {
    size_t blasNodeCount = 0;
    size_t tlasNodeCount = 0;
  };
  SceneAccelerationBuildResult buildSceneAccelerationStructures(
      size_t primitiveCount, size_t primitiveHitBytes);
  void ensureTlasBuildEvent();
  void waitForPendingBlasBuilds();
  void waitForPendingTlasBuild();
  bool hasPendingTlasBuild() const;
  void finalizePendingTlasScratchResize(bool force = false);
  void updateTlasScratchResidentBytes(NS::UInteger bytes);
  struct PendingBlasBuild;
  void enqueueBlasBuild(const std::shared_ptr<PendingBlasBuild> &buildRequest);
  void processBlasBuildQueue();
  enum class BlasBuildStartResult { Started, DeferredCap, Failed };
  BlasBuildStartResult startBlasBuild(
      const std::shared_ptr<PendingBlasBuild> &buildRequest);
  void handleCompletedBlasBuild(
      const std::shared_ptr<PendingBlasBuild> &buildRequest, bool success);
  void requestResidentEviction(size_t objectIndex,
                               ResidentObjectGpuResources &resident,
                               BlasInstanceRecord &instanceRecord);
  void cancelPendingResidentEviction(size_t objectIndex,
                                     ResidentObjectGpuResources &resident);
  void handleDeferredBlasEvictions(MTL::CommandBuffer *commandBuffer,
                                   bool success);
  bool transitionResidentToCold(size_t objectIndex,
                                ResidentObjectGpuResources &resident,
                                BlasInstanceRecord &instanceRecord,
                                bool removePending = true);
  bool submitAsyncCommandBuffer(MTL::CommandBuffer *commandBuffer,
                               std::function<void(bool)> completion);
  void updateAdaptiveSamplingMaps(MTL::CommandBuffer *pCmd);
  void rebuildMaterialTextures();
  void clearMaterialTextures();
  void initializeBenchmarking();
  void ensureBenchmarkStream();
  void writeBenchmarkHeader();
  void writeBenchmarkRow(const BenchmarkSample &sample);
  std::string residencyStrategyName(ResidencyStrategy strategy) const;
  void ensureFrameCaptureDirectory();
  std::shared_ptr<FrameCaptureRequest>
  encodeFrameCapture(MTL::Texture *colorTexture, MTL::Texture *albedoTexture,
                     MTL::Texture *normalTexture, uint64_t frameIndex,
                     MTL::CommandBuffer *cmd, MTL::BlitCommandEncoder *&blit);
  void finalizeFrameCapture(const std::shared_ptr<FrameCaptureRequest> &capture);
  void processPendingCapturedFrames();
  std::array<simd::float3, 8>
  buildFrustumCorners(const Camera::State &state, float nearDistance,
                      float farDistance) const;
  MTL::Buffer *acquireBlasScratchBuffer(NS::UInteger requestedSize,
                                       NS::UInteger &allocatedSize,
                                       bool &reused);
  void recycleBlasScratchBuffer(MTL::Buffer *buffer, NS::UInteger size);
  void releaseBlasScratchPool();
  void updateBlasScratchResidencyBudget();
  double residencyMemoryMB() const;
  size_t textureByteSize(MTL::Texture *texture) const;
  
  MTL::Device *_pDevice = nullptr;
  MTL::CommandQueue *_pCommandQueue = nullptr;
  MTL::RenderPipelineState *_pPSO = nullptr;
  MTL::RenderPipelineState *_pOverlayPSO = nullptr;
  MTL::ComputePipelineState *_pPathTracePSO = nullptr;
  MTL::ComputePipelineState *_pAdaptiveSamplingPSO = nullptr;
  MTL::ComputePipelineState *_pRestirTemporalPSO = nullptr;
  MTL::ComputePipelineState *_pRestirSpatialPSO = nullptr;
  MTL::ComputePipelineState *_pRestirShadePSO = nullptr;

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
  MTL::Buffer *_pRestirReservoirBuffer = nullptr;
  MTL::Buffer *_pRestirReservoirHistoryBuffer = nullptr;
  struct FrameCommandBufferRecord {
    MTL::CommandBuffer *buffer = nullptr;
    std::chrono::steady_clock::time_point trackedSince{};
  };

  std::vector<FrameCommandBufferRecord> _frameCommandBuffers;
  std::mutex _frameCommandBufferMutex;
  MTL::CommandBuffer *_lastRayHitCommandBuffer = nullptr;
  bool _rayHitCopyError = false;
  std::mutex _pathTraceBudgetMutex;
  std::deque<double> _pathTraceGpuMsPerTileHistory;
  size_t _pathTraceTilesPerCommandBudget = 0;
  bool _pathTraceCommandTimeout = false;
  bool _restirHistoryValid = false;
  MTL::Buffer *_pLightIndexBuffer = nullptr;
  MTL::Buffer *_pLightCdfBuffer = nullptr;
  MTL::Buffer *_pInstanceBuffer = nullptr;
  MTL::Buffer *_pTlasInstanceDescriptorBuffer = nullptr;
  MTL::Buffer *_pGeometryHandleBuffer = nullptr;
  MTL::Buffer *_pFrustumVertexBuffer = nullptr;
  MTL::Buffer *_pTlasScratchBuffer = nullptr;
  NS::UInteger _tlasScratchCapacity = 0;
  size_t _tlasScratchResidentBytes = 0;
  TlasScratchTracker _tlasScratchTracker;
  ResidencyBudget _residencyBudget;
  GpuMemoryTracker _gpuMemoryTracker;
  size_t _blasNodeCount = 0;
  size_t _tlasNodeCount = 0;
  size_t _activeNodeCount = 0;
  size_t _residentNodeCount = 0;
  size_t _totalNodeCount = 0;
  struct ManagedTextureSlot {
    enum class HistoryBacking { None, SharedBuffer, CpuData };

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
    HistoryBacking historyBacking = HistoryBacking::None;
    bool historyIsProxy = false;
    bool needsGpuRefresh = false;
    NS::UInteger historyWidth = 0;
    NS::UInteger historyHeight = 0;
    size_t historyBytesPerRow = 0;
    std::vector<uint8_t> historyData;
    uint64_t lastUsedFrame = 0;
  };

  static constexpr size_t kRestirReservoirStride = 48;

  // Framebuffers
  ManagedTextureSlot _colorSlot;
  ManagedTextureSlot _albedoSlot;
  ManagedTextureSlot _normalSlot;
  ManagedTextureSlot _positionSlot;
  ManagedTextureSlot _albedoHistorySlot;
  ManagedTextureSlot _normalHistorySlot;
  ManagedTextureSlot _positionHistorySlot;

  struct PendingBlasBuild {
    Renderer *renderer = nullptr;
    ResidentObjectGpuResources *resident = nullptr;
    size_t objectIndex = 0;
    std::vector<simd::float3> vertices;
    std::vector<uint32_t> indices;
    size_t triangleCount = 0;
    size_t vertexCount = 0;
    NS::UInteger totalHeapBytes = 0;
    MTL::AccelerationStructureTriangleGeometryDescriptor *geometryDesc = nullptr;
    MTL::PrimitiveAccelerationStructureDescriptor *accelDesc = nullptr;
    NS::Array *geometryArray = nullptr;
    MTL::AccelerationStructure *accelerationStructure = nullptr;
    MTL::Buffer *vertexStaging = nullptr;
    MTL::Buffer *indexStaging = nullptr;
    MTL::Buffer *scratchBuffer = nullptr;
    NS::UInteger scratchSize = 0;
    MTL::CommandBuffer *commandBuffer = nullptr;

    void releaseResources();
  };

  static constexpr size_t kMaxBlasBuildsInFlight = 3;
  std::deque<std::shared_ptr<PendingBlasBuild>> _pendingBlasBuilds;
  std::deque<std::shared_ptr<PendingBlasBuild>> _activeBlasBuilds;
  std::map<NS::UInteger, std::vector<MTL::Buffer *>> _blasScratchPool;
  size_t _blasScratchPoolAvailableBytes = 0;
  size_t _blasScratchPoolInUseBytes = 0;
  size_t _blasScratchPoolCreatedCount = 0;
  size_t _blasScratchPoolReusedCount = 0;

  struct BenchmarkSample {
    size_t frameIndex = 0;
    size_t rayCount = 0;
    size_t primitiveActivations = 0;
    size_t primitiveDeactivations = 0;
    size_t objectActivations = 0;
    size_t objectDeactivations = 0;
    size_t activePrimitiveCount = 0;
    size_t residentPrimitiveCount = 0;
    size_t totalPrimitiveCount = 0;
    size_t activeTriangleCount = 0;
    size_t residentTriangleCount = 0;
    size_t totalTriangleCount = 0;
    size_t activeNodeCount = 0;
    size_t residentNodeCount = 0;
    size_t totalNodeCount = 0;
    size_t activeObjectCount = 0;
    size_t residentObjectCount = 0;
    double gpuMemoryMB = 0.0;
    double scratchMemoryMB = 0.0;
    double residentGeometryMemoryMB = 0.0;
    double residentTextureMemoryMB = 0.0;
    double residencyMemoryMB = 0.0;
    double textureMemoryCapMB = 0.0;
    double geometryMemoryCapMB = 0.0;
    double totalMemoryCapMB = 0.0;
    double minimumResidentFootprintMB = 0.0;
    double totalMemoryCapRelaxedMB = 0.0;
    double deltaTimeSeconds = 0.0;
    double wallSeconds = 0.0;
    double cpuTimeSeconds = 0.0;
    double gpuTimeSeconds = 0.0;
    double raysPerSecond = 0.0;
    double avgHitProbability = 0.0;
    double p95HitProbability = 0.0;
    double probabilityThreshold = 0.0;
    double probabilityTargetFraction = 0.0;
    double probabilityVisibleFloor = 0.0;
    double cameraMotionMetric = 0.0;
    std::string primitiveProbabilities;
    std::string objectProbabilities;
    size_t probabilisticToggles = 0;
    size_t probabilityTargetPrimitives = 0;
    size_t probabilityInitialDesiredPrimitives = 0;
    size_t probabilityFinalDesiredPrimitives = 0;
    size_t probabilityTrimmedPrimitives = 0;
    bool probabilityBudgetHit = false;
    size_t screenMinPixelCoverageSkips = 0;
    double environmentTargetActiveFraction = 0.0;
    double environmentEscapeThreshold = 0.0;
    std::string environmentDepthWeights;
    std::string environmentDepthRadii;
    double envHighEscapeThreshold = 0.0;
    double envLowEscapeThreshold = 0.0;
    double globalEnvEscape = 0.0;
    size_t environmentActivationFloor = 0;
    ResidencyStrategy strategy = ResidencyStrategy::DistanceLOD;
    std::string strategyName;
    uint32_t minSamplesPerPixel = 1;
    uint32_t maxSamplesPerPixel = 1;
    bool residentCompacted = false;
    bool overMemoryCap = false;
    bool geometryOverMemoryCap = false;
    bool totalOverMemoryCap = false;
    bool totalMemoryEvictionStall = false;
    size_t geometryCapHitCount = 0;
    size_t geometryHardCapDeniedCount = 0;
    size_t totalMemoryOverageWarnings = 0;
    size_t totalMemoryCapDeniedCount = 0;
  };

  struct FrameCaptureRequest {
    uint64_t frameIndex = 0;
    std::string filePath;
    std::string albedoPath;
    std::string normalPath;
    MTL::Buffer *buffer = nullptr;
    MTL::Buffer *albedoBuffer = nullptr;
    MTL::Buffer *normalBuffer = nullptr;
    size_t width = 0;
    size_t height = 0;
    size_t alignedRowBytes = 0;
    size_t albedoAlignedRowBytes = 0;
    size_t normalAlignedRowBytes = 0;
    MTL::PixelFormat format = MTL::PixelFormat::PixelFormatInvalid;
    MTL::PixelFormat albedoFormat = MTL::PixelFormat::PixelFormatInvalid;
    MTL::PixelFormat normalFormat = MTL::PixelFormat::PixelFormatInvalid;
  };

  std::vector<std::shared_ptr<FrameCaptureRequest>> _completedCaptures;
  std::mutex _captureMutex;
  std::atomic<bool> _captureOutputsPending{false};

  std::vector<Primitive> _allPrimitives;
  std::vector<bool> _activePrimitive;
  std::vector<uint32_t> _primitiveCooldown;
  std::vector<int32_t> _primitiveToResidentIndex;
  std::vector<size_t> _primitiveToObject;
  std::vector<BoundingSphere> _primitiveBounds;
  std::vector<SceneObject> _allSceneObjects;
  double _frameCameraMotionMetric = 0.0;
  Camera::State _lastUniformCameraState{};
  bool _lastUniformCameraStateValid = false;
  std::vector<BoundingSphere> _objectBounds;
  std::vector<bool> _objectActive;
  std::vector<uint32_t> _objectCooldown;
  std::vector<uint64_t> _objectLastToggleFrame;
  std::vector<uint8_t> _desiredObjectState;
  std::vector<uint8_t> _pendingDesiredObjects;
  std::vector<uint64_t> _desiredObjectPromotionFrame;
  std::vector<uint64_t> _desiredObjectDemotionFrame;
  std::vector<uint32_t> _objectDemotionDwell;
  std::vector<float> _primitiveImportance;
  std::vector<float> _objectImportance;
  std::vector<float> _objectImportanceHistory;
  std::vector<size_t> _energySortedIndices;
  std::vector<float> _primitiveHitScores;
  std::vector<float> _primitiveHitScoresSnapshot;
  std::vector<uint32_t> _primitiveHitLastFrame;
  std::vector<float> _primitiveRayContributions;
  std::vector<uint32_t> _primitiveRaysTestedLastFrame;
  std::vector<float> _primitiveHitAlpha;
  std::vector<float> _primitiveHitBeta;
  std::vector<float> _primitiveHitProbability;
  std::vector<float> _primitiveHitVariance;
  std::vector<float> _primitivePosteriorMass;
  std::vector<float> _primitiveExplorationScore;
  std::vector<uint32_t> _primitiveIdleFrames;
  std::vector<uint8_t> _primitiveVisible;
  std::vector<size_t> _rayHitSortedIndices;
  std::vector<size_t> _probabilitySortedIndices;
  std::vector<float> _objectHitAlpha;
  std::vector<float> _objectHitBeta;
  std::vector<float> _objectHitProbability;
  std::vector<float> _objectHitVariance;
  std::vector<float> _objectPosteriorMass;
  std::vector<float> _objectExplorationScore;
  std::vector<float> _objectRayHitScore;
  std::vector<uint32_t> _objectHitLastFrame;
  std::vector<uint32_t> _objectRaysTestedLastFrame;
  std::vector<uint32_t> _objectIdleFrames;
  std::vector<uint8_t> _objectVisible;
  std::vector<float> _objectVisibilityEvidence;
  std::vector<size_t> _objectProbabilitySortedIndices;
  std::vector<float> _objectDepthWeight;
  std::vector<uint64_t> _objectDepthWeightVersion;
  std::vector<float> _primitiveScreenCoverage;
  std::vector<float> _primitiveDistanceFalloffCache;
  std::vector<uint8_t> _primitiveCoverageDirty;
  std::vector<uint64_t> _primitiveCoverageBoundsVersion;
  std::vector<uint8_t> _primitiveCoverageVisibilityKey;
  std::vector<uint8_t> _primitiveUnifiedPrevVisible;
  std::vector<float> _primitiveUnifiedLastVisibleHit;
  std::vector<float> _primitiveUnifiedLastVisibleEnergy;
  std::vector<uint64_t> _primitiveBoundsVersion;
  std::vector<uint64_t> _objectBoundsVersion;
  uint64_t _boundsVersionCounter = 1;
  uint64_t _cameraVersion = 1;
  uint64_t _depthWeightCameraVersion = 0;
  uint64_t _lastResidentFlushCameraVersion = 0;
  uint64_t _coverageCameraVersion = 0;
  uint64_t _coverageUpdatedFrame = 0;
  std::vector<size_t> _screenCoverageSortedIndices;
  float _totalPrimitiveImportance = 0.0f;
  double _textureResidencyMemoryCapMB = 2048.0;
  double _geometryResidencyMemoryCapMB = 2048.0;
  double _totalGpuMemoryCapMB = 4096.0;

  std::vector<BlasInstanceRecord> _instanceRecords;
  std::vector<Primitive> _residentPrimitives;
  std::vector<uint32_t> _residentRemap;
  std::vector<size_t> _recentlyActivated;
  std::vector<size_t> _recentlyDeactivated;
  std::vector<size_t> _dirtyResidentObjects;
  std::vector<bool> _objectResidentState;

  std::vector<ResidentObjectGpuResources> _residentObjectGpuResources;

  DeferredPurgeQueue<ResidentObjectGpuResources, MTL::CommandBuffer>
      _pendingBlasEvictions;

  GpuHeapResources _tlasHeap;
  GpuHeapResources _dummyBlasResources;
  MTL::AccelerationStructure *_pTlasStructure = nullptr;
  MTL::AccelerationStructure *_pDummyBlas = nullptr;
  bool _dummyBlasBuildInFlight = false;
  std::vector<MTL::AccelerationStructureInstanceDescriptor>
      _cachedInstanceDescriptors;
  std::vector<MTL::AccelerationStructure *> _cachedInstancedAccelerationStructures;
  MTL::Buffer *_pTlasDescriptorStaging = nullptr;
  size_t _tlasDescriptorStagingCapacity = 0;
  MTL::SharedEvent *_pTlasBuildEvent = nullptr;
  uint64_t _tlasBuildEventValue = 0;
  std::atomic<uint64_t> _tlasCompletedEventValue{0};

  uint32_t _rayHitRebuildCooldown = 0;

  size_t _residentPrimitiveCount = 0;
  size_t _cachedTotalPrimitiveCount = 0;
  size_t _residentTriangleCount = 0;
  size_t _activePrimitiveCount = 0;
  size_t _activeTriangleCount = 0;
  size_t _lightCount = 0;
  float _lightTotalWeight = 0.0f;
  float _lastActivePrimitiveRatio = 1.0f;
  float _lastFrameGlobalEnvEscape = 0.0f;

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
  std::vector<TextureInfo> _cachedTextureInfos;
  std::vector<simd::float4> _cachedTextureData;
  std::vector<MTL::Texture *> _materialTextures;
  MTL::Texture *_environmentTexture = nullptr;
  MTL::SamplerState *_environmentSampler = nullptr;
  std::string _environmentTexturePath;
  float _environmentBrightness = 1.0f;
  std::vector<uint32_t> _cachedLightIndices;
  std::vector<float> _cachedLightCdf;

  struct MeshGroupInfo {
    int meshGroupId = -1;
    std::vector<size_t> objectIndices;
    size_t primitiveCount = 0;
  };

  std::vector<MeshGroupInfo> _meshGroups;
  std::vector<size_t> _objectPrimitiveCounts;
  std::vector<size_t> _objectActivePrimitiveCounts;
  bool _anyMeshGroups = false;

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
  size_t _reservoirBufferCapacity = 0;
  size_t _reservoirHistoryBufferCapacity = 0;
  size_t _primitiveHitReadbackCapacity = 0;
  size_t _frustumVertexCapacity = 0;

  std::chrono::high_resolution_clock::time_point _cpuStart;
  double _lastCPUTime = 0.0;
  double _lastGPUTime = 0.0;
  double _lastRaysPerSecond = 0.0;
  size_t _lastRayCount = 0;

  double _deltaTimeSeconds = 0.0;

  size_t _animationFrame = 0;

  ResidencyParameters _residencyConfig;

  bool _benchmarkEnabled = false;
  bool _benchmarkHeaderWritten = false;
  std::ofstream _benchmarkStream;
  std::string _benchmarkFilePath;
  size_t _benchmarkFrameCounter = 0;
  std::chrono::steady_clock::time_point _benchmarkStartTime;
  size_t _framePrimitiveActivations = 0;
  size_t _framePrimitiveDeactivations = 0;
  size_t _frameObjectActivations = 0;
  size_t _frameObjectDeactivations = 0;
  size_t _frameProbabilisticToggles = 0;
  size_t _frameProbabilityTargetPrimitives = 0;
  size_t _frameProbabilityInitialDesiredPrimitives = 0;
  size_t _frameProbabilityFinalDesiredPrimitives = 0;
  size_t _frameProbabilityTrimmedPrimitives = 0;
  bool _frameProbabilityBudgetHit = false;
  size_t _frameScreenMinPixelCoverageSkips = 0;
  size_t _frameEnvironmentActivationFloor = 0;
  size_t _frameGeometryResidencyCapHitCount = 0;
  size_t _frameGeometryResidencyHardCapDeniedCount = 0;
  size_t _frameTotalMemoryOverageWarnings = 0;
  size_t _frameTotalMemoryCapDeniedCount = 0;
  bool _frameTotalMemoryEvictionStall = false;
  double _frameMinimumResidentFootprintMB = 0.0;
  size_t _geometryResidencyCapHitCount = 0;
  size_t _geometryResidencyHardCapDeniedCount = 0;
  size_t _totalMemoryCapDeniedCount = 0;
  size_t _pendingGeometryResidencyOverageBytes = 0;
  size_t _pendingTotalMemoryOverageBytes = 0;
  size_t _totalMemoryBelowFootprintFrames = 0;
  double _totalMemoryCapRelaxedMB = 0.0;
  bool _memoryCapFallbackActive = false;
  ResidencyStrategy _frameStrategy = ResidencyStrategy::DistanceLOD;
  ResidencyStrategy _lastResidencyStrategy = ResidencyStrategy::DistanceLOD;
  AlwaysResidentCache _alwaysResidentCache;
  bool _forceAlwaysResidentActivation = false;
  bool _pendingAlwaysResidentPrewarm = false;
  bool _frameCaptureEnabled = false;
  size_t _frameCaptureInterval = 4;
  uint64_t _renderedFrameCount = 0;
  bool _frameCaptureDirectoryInitialized = false;
  std::string _frameCaptureDirectory;
  std::deque<BenchmarkSample> _pendingBenchmarkSamples;

  uint32_t _minSamplesPerPixel = 1;
  uint32_t _maxSamplesPerPixel = 4;
  uint32_t _restirSpatialRadius = 2;
  uint32_t _restirSpatialNeighborCount = 8;
  MTL::Buffer *_pTextureClearBuffer = nullptr;
  size_t _textureClearBufferCapacity = 0;

  size_t setObjectActive(size_t objectIndex, bool active);
  void configureTextureSlot(ManagedTextureSlot &slot, NS::UInteger width,
                            NS::UInteger height, MTL::PixelFormat format,
                            MTL::TextureUsage usage);
  size_t textureByteSize(const ManagedTextureSlot &slot) const;
  void clearTextureHistory(ManagedTextureSlot &slot);
  MTL::Texture *requestResidentTexture(ManagedTextureSlot &slot,
                                       MTL::CommandBuffer *cmd,
                                       MTL::BlitCommandEncoder *&blit);
  bool evictTextureSlot(ManagedTextureSlot &slot, MTL::CommandBuffer *cmd,
                        MTL::BlitCommandEncoder *&blit);
  void releaseTextureSlot(ManagedTextureSlot &slot);
  const char *textureSlotLabel(const ManagedTextureSlot &slot) const;
  void updateTextureResidency(MTL::CommandBuffer *cmd);
  void updateGeometryResidency(MTL::CommandBuffer *cmd);
  void disableHistoryForMemoryCap();
};

} // namespace NomadPathTracer

#endif // RENDERER_H
