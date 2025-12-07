#include "Renderer.h"

#include "Camera.h"
#include "IncrementalUpdateUtils.h"
#include "InputSystem.h"
#include "ParallelFor.h"
#include "Scene.h"
#include "SceneLoader.h"
#include "../offline/denoiser_settings.h"
#include <Metal/MTLArgument.hpp>
#include <Metal/MTLComputePipeline.hpp>
#include <algorithm>
#include <array>
#include <cassert>
#include <CoreFoundation/CoreFoundation.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <dispatch/dispatch.h>
#include <thread>
#include <limits>
#include <functional>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <future>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <atomic>
#include <deque>
#include <simd/simd.h>
#include <sstream>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define TINYEXR_IMPLEMENTATION
#include "../tinyexr.h"
#include "../Textures/EnvironmentTexture.h"

namespace {
class TaskLimiter {
public:
  explicit TaskLimiter(size_t maxTasks) : _maxTasks(maxTasks) {}

  void acquire() {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [&] { return _active < _maxTasks; });
    ++_active;
  }

  void release() {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_active > 0) {
      --_active;
      lock.unlock();
      _cv.notify_one();
    }
  }

private:
  size_t _maxTasks = 0;
  size_t _active = 0;
  std::mutex _mutex;
  std::condition_variable _cv;
};

TaskLimiter &sceneBvhTaskLimiter() {
  const size_t hardwareThreads =
      std::max<size_t>(std::thread::hardware_concurrency(), 2);
  constexpr size_t kMinSceneBvhTasks = 4;
  const size_t maxTasks = std::max<size_t>(hardwareThreads, kMinSceneBvhTasks);
  static TaskLimiter limiter(maxTasks);
  return limiter;
}
} // namespace

namespace {
constexpr float kIdleVisibleExploreSeed = 1.0f;
}

namespace MetalCppPathTracer {

ResidentObjectGpuResources::ResidentObjectGpuResources(
    ResidentObjectGpuResources &&other) noexcept {
  std::scoped_lock lock(other.pendingCommandsMutex);
  resources = std::move(other.resources);
  byteSize = other.byteSize;
  triangleCount = other.triangleCount;
  vertexCount = other.vertexCount;
  vertexBufferOffset = other.vertexBufferOffset;
  indexBufferOffset = other.indexBufferOffset;
  geometryValid = other.geometryValid;
  state = other.state;
  lastStateChange = other.lastStateChange;
  pendingCommands = std::move(other.pendingCommands);
}

ResidentObjectGpuResources &
ResidentObjectGpuResources::operator=(ResidentObjectGpuResources &&other) noexcept {
  if (this == &other)
    return *this;

  std::scoped_lock lock(pendingCommandsMutex, other.pendingCommandsMutex);
  resources = std::move(other.resources);
  byteSize = other.byteSize;
  triangleCount = other.triangleCount;
  vertexCount = other.vertexCount;
  vertexBufferOffset = other.vertexBufferOffset;
  indexBufferOffset = other.indexBufferOffset;
  geometryValid = other.geometryValid;
  state = other.state;
  lastStateChange = other.lastStateChange;
  pendingCommands = std::move(other.pendingCommands);
  return *this;
}

void ResidentObjectGpuResources::clearPendingCommand() {
  std::lock_guard<std::mutex> lock(pendingCommandsMutex);
  using Status = MTL::CommandBufferStatus;

  auto isComplete = [](PendingCommand &pending) {
    if (pending.completed || pending.error)
      return true;

    if (!pending.command)
      return true;

    auto status = pending.command->status();
    if (status == Status::CommandBufferStatusCompleted) {
      pending.completed = true;
      return true;
    }
    if (status == Status::CommandBufferStatusError) {
      pending.error = true;
      return true;
    }
    return false;
  };

  auto releaseCommand = [](PendingCommand &pending) {
    if (pending.command) {
      pending.command->release();
      pending.command = nullptr;
    }
  };

  pendingCommands.erase(
      std::remove_if(pendingCommands.begin(), pendingCommands.end(),
                     [&](PendingCommand &pending) {
                       if (!isComplete(pending))
                         return false;
                       releaseCommand(pending);
                       return true;
                     }),
      pendingCommands.end());
}

void ResidentObjectGpuResources::transitionToStreaming(
    MTL::CommandBuffer *pending) {
  clearPendingCommand();
  if (pending) {
    PendingCommand pendingRecord{};
    pendingRecord.command = pending;
    pendingRecord.command->retain();

    pendingRecord.command->addCompletedHandler(
        [this](MTL::CommandBuffer *cmd) {
          std::lock_guard<std::mutex> lock(pendingCommandsMutex);
          for (auto &pending : pendingCommands) {
            if (pending.command == cmd) {
              auto status = cmd->status();
              pending.completed =
                  status == MTL::CommandBufferStatusCompleted;
              pending.error = status == MTL::CommandBufferStatusError;
              break;
            }
          }
        });

    std::lock_guard<std::mutex> lock(pendingCommandsMutex);
    pendingCommands.push_back(std::move(pendingRecord));
  }
  state = ResidencyState::Streaming;
  lastStateChange = std::chrono::steady_clock::now();
}

bool ResidentObjectGpuResources::hasPendingCommands() {
  std::lock_guard<std::mutex> lock(pendingCommandsMutex);
  for (auto &pending : pendingCommands) {
    if (pending.completed || pending.error)
      continue;

    auto status = pending.command ? pending.command->status()
                                  : MTL::CommandBufferStatusCompleted;
    if (status == MTL::CommandBufferStatusCompleted) {
      pending.completed = true;
      continue;
    }
    if (status == MTL::CommandBufferStatusError) {
      pending.error = true;
      continue;
    }

    return true;
  }
  return false;
}

void ResidentObjectGpuResources::transitionToCold(
    BlasInstanceRecord &instanceRecord) {
  clearPendingCommand();
  resources.makeResourcesPurgeable();
  resources.releaseAllAllocations();
  byteSize = 0;
  triangleCount = 0;
  vertexCount = 0;
  vertexBufferOffset = 0;
  indexBufferOffset = 0;
  geometryValid = false;
  state = ResidencyState::Cold;
  lastStateChange = std::chrono::steady_clock::now();
  instanceRecord.primitiveBase = 0;
  instanceRecord.primitiveIndexBase = 0;
  instanceRecord.blasRootIndex = -1;
  instanceRecord.primitiveCount = 0;
}

bool ResidentObjectGpuResources::ensureResident(
    Renderer &renderer, size_t objectIndex, const SceneObject &object,
    BlasInstanceRecord &instanceRecord, bool forceRebuild) {
  renderer.cancelPendingResidentEviction(objectIndex, *this);

  const auto previousState = state;
  const auto previousStateChange = lastStateChange;
  const bool previousGeometryValid = geometryValid;

  if (isResident() && !forceRebuild) {
    lastStateChange = std::chrono::steady_clock::now();
    return true;
  }

  transitionToStreaming();
  geometryValid = false;
  bool built = renderer.buildObjectBlas(objectIndex, object, *this);
  if (!built) {
    if (renderer.isAlwaysResidentStrategy()) {
      state = previousState;
      geometryValid = previousGeometryValid;
      lastStateChange = previousStateChange;
      return false;
    }
    renderer.transitionResidentToCold(objectIndex, *this, instanceRecord);
    return false;
  }
  return true;
}

} // namespace MetalCppPathTracer

using namespace MetalCppPathTracer;

ParallelForConfig primitivePackingConfig(size_t primitiveCount) {
  // Derived from empirical packing timings: keep tiny scenes on a single core
  // to avoid synchronization overhead, then scale aggressively for larger
  // batches where memory traffic dominates.
  ParallelForConfig config{};
  if (primitiveCount < 2048) {
    config.minChunkSize = 64;
    config.preferredChunkSize = 128;
  } else if (primitiveCount < 32768) {
    config.minChunkSize = 256;
    config.preferredChunkSize = 512;
  } else {
    config.minChunkSize = 512;
    config.preferredChunkSize = 1024;
  }
  return config;
}

ParallelForConfig textureUploadConfig(size_t textureCount, size_t totalTexels) {
  // Texture upload benefits from smaller chunks for texture-heavy scenes with
  // only a few pixels. Larger atlases amortize dispatch overhead with a slightly
  // wider chunk.
  ParallelForConfig config{};
  if (totalTexels < (1u << 16)) {
    config.minChunkSize = 1;
    config.preferredChunkSize = 1;
  } else if (totalTexels < (1u << 20)) {
    config.minChunkSize = 2;
    config.preferredChunkSize = 4;
  } else {
    config.minChunkSize = 4;
    config.preferredChunkSize = 8;
  }

  // Avoid overshooting when scenes only have a handful of textures.
  if (config.preferredChunkSize > textureCount)
    config.preferredChunkSize = textureCount;

  return config;
}

void Renderer::PendingBlasBuild::releaseResources() {
  if (vertexStaging) {
    vertexStaging->release();
    vertexStaging = nullptr;
  }
  if (indexStaging) {
    indexStaging->release();
    indexStaging = nullptr;
  }
  if (scratchBuffer) {
    if (renderer) {
      renderer->recycleBlasScratchBuffer(scratchBuffer, scratchSize);
    } else {
      scratchBuffer->release();
    }
    scratchBuffer = nullptr;
    scratchSize = 0;
  }
  if (geometryArray) {
    geometryArray->release();
    geometryArray = nullptr;
  }
  if (geometryDesc) {
    geometryDesc->release();
    geometryDesc = nullptr;
  }
  if (accelDesc) {
    accelDesc->release();
    accelDesc = nullptr;
  }
  accelerationStructure = nullptr;
  commandBuffer = nullptr;
}

MTL::Buffer *Renderer::acquireBlasScratchBuffer(NS::UInteger requestedSize,
                                                NS::UInteger &allocatedSize,
                                                bool &reused) {
  allocatedSize = requestedSize;
  reused = false;
  if (!_pDevice || requestedSize == 0)
    return nullptr;

  auto it = _blasScratchPool.lower_bound(requestedSize);
  while (it != _blasScratchPool.end()) {
    auto &bucket = it->second;
    if (!bucket.empty()) {
      auto *buffer = bucket.back();
      bucket.pop_back();
      NS::UInteger bucketSize = it->first;
      if (bucket.empty())
        _blasScratchPool.erase(it);
      allocatedSize = bucketSize;
      reused = true;
      if (_blasScratchPoolAvailableBytes >= bucketSize)
        _blasScratchPoolAvailableBytes -= bucketSize;
      else
        _blasScratchPoolAvailableBytes = 0;
      _blasScratchPoolInUseBytes += bucketSize;
      ++_blasScratchPoolReusedCount;
      std::printf(
          "[BLAS] Reusing scratch buffer (%llu bytes, requested %llu). Pool "
          "in-flight: %zu bytes, available: %zu bytes. (created=%zu, reused=%zu)\n",
          static_cast<unsigned long long>(bucketSize),
          static_cast<unsigned long long>(requestedSize),
          _blasScratchPoolInUseBytes, _blasScratchPoolAvailableBytes,
          _blasScratchPoolCreatedCount, _blasScratchPoolReusedCount);
      updateBlasScratchResidencyBudget();
      return buffer;
    }
    it = _blasScratchPool.erase(it);
  }

  auto *buffer = _pDevice->newBuffer(requestedSize,
                                     MTL::ResourceStorageModePrivate);
  if (buffer) {
    allocatedSize = requestedSize;
    _blasScratchPoolInUseBytes += requestedSize;
    ++_blasScratchPoolCreatedCount;
    std::printf(
        "[BLAS] Allocated scratch buffer (%llu bytes). Pool in-flight: %zu "
        "bytes, available: %zu bytes. (created=%zu, reused=%zu)\n",
        static_cast<unsigned long long>(requestedSize),
        _blasScratchPoolInUseBytes, _blasScratchPoolAvailableBytes,
        _blasScratchPoolCreatedCount, _blasScratchPoolReusedCount);
  }
  updateBlasScratchResidencyBudget();
  return buffer;
}

void Renderer::recycleBlasScratchBuffer(MTL::Buffer *buffer, NS::UInteger size) {
  if (!buffer)
    return;

  if (size == 0) {
    buffer->release();
    return;
  }

  if (_blasScratchPoolInUseBytes >= size)
    _blasScratchPoolInUseBytes -= size;
  else
    _blasScratchPoolInUseBytes = 0;

  _blasScratchPoolAvailableBytes += size;
  _blasScratchPool[size].push_back(buffer);

  updateBlasScratchResidencyBudget();

  std::printf(
      "[BLAS] Recycled scratch buffer (%llu bytes). Pool in-flight: %zu bytes, "
      "available: %zu bytes. (created=%zu, reused=%zu)\n",
      static_cast<unsigned long long>(size), _blasScratchPoolInUseBytes,
      _blasScratchPoolAvailableBytes, _blasScratchPoolCreatedCount,
      _blasScratchPoolReusedCount);
}

void Renderer::releaseBlasScratchPool() {
  if (!_blasScratchPool.empty() || _blasScratchPoolInUseBytes != 0 ||
      _blasScratchPoolCreatedCount != 0 || _blasScratchPoolReusedCount != 0) {
    std::printf(
        "[BLAS] Releasing scratch pool (available=%zu bytes, in-flight=%zu "
        "bytes, created=%zu, reused=%zu).\n",
        _blasScratchPoolAvailableBytes, _blasScratchPoolInUseBytes,
        _blasScratchPoolCreatedCount, _blasScratchPoolReusedCount);
  }

  for (auto &entry : _blasScratchPool) {
    for (auto *buffer : entry.second) {
      if (buffer)
        buffer->release();
    }
  }

  _blasScratchPool.clear();
  _blasScratchPoolAvailableBytes = 0;

  if (_blasScratchPoolInUseBytes != 0) {
    std::printf(
        "[BLAS] Warning: scratch pool destroyed with %zu bytes still in "
        "flight.\n",
        _blasScratchPoolInUseBytes);
    _blasScratchPoolInUseBytes = 0;
  }

  updateBlasScratchResidencyBudget();
}

void Renderer::updateBlasScratchResidencyBudget() {
  _residencyBudget.setBlasScratchBytes(_blasScratchPoolAvailableBytes,
                                       _blasScratchPoolInUseBytes);
}

double Renderer::scratchMemoryMB() const {
  return _residencyBudget.scratchMemoryMB();
}

double Renderer::residencyMemoryMB() const {
  double totalMB = currentGPUMemoryMB();
  return _residencyBudget.residencyMemoryMB(totalMB);
}

struct UniformsData {
  int primitiveIndex;
  simd::float3 cameraPosition;
  simd::float2 screenSize;

  simd::float3 viewportU;
  simd::float3 viewportV;
  simd::float3 firstPixelPosition;
  simd::float3 rayDx;
  simd::float3 rayDy;

  simd::float3 randomSeed;

  uint64_t primitiveCount;
  uint64_t triangleCount;
  uint64_t frameCount = 0;
  uint64_t totalPrimitiveCount;
  uint64_t tlasNodeCount;
  uint64_t blasNodeCount;
  uint32_t maxRayDepth;
  uint32_t debugAS;
  uint32_t lightCount;
  float lightTotalWeight;
  uint32_t sampleCountTextureIndex = 0;
  uint32_t sampleImportanceTextureIndex = 0;
  uint32_t minSamplesPerPixel = 1;
  uint32_t maxSamplesPerPixel = 1;
  uint32_t textureCount = 0;
  uint32_t environmentMapEnabled = 0;
  float environmentMapIntensity = 1.0f;
  float environmentPadding0 = 0.0f;
  float environmentPadding1 = 0.0f;
};

struct TileDispatchRegion {
  uint32_t originX;
  uint32_t originY;
  uint32_t width;
  uint32_t height;
};

constexpr uint32_t kPathTraceTileWidth = 128;
constexpr uint32_t kPathTraceTileHeight = 128;
// Limit the amount of pixel/sample work recorded into a single command buffer to
// reduce the chance of long-running kernels triggering GPU timeout errors when
// high sample counts are requested. With the always-resident residency strategy
// scenes keep significantly more geometry active, so reduce the per-command
// budget to lower kernel runtimes and avoid tripping the macOS GPU watchdog.
constexpr size_t kMaxTileSampleWorkPerCommand = 128 * 128 * 4;
constexpr std::chrono::milliseconds kFrameCommandBufferWaitTimeout(4);
constexpr uint32_t kMaxMaterialTextureSlots = 64;

inline uint32_t bitm_random() {
  static uint32_t current_seed = 92407235;
  const uint32_t state = current_seed * 747796405u + 2891336453u;
  const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state);
  return (current_seed = (word >> 22u) ^ word);
}

inline float randomFloat() {
  return (float)bitm_random() / (float)std::numeric_limits<uint32_t>::max();
}

namespace {

size_t alignTo(size_t value, size_t alignment) {
  if (alignment == 0)
    return value;
  return (value + alignment - 1) & ~(alignment - 1);
}

constexpr size_t kTextureResidencyPrimitiveBudget = 1;
constexpr double kDefaultTextureResidencyMemoryCapMB = 2048.0;
constexpr size_t kMaxTextureHistoryBytes = 16ull * 1024ull * 1024ull;
constexpr float kFrustumDebugNear = 0.1f;
constexpr float kFrustumDebugFarMultiplier = 5.0f;

struct OverlayUniforms {
  simd::float4x4 viewProjection;
};

constexpr std::array<std::pair<uint32_t, uint32_t>, 12> kFrustumEdges = {
    {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4},
     {0, 4}, {1, 5}, {2, 6}, {3, 7}}};

size_t bytesPerPixel(MTL::PixelFormat format) {
  switch (format) {
  case MTL::PixelFormat::PixelFormatRGBA32Float:
    return sizeof(float) * 4;
  case MTL::PixelFormat::PixelFormatR32Float:
    return sizeof(float);
  case MTL::PixelFormat::PixelFormatRGBA16Float:
    return sizeof(uint16_t) * 4;
  case MTL::PixelFormat::PixelFormatR16Float:
    return sizeof(uint16_t);
  default:
    return 0;
  }
}

simd::float4x4 makeViewMatrix(const Camera::State &state) {
  simd::float3 forward = state.forward;
  if (simd::length_squared(forward) < 1e-6f)
    forward = {0.0f, 0.0f, -1.0f};
  forward = simd::normalize(forward);

  simd::float3 up = state.up;
  if (simd::length_squared(up) < 1e-6f)
    up = {0.0f, 1.0f, 0.0f};
  up = simd::normalize(up);

  simd::float3 right = simd::cross(forward, up);
  if (simd::length_squared(right) < 1e-6f)
    right = {1.0f, 0.0f, 0.0f};
  right = simd::normalize(right);
  up = simd::normalize(simd::cross(right, forward));

  simd::float4x4 view;
  view.columns[0] = {right.x, up.x, forward.x, 0.0f};
  view.columns[1] = {right.y, up.y, forward.y, 0.0f};
  view.columns[2] = {right.z, up.z, forward.z, 0.0f};
  view.columns[3] = {-simd::dot(right, state.position),
                     -simd::dot(up, state.position),
                     -simd::dot(forward, state.position), 1.0f};
  return view;
}

simd::float4x4 makePerspectiveMatrix(float verticalFovDegrees, float aspect,
                                     float nearZ, float farZ) {
  float fovRad = verticalFovDegrees * static_cast<float>(M_PI) / 180.0f;
  float yScale = 1.0f / std::tan(fovRad * 0.5f);
  float xScale = yScale / std::max(aspect, 1e-6f);
  float zRange = std::max(farZ - nearZ, 1e-6f);
  float zScale = farZ / zRange;
  float wz = -nearZ * zScale;

  simd::float4x4 proj;
  proj.columns[0] = {xScale, 0.0f, 0.0f, 0.0f};
  proj.columns[1] = {0.0f, yScale, 0.0f, 0.0f};
  proj.columns[2] = {0.0f, 0.0f, zScale, 1.0f};
  proj.columns[3] = {0.0f, 0.0f, wz, 0.0f};
  return proj;
}

float luminance(const simd::float3 &c) {
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

template <typename Func>
void parallelChunkedAsync(size_t start, size_t end, Func &&func) {
  if (end <= start)
    return;

  auto work = std::forward<Func>(func);
  const size_t total = end - start;
  const unsigned int threadCount =
      std::max(1u, std::thread::hardware_concurrency());
  const size_t smallWorkThreshold =
      static_cast<size_t>(threadCount) * static_cast<size_t>(64);

  if (total <= smallWorkThreshold) {
    work(start, end);
    return;
  }

  const size_t chunkSize = std::max<size_t>(
      1, (total + static_cast<size_t>(threadCount) - 1) /
             static_cast<size_t>(threadCount));

  struct Context {
    size_t start;
    size_t end;
    size_t chunkSize;
    decltype(work) *func;
  } ctx{start, end, chunkSize, &work};

  const size_t chunkCount = (total + chunkSize - 1) / chunkSize;
  dispatch_queue_t queue =
      dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
  dispatch_apply_f(chunkCount, queue, &ctx, [](void *rawCtx, size_t chunkIndex) {
    auto &ctx = *static_cast<Context *>(rawCtx);
    size_t chunkBegin = ctx.start + chunkIndex * ctx.chunkSize;
    size_t chunkEnd = std::min(chunkBegin + ctx.chunkSize, ctx.end);
    if (chunkBegin < ctx.end)
      (*ctx.func)(chunkBegin, chunkEnd);
  });
}

float primitiveArea(const Primitive &p) {
  switch (p.type) {
  case PrimitiveType::Sphere: {
    float r = p.sphere.radius;
    return 4.0f * static_cast<float>(M_PI) * r * r;
  }
  case PrimitiveType::Triangle: {
    simd::float3 e1 = p.triangle.v1 - p.triangle.v0;
    simd::float3 e2 = p.triangle.v2 - p.triangle.v0;
    return 0.5f * simd::length(simd::cross(e1, e2));
  }
  case PrimitiveType::Rectangle: {
    simd::float3 e1 = p.rectangle.u;
    simd::float3 e2 = p.rectangle.v;
    return 4.0f * simd::length(simd::cross(e1, e2));
  }
  }
  return 0.0f;
}

float primitiveImportance(const Primitive &p) {
  float area = std::max(primitiveArea(p), 1e-4f);
  const Material &m = p.material;
  float emissive = m.emissionPower * luminance(m.emissionColor);
  float diffuse = luminance(m.diffuseColor) * m.opacity;
  float specular = luminance(m.specularColor) * m.opacity;
  float transmission = (1.0f - m.opacity) * luminance(m.transmissionColor);
  return area * (emissive + diffuse + specular + transmission);
}

float sanitizeSortValue(float value) {
  if (std::isnan(value))
    return -std::numeric_limits<float>::max();

  const float finiteMax = std::numeric_limits<float>::max();
  if (value >= finiteMax)
    return finiteMax;
  if (value <= -finiteMax)
    return -finiteMax;

  return value;
}

float boundingSurfaceArea(const simd::float3 &bmin, const simd::float3 &bmax) {
  simd::float3 d = bmax - bmin;
  return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

float primitiveAxisValueLocal(const Primitive &p, int axis) {
  switch (p.type) {
  case PrimitiveType::Sphere:
    return p.sphere.center[axis];
  case PrimitiveType::Rectangle:
    return p.rectangle.center[axis];
  case PrimitiveType::Triangle:
    return (p.triangle.v0[axis] + p.triangle.v1[axis] + p.triangle.v2[axis]) /
           3.0f;
  }
  return 0.0f;
}

simd::float3 primitiveCentroidLocal(const Primitive &p) {
  switch (p.type) {
  case PrimitiveType::Sphere:
    return p.sphere.center;
  case PrimitiveType::Rectangle:
    return p.rectangle.center;
  case PrimitiveType::Triangle:
    return (p.triangle.v0 + p.triangle.v1 + p.triangle.v2) / 3.0f;
  }
  return simd::float3(0.0f);
}

void primitiveBoundsLocal(const Primitive &p, simd::float3 &pMin,
                          simd::float3 &pMax) {
  if (p.type == PrimitiveType::Sphere) {
    float r = p.sphere.radius;
    pMin = p.sphere.center - r;
    pMax = p.sphere.center + r;
  } else if (p.type == PrimitiveType::Rectangle) {
    simd::float3 c = p.rectangle.center;
    simd::float3 e1 = p.rectangle.u;
    simd::float3 e2 = p.rectangle.v;
    simd::float3 c1 = c - e1 - e2;
    simd::float3 c2 = c - e1 + e2;
    simd::float3 c3 = c + e1 - e2;
    simd::float3 c4 = c + e1 + e2;
    pMin = simd::min(simd::min(c1, c2), simd::min(c3, c4));
    pMax = simd::max(simd::max(c1, c2), simd::max(c3, c4));
  } else {
    const auto &t = p.triangle;
    pMin = simd::min(t.v0, simd::min(t.v1, t.v2));
    pMax = simd::max(t.v0, simd::max(t.v1, t.v2));
  }
}

void markBufferModified(MTL::Buffer *buffer, NS::Range range) {
  if (!buffer)
    return;

  if (buffer->storageMode() == MTL::StorageModeManaged)
    buffer->didModifyRange(range);
}

static bool cameraStatesDiffer(const Camera::State &a, const Camera::State &b,
                               float epsilon = 1e-3f) {
  if (simd::length(a.position - b.position) > epsilon)
    return true;
  if (simd::length(a.forward - b.forward) > epsilon)
    return true;
  if (simd::length(a.up - b.up) > epsilon)
    return true;
  if (std::abs(a.verticalFov - b.verticalFov) > epsilon)
    return true;
  if (std::abs(a.focalLength - b.focalLength) > epsilon)
    return true;
  return false;
}

static Camera::State makeCameraState(const simd::float3 &position,
                                     const simd::float3 &lookAt,
                                     float verticalFov, float focalLength,
                                     const simd::float3 &fallbackForward) {
  Camera::State state{};
  state.position = position;
  simd::float3 direction = lookAt - position;
  if (simd::length_squared(direction) > 1e-6f) {
    state.forward = simd::normalize(direction);
  } else {
    state.forward = fallbackForward;
  }
  state.up = {0, 1, 0};
  state.verticalFov = verticalFov;
  state.focalLength = focalLength;
  return state;
}

int buildBVHRecursive(const std::vector<Primitive> &primitives,
                      std::vector<int> &primitiveIndices,
                      std::vector<BVHNode> &nodes, size_t start, size_t end) {
  BVHNode node;
  simd::float3 bMin(std::numeric_limits<float>::max());
  simd::float3 bMax(-std::numeric_limits<float>::max());

  for (size_t i = start; i < end; ++i) {
    const Primitive &p = primitives[primitiveIndices[i]];
    simd::float3 pMin, pMax;
    primitiveBoundsLocal(p, pMin, pMax);
    bMin = simd::min(bMin, pMin);
    bMax = simd::max(bMax, pMax);
  }

  node.boundsMin = bMin;
  node.boundsMax = bMax;
  node.leftFirst = static_cast<int>(start);
  node.count = static_cast<int>(end - start);

  int nodeIndex = static_cast<int>(nodes.size());
  nodes.push_back(node);

  size_t range = end - start;
  if (range <= 8)
    return nodeIndex;

  const float parentArea = boundingSurfaceArea(bMin, bMax);
  if (parentArea <= 0.0f)
    return nodeIndex;

  std::vector<simd::float3> primitiveMins(range);
  std::vector<simd::float3> primitiveMaxs(range);
  std::vector<simd::float3> primitiveCentroids(range);
  std::array<float, 3> centroidMin = {
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::max()};
  std::array<float, 3> centroidMax = {
      -std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::max()};

  for (size_t i = start; i < end; ++i) {
    const Primitive &p = primitives[primitiveIndices[i]];
    simd::float3 pMin, pMax;
    primitiveBoundsLocal(p, pMin, pMax);
    primitiveMins[i - start] = pMin;
    primitiveMaxs[i - start] = pMax;

    simd::float3 centroid = primitiveCentroidLocal(p);
    primitiveCentroids[i - start] = centroid;
    centroidMin[0] = std::min(centroidMin[0], centroid.x);
    centroidMin[1] = std::min(centroidMin[1], centroid.y);
    centroidMin[2] = std::min(centroidMin[2], centroid.z);
    centroidMax[0] = std::max(centroidMax[0], centroid.x);
    centroidMax[1] = std::max(centroidMax[1], centroid.y);
    centroidMax[2] = std::max(centroidMax[2], centroid.z);
  }

  float bestCost = std::numeric_limits<float>::max();
  int bestAxis = -1;
  size_t bestLeftCount = range / 2;

  constexpr int kBinCount = 12;
  std::array<int, kBinCount> binCount{};
  std::array<simd::float3, kBinCount> binMin;
  std::array<simd::float3, kBinCount> binMax;
  std::array<simd::float3, kBinCount> prefixMin;
  std::array<simd::float3, kBinCount> prefixMax;
  std::array<int, kBinCount> prefixCount{};
  std::array<simd::float3, kBinCount> suffixMin;
  std::array<simd::float3, kBinCount> suffixMax;
  std::array<int, kBinCount> suffixCount{};

  for (int axis = 0; axis < 3; ++axis) {
    float axisMin = centroidMin[axis];
    float axisMax = centroidMax[axis];
    float axisRange = axisMax - axisMin;
    if (axisRange <= 1e-6f)
      continue;

    for (int i = 0; i < kBinCount; ++i) {
      binCount[i] = 0;
      binMin[i] = simd::float3(std::numeric_limits<float>::max());
      binMax[i] = simd::float3(-std::numeric_limits<float>::max());
    }

    float invRange = 1.0f / axisRange;
    for (size_t i = 0; i < range; ++i) {
      const simd::float3 &pMin = primitiveMins[i];
      const simd::float3 &pMax = primitiveMaxs[i];
      float centroid = primitiveCentroids[i][axis];
      int bin = static_cast<int>((centroid - axisMin) * invRange * kBinCount);
      bin = std::max(0, std::min(kBinCount - 1, bin));
      ++binCount[bin];
      binMin[bin] = simd::min(binMin[bin], pMin);
      binMax[bin] = simd::max(binMax[bin], pMax);
    }

    simd::float3 runningMin(std::numeric_limits<float>::max());
    simd::float3 runningMax(-std::numeric_limits<float>::max());
    int runningCount = 0;
    for (int i = 0; i < kBinCount; ++i) {
      if (binCount[i] > 0) {
        runningMin = simd::min(runningMin, binMin[i]);
        runningMax = simd::max(runningMax, binMax[i]);
      }
      runningCount += binCount[i];
      prefixMin[i] = runningMin;
      prefixMax[i] = runningMax;
      prefixCount[i] = runningCount;
    }

    runningMin = simd::float3(std::numeric_limits<float>::max());
    runningMax = simd::float3(-std::numeric_limits<float>::max());
    runningCount = 0;
    for (int i = kBinCount - 1; i >= 0; --i) {
      if (binCount[i] > 0) {
        runningMin = simd::min(runningMin, binMin[i]);
        runningMax = simd::max(runningMax, binMax[i]);
      }
      runningCount += binCount[i];
      suffixMin[i] = runningMin;
      suffixMax[i] = runningMax;
      suffixCount[i] = runningCount;
    }

    for (int i = 0; i < kBinCount - 1; ++i) {
      int leftCount = prefixCount[i];
      int rightCount = suffixCount[i + 1];
      if (leftCount == 0 || rightCount == 0)
        continue;

      float saLeft = boundingSurfaceArea(prefixMin[i], prefixMax[i]);
      float saRight = boundingSurfaceArea(suffixMin[i + 1], suffixMax[i + 1]);
      float cost = 0.125f + (saLeft / parentArea) * leftCount +
                   (saRight / parentArea) * rightCount;

      if (cost < bestCost) {
        bestCost = cost;
        bestAxis = axis;
        bestLeftCount = static_cast<size_t>(leftCount);
      }
    }
  }

  if (bestAxis == -1 || bestLeftCount == 0 || bestLeftCount >= range)
    return nodeIndex;

  auto begin = primitiveIndices.begin() + start;
  auto endIt = primitiveIndices.begin() + end;
  std::stable_sort(begin, endIt, [&](int a, int b) {
    return primitiveAxisValueLocal(primitives[a], bestAxis) <
           primitiveAxisValueLocal(primitives[b], bestAxis);
  });

  size_t bestSplit = start + bestLeftCount;

  int leftChild = buildBVHRecursive(primitives, primitiveIndices, nodes, start,
                                    bestSplit);
  int rightChild =
      buildBVHRecursive(primitives, primitiveIndices, nodes, bestSplit, end);

  nodes[nodeIndex].leftFirst = leftChild;
  nodes[nodeIndex].count = -rightChild;
  return nodeIndex;
}

struct BufferCountInfo {
  size_t current = 0;
  size_t previous = 0;
  size_t zeroStart = 0;
  size_t zeroCount = 0;
  size_t touched = 0;
};

BufferCountInfo prepareBufferCounts(size_t current, size_t previous,
                                    size_t capacity) {
  BufferCountInfo info{};
  if (capacity == 0)
    return info;

  info.current = std::min(current, capacity);
  info.previous = std::min(previous, capacity);
  info.zeroStart = std::min(info.current, capacity);
  if (info.previous > info.zeroStart)
    info.zeroCount = info.previous - info.zeroStart;

  info.touched = std::max({info.current, info.previous, size_t(1)});
  info.touched = std::min(info.touched, capacity);
  return info;
}

} // namespace

void Renderer::ensureBufferCapacity(MTL::Buffer *&buffer, size_t requiredBytes,
                                    size_t &currentCapacity,
                                    bool allowShrink,
                                    MTL::ResourceOptions storageMode,
                                    const char *label,
                                    const char *resizeContext) {
  if (requiredBytes == 0)
    requiredBytes = 1;

  size_t desiredCapacity = requiredBytes;
  size_t originalCapacity = currentCapacity;
  bool hadBuffer = buffer != nullptr;
  bool shrinking = false;
  bool growing = false;

  if (!allowShrink) {
    if (buffer && requiredBytes <= currentCapacity)
      return;
    desiredCapacity = std::max(requiredBytes, currentCapacity);
    if (buffer && requiredBytes > currentCapacity)
      growing = true;
  } else if (buffer) {
    size_t shrinkThreshold = currentCapacity / 2;
    if (requiredBytes <= currentCapacity && requiredBytes > shrinkThreshold)
      return;
    if (requiredBytes <= shrinkThreshold) {
      shrinking = requiredBytes < currentCapacity;
    } else if (requiredBytes > currentCapacity) {
      desiredCapacity = std::max(requiredBytes, currentCapacity * 2);
      growing = true;
    }
  }

  if (!hadBuffer)
    growing = true;

  if (buffer) {
    buffer->release();
    buffer = nullptr;
    currentCapacity = 0;
  }

  if (allowShrink && desiredCapacity < requiredBytes)
    desiredCapacity = requiredBytes;

  desiredCapacity = std::max(desiredCapacity, size_t(1));
  buffer = _pDevice->newBuffer(desiredCapacity, storageMode);
  currentCapacity = buffer ? buffer->length() : 0;

  const char *name = label ? label : "UnnamedBuffer";
  const char *context = resizeContext
                            ? resizeContext
                            : (allowShrink ? "shrink-policy" : "capacity");
  if (!buffer) {
    std::printf(
        "[Renderer][Buffer] Failed to allocate %s (requested=%zu bytes, context=%s).\n",
        name, desiredCapacity, context);
    return;
  }

  const char *action = hadBuffer
                           ? (shrinking ? "shrunk" : (growing ? "grown" : "rebound"))
                           : "allocated";
  std::printf(
      "[Renderer][Buffer] %s %s from %zu to %zu bytes (required=%zu, allowShrink=%s, context=%s, activeRatio=%.3f).\n",
      name, action, originalCapacity, currentCapacity, requiredBytes,
      allowShrink ? "true" : "false", context, _lastActivePrimitiveRatio);
}

bool Renderer::isInView(const BoundingSphere &b) {
  simd::float3 toCenter = b.center - Camera::position;
  float dist = simd::length(toCenter);
  if (dist < 1e-5f)
    return true;

  simd::float3 forward = simd::normalize(Camera::forward);
  float forwardLenSq = simd::length_squared(forward);
  if (forwardLenSq < 1e-6f)
    return true;

  simd::float3 dir = toCenter / dist;
  float forwardDot = simd::dot(dir, forward);
  if (forwardDot <= 0.0f)
    return false;

  simd::float3 up = Camera::up;
  float upLenSq = simd::length_squared(up);
  if (upLenSq < 1e-6f) {
    up = {0.0f, 1.0f, 0.0f};
    upLenSq = 1.0f;
  }
  up /= std::sqrt(upLenSq);

  simd::float3 right = simd::cross(forward, up);
  float rightLenSq = simd::length_squared(right);
  if (rightLenSq < 1e-6f) {
    right = {1.0f, 0.0f, 0.0f};
    rightLenSq = 1.0f;
  }
  right /= std::sqrt(rightLenSq);

  up = simd::normalize(simd::cross(right, forward));

  float verticalHalfFov =
      Camera::verticalFov * static_cast<float>(M_PI) / 180.0f * 0.5f;
  if (verticalHalfFov <= 0.0f)
    verticalHalfFov = 1e-3f;

  float aspect = Camera::screenSize.y > 0.0f
                     ? Camera::screenSize.x / Camera::screenSize.y
                     : 1.0f;
  float horizontalHalfFov = std::atan(std::tan(verticalHalfFov) * aspect);

  float radiusAngle = asinf(std::min(b.radius / dist, 1.0f));

  float horizontalAngle =
      std::atan2(std::fabs(simd::dot(dir, right)), forwardDot);
  float verticalAngle =
      std::atan2(std::fabs(simd::dot(dir, up)), forwardDot);

  return horizontalAngle <= horizontalHalfFov + radiusAngle &&
         verticalAngle <= verticalHalfFov + radiusAngle;
}

Renderer::Renderer(MTL::Device *pDevice)
    : _pDevice(pDevice->retain()), _pScene(new Scene()),
      _pendingBlasEvictions(
          [](MTL::CommandBuffer *command) {
            if (command)
              command->retain();
          },
          [](MTL::CommandBuffer *command) {
            if (command)
              command->release();
          }) {
  _pCommandQueue = _pDevice->newCommandQueue();
  _tlasHeap.initialize(_pDevice);
  _dummyBlasResources.initialize(_pDevice);

  Camera::reset();
  _primaryCameraState = Camera::captureState();
  _observerCameraState = _primaryCameraState;

  updateVisibleScene();
  buildShaders();
  buildTextures();
  rebuildEnvironmentTexture();

  recalculateViewport();
  initializeBenchmarking();
}

Renderer::~Renderer() {
  processPendingCapturedFrames();
  std::chrono::steady_clock::time_point frameWaitSnapshot;
  waitForPendingFrameCommands(std::chrono::milliseconds::max(),
                              &frameWaitSnapshot);

  _pendingBlasEvictions.clear();
  assert(_pendingBlasEvictions.empty());

  if (_pSphereBuffer)
    _pSphereBuffer->release();
  if (_pSphereMaterialBuffer)
    _pSphereMaterialBuffer->release();
  if (_pTriangleVertexBuffer)
    _pTriangleVertexBuffer->release();
  if (_pTriangleIndexBuffer)
    _pTriangleIndexBuffer->release();
  if (_pUniformsBuffer)
    _pUniformsBuffer->release();
  if (_pBVHBuffer)
    _pBVHBuffer->release();
  if (_pPrimitiveIndexBuffer)
    _pPrimitiveIndexBuffer->release();
  if (_pTLASBuffer)
    _pTLASBuffer->release();
  if (_pActiveBuffer)
    _pActiveBuffer->release();
  if (_pPrimitiveRemapBuffer)
    _pPrimitiveRemapBuffer->release();
  flushRayHitCopy();
  if (_pPrimitiveHitBufferGPU)
    _pPrimitiveHitBufferGPU->release();
  if (_pPrimitiveHitReadback)
    _pPrimitiveHitReadback->release();
  if (_pLightIndexBuffer)
    _pLightIndexBuffer->release();
  if (_pLightCdfBuffer)
    _pLightCdfBuffer->release();
  if (_pInstanceBuffer)
    _pInstanceBuffer->release();
  if (_pGeometryHandleBuffer)
    _pGeometryHandleBuffer->release();
  if (_pTlasScratchBuffer)
    _pTlasScratchBuffer->release();
  if (_pTlasBuildEvent)
    _pTlasBuildEvent->release();
  if (_pTlasDescriptorStaging)
    _pTlasDescriptorStaging->release();
  _pTlasDescriptorStaging = nullptr;
  _tlasDescriptorStagingCapacity = 0;
  _pTlasScratchBuffer = nullptr;
  _tlasScratchCapacity = 0;
  _tlasScratchTracker.reset();
  updateTlasScratchResidentBytes(0);
  _pTlasBuildEvent = nullptr;
  _tlasBuildEventValue = 0;
  _tlasCompletedEventValue.store(0, std::memory_order_relaxed);
  if (_pFrustumVertexBuffer)
    _pFrustumVertexBuffer->release();
  releaseEnvironmentTexture();
  if (_environmentSampler)
    _environmentSampler->release();
  _environmentSampler = nullptr;
  clearMaterialTextures();

  _cachedInstanceDescriptors.clear();
  _cachedInstancedAccelerationStructures.clear();
  _pTlasInstanceDescriptorBuffer = nullptr;
  _pTlasStructure = nullptr;
  _pDummyBlas = nullptr;

  _tlasHeap.destroy();
  _dummyBlasResources.destroy();

  for (auto &slot : _accumulationSlots)
    releaseTextureSlot(slot);
  releaseTextureSlot(_sampleCountSlot);
  releaseTextureSlot(_sampleImportanceSlot);
  releaseTextureSlot(_albedoSlot);
  releaseTextureSlot(_normalSlot);

  if (_pTextureClearBuffer)
    _pTextureClearBuffer->release();

  if (_pPathTracePSO)
    _pPathTracePSO->release();
  if (_pAdaptiveSamplingPSO)
    _pAdaptiveSamplingPSO->release();
  if (_pOverlayPSO)
    _pOverlayPSO->release();

  for (auto &resident : _residentObjectGpuResources) {
    resident.clearPendingCommand();
    resident.resources.destroy();
  }
  _residentObjectGpuResources.clear();

  releaseBlasScratchPool();

  if (_pPSO)
    _pPSO->release();
  if (_pCommandQueue)
    _pCommandQueue->release();
  if (_pDevice)
    _pDevice->release();

  if (_benchmarkStream.is_open())
    _benchmarkStream.close();

  delete _pScene;
}

void Renderer::setBenchmarkMode(bool enabled) {
  if (enabled == _benchmarkEnabled)
    return;

  _benchmarkEnabled = enabled;
  if (_benchmarkEnabled) {
    _benchmarkFrameCounter = 0;
    _benchmarkHeaderWritten = false;
    _pendingBenchmarkSamples.clear();
    _benchmarkStartTime = std::chrono::steady_clock::now();
    ensureBenchmarkStream();
    resetProbabilisticResidencyState();
    if (_benchmarkStream.is_open()) {
      printf("Benchmark logging enabled: %s\n",
             _benchmarkFilePath.empty() ? "<memory>"
                                        : _benchmarkFilePath.c_str());
    }
  } else {
    if (_benchmarkStream.is_open())
      _benchmarkStream.close();
    _pendingBenchmarkSamples.clear();
  }
}

void Renderer::resetProbabilisticResidencyState() {
  std::fill(_primitiveHitScores.begin(), _primitiveHitScores.end(), 0.0f);
  std::fill(_primitiveHitAlpha.begin(), _primitiveHitAlpha.end(), 1.0f);
  std::fill(_primitiveHitBeta.begin(), _primitiveHitBeta.end(), 1.0f);
  std::fill(_primitiveHitProbability.begin(), _primitiveHitProbability.end(),
            0.5f);
  std::fill(_primitiveHitVariance.begin(), _primitiveHitVariance.end(),
            1.0f / 12.0f);
  std::fill(_primitivePosteriorMass.begin(), _primitivePosteriorMass.end(),
            2.0f);
  std::fill(_primitiveExplorationScore.begin(),
            _primitiveExplorationScore.end(), 0.0f);

  std::fill(_objectHitAlpha.begin(), _objectHitAlpha.end(), 1.0f);
  std::fill(_objectHitBeta.begin(), _objectHitBeta.end(), 1.0f);
  std::fill(_objectHitProbability.begin(), _objectHitProbability.end(), 0.5f);
  std::fill(_objectHitVariance.begin(), _objectHitVariance.end(), 1.0f / 12.0f);
  std::fill(_objectPosteriorMass.begin(), _objectPosteriorMass.end(), 2.0f);
  std::fill(_objectExplorationScore.begin(), _objectExplorationScore.end(),
            0.0f);

  std::fill(_primitiveCooldown.begin(), _primitiveCooldown.end(), 0u);
  std::fill(_objectCooldown.begin(), _objectCooldown.end(), 0u);
  std::fill(_objectLastToggleFrame.begin(), _objectLastToggleFrame.end(), 0u);
  _rayHitRebuildCooldown = 0;
}

void Renderer::setFrameCaptureEnabled(bool enabled) {
  if (_frameCaptureEnabled == enabled)
    return;

  _frameCaptureEnabled = enabled;
  if (_frameCaptureEnabled)
    ensureFrameCaptureDirectory();
  else if (_captureOutputsPending.load(std::memory_order_acquire))
    processPendingCapturedFrames();
}

void Renderer::setFrameCaptureInterval(size_t interval) {
  if (interval == 0)
    interval = 1;
  _frameCaptureInterval = interval;
}

void Renderer::initializeBenchmarking() {
  const char *primaryEnv = std::getenv("METALPT_BENCH");
  const char *legacyEnv = std::getenv("METALAPT_BENCH");
  const char *env = primaryEnv ? primaryEnv : legacyEnv;
  if (!env)
    return;

  if (!primaryEnv && legacyEnv) {
    printf("METALAPT_BENCH detected; please update to METALPT_BENCH for future runs.\n");
  }

  std::string value(env);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  bool enable = value == "1" || value == "true" || value == "yes" || value == "on";
  setBenchmarkMode(enable);
}

void Renderer::ensureBenchmarkStream() {
  if (!_benchmarkEnabled)
    return;
  if (_benchmarkStream.is_open())
    return;

  std::filesystem::path benchmarksDir = std::filesystem::current_path() / "Benchmarks";
  std::error_code ec;
  std::filesystem::create_directories(benchmarksDir, ec);

  auto now = std::chrono::system_clock::now();
  std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
  std::tm timeInfo{};
#if defined(_WIN32)
  localtime_s(&timeInfo, &nowTime);
#else
  localtime_r(&nowTime, &timeInfo);
#endif
  std::ostringstream nameBuilder;
  nameBuilder << "metrics_" << std::put_time(&timeInfo, "%Y%m%d_%H%M%S") << ".csv";
  _benchmarkFilePath = (benchmarksDir / nameBuilder.str()).string();

  _benchmarkStream.open(_benchmarkFilePath, std::ios::out | std::ios::trunc);
  if (_benchmarkStream.is_open()) {
    writeBenchmarkHeader();
  } else {
    printf("Failed to open benchmark log '%s'\n", _benchmarkFilePath.c_str());
  }
}

void Renderer::writeBenchmarkHeader() {
  if (!_benchmarkStream.is_open())
    return;
  if (_benchmarkHeaderWritten)
    return;

  _benchmarkStream
      << "frame,wall_seconds,cpu_ms,gpu_ms,rays_per_second,rays_cast,strategy,"
         "strategy_id,delta_time_seconds,min_samples_per_pixel,max_samples_per_pixel,"
         "primitive_activations,primitive_deactivations,object_activations,"
         "object_deactivations,active_primitives,resident_primitives,total_primitives,"
         "active_triangles,resident_triangles,total_triangles,active_nodes,"
         "resident_nodes,total_nodes,active_objects,resident_objects,"
         "avg_hit_probability,p95_hit_probability,probability_threshold,"
         "probability_target_fraction,probability_visible_floor,"
         "probability_target_primitives,"
         "probability_initial_desired_primitives,"
         "probability_final_desired_primitives,probability_trimmed_primitives,"
         "probability_budget_hit,primitive_probabilities,object_probabilities,"
         "probabilistic_toggles,"
         "gpu_memory_mb,scratch_memory_mb,"
         "residency_memory_mb,texture_memory_cap_mb,"
         "over_memory_cap,residency_compacted,"
         "accumulation_reset,ray_hit_decay,state_cooldown_frames,lod_toggle_budget,"
         "energy_toggle_budget,screen_toggle_budget,rayhit_toggle_budget,"
         "rayhit_target_fraction,rayhit_min_active,rayhit_rebuild_cooldown,"
         "lod_enter_distance,lod_exit_distance,lod_enter_view_margin,"
         "lod_exit_view_margin,energy_target_fraction,"
         "energy_min_active,energy_visibility_boost,screen_target_fraction,"
         "screen_min_pixels,screen_min_active,environment_target_fraction,"
         "environment_escape_threshold,environment_min_active,environment_toggle_budget,"
         "environment_depth_weights,environment_depth_radii";
  _benchmarkStream << '\n';
  _benchmarkHeaderWritten = true;
}

static std::string formatFixed(double value, int precision) {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(precision) << value;
  return ss.str();
}

static std::string formatFloatList(const std::vector<float> &values,
                                   int precision = 3) {
  if (values.empty())
    return "";

  std::ostringstream ss;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0)
      ss << '|';
    ss << formatFixed(values[i], precision);
  }
  return ss.str();
}

static void appendCsvEscaped(std::ostringstream &ss, const std::string &value) {
  ss << '"';
  for (char c : value) {
    if (c == '"')
      ss << '"';
    ss << c;
  }
  ss << '"';
}

void Renderer::writeBenchmarkRow(const BenchmarkSample &sample) {
  if (!_benchmarkStream.is_open())
    return;

  auto boolToInt = [](bool v) { return v ? 1 : 0; };

  std::ostringstream row;
  row << sample.frameIndex << ',' << formatFixed(sample.wallSeconds, 6) << ','
      << formatFixed(sample.cpuTimeSeconds * 1000.0, 3) << ','
      << formatFixed(sample.gpuTimeSeconds * 1000.0, 3) << ','
      << formatFixed(sample.raysPerSecond, 2) << ',' << sample.rayCount << ',';
  appendCsvEscaped(row, sample.strategyName);
  row << ',' << static_cast<int>(sample.strategy)
      << ',' << formatFixed(sample.deltaTimeSeconds, 6) << ','
      << sample.minSamplesPerPixel << ',' << sample.maxSamplesPerPixel << ','
      << sample.primitiveActivations << ',' << sample.primitiveDeactivations << ','
      << sample.objectActivations << ',' << sample.objectDeactivations << ','
      << sample.activePrimitiveCount << ',' << sample.residentPrimitiveCount << ','
      << sample.totalPrimitiveCount << ',' << sample.activeTriangleCount << ','
      << sample.residentTriangleCount << ',' << sample.totalTriangleCount << ','
      << sample.activeNodeCount << ',' << sample.residentNodeCount << ','
      << sample.totalNodeCount << ',' << sample.activeObjectCount << ','
      << sample.residentObjectCount << ','
      << formatFixed(sample.avgHitProbability, 6) << ','
      << formatFixed(sample.p95HitProbability, 6) << ','
      << formatFixed(sample.probabilityThreshold, 3) << ','
      << formatFixed(sample.probabilityTargetFraction, 3) << ','
      << formatFixed(sample.probabilityVisibleFloor, 3) << ','
      << sample.probabilityTargetPrimitives << ','
      << sample.probabilityInitialDesiredPrimitives << ','
      << sample.probabilityFinalDesiredPrimitives << ','
      << sample.probabilityTrimmedPrimitives << ','
      << boolToInt(sample.probabilityBudgetHit) << ',';
  appendCsvEscaped(row, sample.primitiveProbabilities);
  row << ',';
  appendCsvEscaped(row, sample.objectProbabilities);
  row << ','
      << sample.probabilisticToggles << ','
      << formatFixed(sample.gpuMemoryMB, 3) << ','
      << formatFixed(sample.scratchMemoryMB, 3) << ','
      << formatFixed(sample.residencyMemoryMB, 3) << ','
      << formatFixed(sample.textureMemoryCapMB, 3) << ','
      << boolToInt(sample.overMemoryCap) << ','
      << boolToInt(sample.residentCompacted) << ','
      << boolToInt(sample.accumulationReset) << ','
      << formatFixed(_residencyConfig.rayHitDecay, 3) << ','
      << _residencyConfig.stateCooldownFrames << ','
      << _residencyConfig.lodMaxTogglesPerFrame << ','
      << _residencyConfig.energyMaxTogglesPerFrame << ','
      << _residencyConfig.screenFootprintMaxTogglesPerFrame << ','
      << _residencyConfig.rayHitMaxTogglesPerFrame << ','
      << formatFixed(_residencyConfig.rayHitTargetFraction, 3) << ','
      << _residencyConfig.rayHitMinActivePrimitives << ','
      << _residencyConfig.rayHitRebuildCooldownFrames << ','
      << formatFixed(_residencyConfig.lodEnterDistance, 3) << ','
      << formatFixed(_residencyConfig.lodExitDistance, 3) << ','
      << formatFixed(_residencyConfig.lodEnterViewDegrees, 3) << ','
      << formatFixed(_residencyConfig.lodExitViewDegrees, 3) << ','
      << formatFixed(_residencyConfig.energyTargetFraction, 3) << ','
      << _residencyConfig.energyMinActivePrimitives << ','
      << formatFixed(_residencyConfig.energyVisibilityBoost, 3) << ','
      << formatFixed(_residencyConfig.screenFootprintTargetFraction, 3) << ','
      << formatFixed(_residencyConfig.screenFootprintMinPixelCoverage, 3) << ','
      << _residencyConfig.screenFootprintMinActivePrimitives << ','
      << formatFixed(sample.environmentTargetActiveFraction, 3) << ','
      << formatFixed(sample.environmentEscapeThreshold, 3) << ','
      << _residencyConfig.environmentMinActivePrimitives << ','
      << _residencyConfig.environmentMaxTogglesPerFrame << ',';
  appendCsvEscaped(row, sample.environmentDepthWeights);
  row << ',';
  appendCsvEscaped(row, sample.environmentDepthRadii);

  _benchmarkStream << row.str() << '\n';
  _benchmarkStream.flush();
}

std::string Renderer::residencyStrategyName(ResidencyStrategy strategy) const {
  switch (strategy) {
  case ResidencyStrategy::EnergyImportance:
    return "Energy importance";
  case ResidencyStrategy::RayHitBudget:
    return "Ray-hit budget";
  case ResidencyStrategy::ScreenSpaceFootprint:
    return "Screen-space footprint";
  case ResidencyStrategy::Probabilistic:
    return "Probabilistic";
  case ResidencyStrategy::UnifiedScore:
    return "Unified score";
  case ResidencyStrategy::EnvironmentHit:
    return "Environment hit";
  case ResidencyStrategy::AlwaysResident:
    return "Always resident";
  case ResidencyStrategy::DistanceLOD:
  default:
    return "Distance-based LOD";
  }
}

void Renderer::ensureFrameCaptureDirectory() {
  if (_frameCaptureDirectoryInitialized)
    return;

  std::filesystem::path benchmarksDir =
      std::filesystem::current_path() / "Benchmarks";
  std::filesystem::path framesDir = benchmarksDir / "frames";

  std::error_code ec;
  std::filesystem::create_directories(framesDir, ec);
  if (ec) {
    std::error_code fallbackError;
    std::filesystem::create_directories(benchmarksDir, fallbackError);
    if (fallbackError) {
      std::printf(
          "Failed to initialize frame capture directory '%s': %s\n",
          framesDir.string().c_str(), ec.message().c_str());
      _frameCaptureDirectory.clear();
      _frameCaptureDirectoryInitialized = true;
      return;
    }
    _frameCaptureDirectory = benchmarksDir.string();
  } else {
    _frameCaptureDirectory = framesDir.string();
  }

  _frameCaptureDirectoryInitialized = true;
}

std::shared_ptr<Renderer::FrameCaptureRequest>
Renderer::encodeFrameCapture(MTL::Texture *colorTexture,
                             MTL::Texture *albedoTexture,
                             MTL::Texture *normalTexture, uint64_t frameIndex,
                             MTL::CommandBuffer *cmd,
                             MTL::BlitCommandEncoder *&blit) {
  if (!colorTexture || !cmd)
    return nullptr;

  ensureFrameCaptureDirectory();
  if (_frameCaptureDirectory.empty())
    return nullptr;

  NS::UInteger width = colorTexture->width();
  NS::UInteger height = colorTexture->height();
  if (width == 0 || height == 0)
    return nullptr;

  size_t pixelBytes = bytesPerPixel(colorTexture->pixelFormat());
  if (pixelBytes == 0)
    return nullptr;

  size_t alignedRowBytes =
      alignTo(static_cast<size_t>(width) * pixelBytes, size_t(256));
  size_t totalBytes = alignedRowBytes * static_cast<size_t>(height);
  if (totalBytes == 0)
    return nullptr;

  MTL::Buffer *readback =
      _pDevice->newBuffer(static_cast<NS::UInteger>(totalBytes),
                          MTL::ResourceStorageModeShared);
  if (!readback)
    return nullptr;

  if (!blit)
    blit = cmd->blitCommandEncoder();
  if (!blit) {
    readback->release();
    return nullptr;
  }

  NS::UInteger bytesPerImage =
      static_cast<NS::UInteger>(alignedRowBytes * static_cast<size_t>(height));
  MTL::Size size = MTL::Size::Make(width, height, 1);
  MTL::Origin origin = MTL::Origin::Make(0, 0, 0);
  blit->copyFromTexture(colorTexture, 0, 0, origin, size, readback, 0,
                        static_cast<NS::UInteger>(alignedRowBytes),
                        bytesPerImage);

  std::ostringstream nameBuilder;
  nameBuilder << "frame_" << std::setfill('0') << std::setw(6) << frameIndex
              << ".exr";

  std::filesystem::path outputPath;
  try {
    outputPath = std::filesystem::path(_frameCaptureDirectory) / nameBuilder.str();
  } catch (const std::exception &e) {
    std::printf("Failed to build frame capture path: %s\n", e.what());
    readback->release();
    return nullptr;
  }

  auto capture = std::make_shared<FrameCaptureRequest>();
  capture->frameIndex = frameIndex;
  capture->filePath = outputPath.string();
  capture->buffer = readback;
  capture->width = static_cast<size_t>(width);
  capture->height = static_cast<size_t>(height);
  capture->alignedRowBytes = alignedRowBytes;
  capture->format = colorTexture->pixelFormat();

  auto captureAuxiliaryTexture = [&](MTL::Texture *src, MTL::Buffer *&dstBuffer,
                                     size_t &rowBytes, MTL::PixelFormat &format,
                                     const std::string &pathSuffix) {
    if (!src)
      return;
    if (src->width() != width || src->height() != height)
      return;
    size_t auxPixelBytes = bytesPerPixel(src->pixelFormat());
    if (auxPixelBytes == 0)
      return;
    rowBytes =
        alignTo(static_cast<size_t>(width) * auxPixelBytes, static_cast<size_t>(256));
    size_t auxTotalBytes = rowBytes * static_cast<size_t>(height);
    if (auxTotalBytes == 0)
      return;
    dstBuffer =
        _pDevice->newBuffer(static_cast<NS::UInteger>(auxTotalBytes),
                            MTL::ResourceStorageModeShared);
    if (!dstBuffer) {
      rowBytes = 0;
      return;
    }
    if (!blit)
      blit = cmd->blitCommandEncoder();
    if (!blit) {
      dstBuffer->release();
      dstBuffer = nullptr;
      rowBytes = 0;
      return;
    }
    blit->copyFromTexture(src, 0, 0, origin, size, dstBuffer, 0,
                          static_cast<NS::UInteger>(rowBytes), bytesPerImage);
    format = src->pixelFormat();
    try {
      std::filesystem::path auxPath = outputPath;
      auxPath.replace_filename(auxPath.stem().string() + pathSuffix +
                               auxPath.extension().string());
      if (pathSuffix == "_albedo")
        capture->albedoPath = auxPath.string();
      else if (pathSuffix == "_normal")
        capture->normalPath = auxPath.string();
    } catch (const std::exception &e) {
      std::printf("Failed to build auxiliary capture path: %s\n", e.what());
    }
  };

  captureAuxiliaryTexture(albedoTexture, capture->albedoBuffer,
                          capture->albedoAlignedRowBytes, capture->albedoFormat,
                          std::string("_albedo"));
  captureAuxiliaryTexture(normalTexture, capture->normalBuffer,
                          capture->normalAlignedRowBytes, capture->normalFormat,
                          std::string("_normal"));

  std::printf("Queued frame %llu for EXR capture: %s\n",
              static_cast<unsigned long long>(frameIndex),
              capture->filePath.c_str());

  return capture;
}

void Renderer::finalizeFrameCapture(
    const std::shared_ptr<FrameCaptureRequest> &capture) {
  if (!capture)
    return;

  auto releaseCaptureBuffers = [&]() {
    if (capture->buffer) {
      capture->buffer->release();
      capture->buffer = nullptr;
    }
    if (capture->albedoBuffer) {
      capture->albedoBuffer->release();
      capture->albedoBuffer = nullptr;
    }
    if (capture->normalBuffer) {
      capture->normalBuffer->release();
      capture->normalBuffer = nullptr;
    }
  };

  if (!capture->buffer) {
    std::printf("Frame capture buffer missing for frame %llu\n",
               static_cast<unsigned long long>(capture->frameIndex));
    releaseCaptureBuffers();
    return;
  }

  {
    std::lock_guard<std::mutex> lock(_captureMutex);
    _completedCaptures.push_back(capture);
  }
  _captureOutputsPending.store(true, std::memory_order_release);

  std::printf("Frame %llu queued for offline EXR denoising: %s\n",
             static_cast<unsigned long long>(capture->frameIndex),
             capture->filePath.c_str());
}

void Renderer::processPendingCapturedFrames() {
  while (true) {
    std::vector<std::shared_ptr<FrameCaptureRequest>> captures;
    {
      std::lock_guard<std::mutex> lock(_captureMutex);
      if (_completedCaptures.empty()) {
        _captureOutputsPending.store(false, std::memory_order_release);
        break;
      }
      captures.swap(_completedCaptures);
    }

    for (const auto &capture : captures) {
      if (!capture)
        continue;

      auto releaseCaptureBuffers = [&]() {
        if (capture->buffer) {
          capture->buffer->release();
          capture->buffer = nullptr;
        }
        if (capture->albedoBuffer) {
          capture->albedoBuffer->release();
          capture->albedoBuffer = nullptr;
        }
        if (capture->normalBuffer) {
          capture->normalBuffer->release();
          capture->normalBuffer = nullptr;
        }
      };

      if (!capture->buffer) {
        std::printf("Frame capture buffer missing for frame %llu\n",
                   static_cast<unsigned long long>(capture->frameIndex));
        releaseCaptureBuffers();
        continue;
      }

      void *contents = capture->buffer->contents();
    if (!contents) {
      std::printf("Failed to map frame capture buffer for frame %llu\n",
                 static_cast<unsigned long long>(capture->frameIndex));
      releaseCaptureBuffers();
      continue;
    }

    size_t pixelCount = capture->width * capture->height;
    if (pixelCount == 0) {
      releaseCaptureBuffers();
      continue;
    }

    auto convertBuffer = [&](const void *src, size_t rowBytes,
                             MTL::PixelFormat format, std::vector<float> &dst) {
      if (!src)
        return false;
      dst.assign(pixelCount * 4, 0.0f);
      if (format == MTL::PixelFormat::PixelFormatRGBA16Float) {
        for (size_t y = 0; y < capture->height; ++y) {
          const uint8_t *rowBase = static_cast<const uint8_t *>(src) +
                                   y * rowBytes;
          const uint16_t *row = reinterpret_cast<const uint16_t *>(rowBase);
          float *dstRow = dst.data() + y * capture->width * 4;
          for (size_t x = 0; x < capture->width; ++x) {
            for (size_t c = 0; c < 4; ++c) {
              tinyexr::FP16 value{};
              value.u = row[x * 4 + c];
              dstRow[x * 4 + c] = half_to_float(value).f;
            }
          }
        }
        return true;
      }
      if (format == MTL::PixelFormat::PixelFormatRGBA32Float) {
        for (size_t y = 0; y < capture->height; ++y) {
          const uint8_t *rowBase = static_cast<const uint8_t *>(src) +
                                   y * rowBytes;
          const float *row = reinterpret_cast<const float *>(rowBase);
          float *dstRow = dst.data() + y * capture->width * 4;
          std::memcpy(dstRow, row, sizeof(float) * capture->width * 4);
        }
        return true;
      }
      return false;
    };

    std::vector<float> rgba;
    if (!convertBuffer(contents, capture->alignedRowBytes, capture->format,
                       rgba)) {
      std::printf("Unsupported pixel format %u for frame capture.\n",
                 static_cast<unsigned>(capture->format));
      releaseCaptureBuffers();
      continue;
    }

    std::vector<float> albedoData;
    if (capture->albedoBuffer && capture->albedoAlignedRowBytes > 0) {
      if (void *albedoContents = capture->albedoBuffer->contents()) {
        if (!convertBuffer(albedoContents, capture->albedoAlignedRowBytes,
                           capture->albedoFormat, albedoData))
          albedoData.clear();
      }
    }

    std::vector<float> normalData;
    if (capture->normalBuffer && capture->normalAlignedRowBytes > 0) {
      if (void *normalContents = capture->normalBuffer->contents()) {
        if (!convertBuffer(normalContents, capture->normalAlignedRowBytes,
                           capture->normalFormat, normalData))
          normalData.clear();
      }
    }

    if (!albedoData.empty() && !normalData.empty()) {
      std::printf("Applying offline EXR denoiser to frame %llu\n",
                 static_cast<unsigned long long>(capture->frameIndex));

      const int radius = 1;
      const float spatialSigma =
          MetalCppPathTracer::DenoiserSettings::kSharpenedSpatialSigma;
      const float albedoSigma =
          MetalCppPathTracer::DenoiserSettings::kSharpenedAlbedoSigma;
      const float normalSigma =
          MetalCppPathTracer::DenoiserSettings::kSharpenedNormalSigma;
      std::vector<float> denoised(pixelCount * 3, 0.0f);

      auto normalizeVector = [](const float *v) {
        std::array<float, 3> out{v[0], v[1], v[2]};
        float len = std::sqrt(out[0] * out[0] + out[1] * out[1] +
                              out[2] * out[2]);
        if (len > 1e-6f) {
          out[0] /= len;
          out[1] /= len;
          out[2] /= len;
        }
        return out;
      };

      for (size_t y = 0; y < capture->height; ++y) {
        for (size_t x = 0; x < capture->width; ++x) {
          size_t index = y * capture->width + x;
          const float *baseColor = &rgba[index * 4];
          const float *baseAlbedo = &albedoData[index * 4];
          const float *baseNormalPtr = &normalData[index * 4];
          auto baseNormal = normalizeVector(baseNormalPtr);

          float accum[3] = {0.0f, 0.0f, 0.0f};
          float totalWeight = 0.0f;

          for (int dy = -radius; dy <= radius; ++dy) {
            int ny = static_cast<int>(y) + dy;
            if (ny < 0 || ny >= static_cast<int>(capture->height))
              continue;
            for (int dx = -radius; dx <= radius; ++dx) {
              int nx = static_cast<int>(x) + dx;
              if (nx < 0 || nx >= static_cast<int>(capture->width))
                continue;
              size_t neighborIndex = static_cast<size_t>(ny) * capture->width +
                                     static_cast<size_t>(nx);
              const float *neighborColor = &rgba[neighborIndex * 4];
              const float *neighborAlbedo = &albedoData[neighborIndex * 4];
              const float *neighborNormalPtr = &normalData[neighborIndex * 4];
              auto neighborNormal = normalizeVector(neighborNormalPtr);

              float spatialDist2 = static_cast<float>(dx * dx + dy * dy);
              float spatialWeight = std::exp(
                  -spatialDist2 / (2.0f * spatialSigma * spatialSigma));

              float albedoDiff0 = neighborAlbedo[0] - baseAlbedo[0];
              float albedoDiff1 = neighborAlbedo[1] - baseAlbedo[1];
              float albedoDiff2 = neighborAlbedo[2] - baseAlbedo[2];
              float albedoDist2 = albedoDiff0 * albedoDiff0 +
                                  albedoDiff1 * albedoDiff1 +
                                  albedoDiff2 * albedoDiff2;
              float albedoWeight = std::exp(
                  -albedoDist2 / (2.0f * albedoSigma * albedoSigma));

              float dotVal = baseNormal[0] * neighborNormal[0] +
                             baseNormal[1] * neighborNormal[1] +
                             baseNormal[2] * neighborNormal[2];
              dotVal = std::clamp(dotVal, -1.0f, 1.0f);
              float normalDiff = std::max(0.0f, 1.0f - dotVal);
              float normalWeight = std::exp(
                  -normalDiff / (2.0f * normalSigma * normalSigma));

              float weight = spatialWeight * albedoWeight * normalWeight;
              accum[0] += neighborColor[0] * weight;
              accum[1] += neighborColor[1] * weight;
              accum[2] += neighborColor[2] * weight;
              totalWeight += weight;
            }
          }

          std::array<float, 3> filtered{baseColor[0], baseColor[1],
                                         baseColor[2]};
          if (totalWeight > 1e-6f) {
            filtered[0] = accum[0] / totalWeight;
            filtered[1] = accum[1] / totalWeight;
            filtered[2] = accum[2] / totalWeight;
          }
          const float strength =
              MetalCppPathTracer::DenoiserSettings::kSharpenedDenoiseStrength;
          std::array<float, 3> blended{
              baseColor[0] + (filtered[0] - baseColor[0]) * strength,
              baseColor[1] + (filtered[1] - baseColor[1]) * strength,
              baseColor[2] + (filtered[2] - baseColor[2]) * strength,
          };
          if (MetalCppPathTracer::DenoiserSettings::kSharpenedClampOutput) {
            denoised[index * 3 + 0] = std::clamp(blended[0], 0.0f, 1.0f);
            denoised[index * 3 + 1] = std::clamp(blended[1], 0.0f, 1.0f);
            denoised[index * 3 + 2] = std::clamp(blended[2], 0.0f, 1.0f);
          } else {
            denoised[index * 3 + 0] = blended[0];
            denoised[index * 3 + 1] = blended[1];
            denoised[index * 3 + 2] = blended[2];
          }
        }
      }

      for (size_t i = 0; i < pixelCount; ++i) {
        rgba[i * 4 + 0] = denoised[i * 3 + 0];
        rgba[i * 4 + 1] = denoised[i * 3 + 1];
        rgba[i * 4 + 2] = denoised[i * 3 + 2];
        rgba[i * 4 + 3] = 1.0f;
      }
    } else {
      for (size_t i = 0; i < pixelCount; ++i)
        rgba[i * 4 + 3] = 1.0f;
    }

    const char *err = nullptr;
    int saveAsFp16 =
        capture->format == MTL::PixelFormat::PixelFormatRGBA16Float ? 1 : 0;
    int ret = SaveEXR(rgba.data(), static_cast<int>(capture->width),
                      static_cast<int>(capture->height), 4, saveAsFp16,
                      capture->filePath.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
      if (err) {
        std::printf("Failed to save EXR frame %llu: %s\n",
                   static_cast<unsigned long long>(capture->frameIndex), err);
        FreeEXRErrorMessage(err);
      } else {
        std::printf("Failed to save EXR frame %llu (error code %d)\n",
                   static_cast<unsigned long long>(capture->frameIndex), ret);
      }
    } else {
      std::printf("Saved EXR frame %llu to %s\n",
                 static_cast<unsigned long long>(capture->frameIndex),
                 capture->filePath.c_str());
    }

    auto saveAuxiliaryExr = [&](const std::vector<float> &data,
                                const std::string &path,
                                MTL::PixelFormat format, const char *label) {
      if (data.empty() || path.empty())
        return;
      std::vector<float> temp(data);
      for (size_t i = 0; i < pixelCount; ++i)
        temp[i * 4 + 3] = 1.0f;
      int auxSaveAsFp16 =
          format == MTL::PixelFormat::PixelFormatRGBA16Float ? 1 : 0;
      const char *auxErr = nullptr;
      int auxRet = SaveEXR(temp.data(), static_cast<int>(capture->width),
                           static_cast<int>(capture->height), 4, auxSaveAsFp16,
                           path.c_str(), &auxErr);
      if (auxRet != TINYEXR_SUCCESS) {
        if (auxErr) {
          std::printf("Failed to save %s for frame %llu: %s\n", label,
                     static_cast<unsigned long long>(capture->frameIndex),
                     auxErr);
          FreeEXRErrorMessage(auxErr);
        } else {
          std::printf("Failed to save %s for frame %llu (error %d)\n", label,
                     static_cast<unsigned long long>(capture->frameIndex),
                     auxRet);
        }
      } else {
        std::printf("Saved %s for frame %llu to %s\n", label,
                   static_cast<unsigned long long>(capture->frameIndex),
                   path.c_str());
      }
    };

    if (!albedoData.empty())
      saveAuxiliaryExr(albedoData, capture->albedoPath, capture->albedoFormat,
                       "albedo");

    if (!normalData.empty()) {
      for (size_t i = 0; i < pixelCount; ++i) {
        float nx = normalData[i * 4 + 0];
        float ny = normalData[i * 4 + 1];
        float nz = normalData[i * 4 + 2];
        float len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 1e-6f) {
          normalData[i * 4 + 0] = nx / len;
          normalData[i * 4 + 1] = ny / len;
          normalData[i * 4 + 2] = nz / len;
        }
      }
      saveAuxiliaryExr(normalData, capture->normalPath, capture->normalFormat,
                       "normal");
    }

      releaseCaptureBuffers();
    }
  }
}

void Renderer::setDeltaTime(double deltaSeconds) {
  if (deltaSeconds < 0.0)
    deltaSeconds = 0.0;

  _deltaTimeSeconds = deltaSeconds;
  Camera::deltaTime = static_cast<float>(_deltaTimeSeconds);
}

void Renderer::buildShaders() {
  using NS::StringEncoding::UTF8StringEncoding;

  if (_pPSO) {
    _pPSO->release();
    _pPSO = nullptr;
  }
  if (_pOverlayPSO) {
    _pOverlayPSO->release();
    _pOverlayPSO = nullptr;
  }
  if (_pPathTracePSO) {
    _pPathTracePSO->release();
    _pPathTracePSO = nullptr;
  }
  if (_pAdaptiveSamplingPSO) {
    _pAdaptiveSamplingPSO->release();
    _pAdaptiveSamplingPSO = nullptr;
  }

  NS::Error *pError = nullptr;
  MTL::Library *pLibrary = _pDevice->newDefaultLibrary();

  if (!pLibrary) {
    __builtin_printf("Failed to load Metal library\n");
    assert(false);
  }

  MTL::Function *pVertexFn = pLibrary->newFunction(
      NS::String::string("vertexMain", UTF8StringEncoding));
  MTL::Function *pFragFn = pLibrary->newFunction(
      NS::String::string("presentMain", UTF8StringEncoding));

  MTL::RenderPipelineDescriptor *pDesc =
      MTL::RenderPipelineDescriptor::alloc()->init();
  pDesc->setVertexFunction(pVertexFn);
  pDesc->setFragmentFunction(pFragFn);
  pDesc->colorAttachments()->object(0)->setPixelFormat(
      MTL::PixelFormat::PixelFormatRGBA16Float);

  _pPSO = _pDevice->newRenderPipelineState(pDesc, &pError);
  if (!_pPSO) {
    __builtin_printf("%s\n", pError->localizedDescription()->utf8String());
    assert(false);
  }

  pError = nullptr;
  MTL::Function *pOverlayVertexFn = pLibrary->newFunction(
      NS::String::string("frustumDebugVertexMain", UTF8StringEncoding));
  MTL::Function *pOverlayFragmentFn = pLibrary->newFunction(
      NS::String::string("frustumDebugFragmentMain", UTF8StringEncoding));
  if (pOverlayVertexFn && pOverlayFragmentFn) {
    MTL::RenderPipelineDescriptor *pOverlayDesc =
        MTL::RenderPipelineDescriptor::alloc()->init();
    pOverlayDesc->setVertexFunction(pOverlayVertexFn);
    pOverlayDesc->setFragmentFunction(pOverlayFragmentFn);
    pOverlayDesc->colorAttachments()->object(0)->setPixelFormat(
        MTL::PixelFormat::PixelFormatRGBA16Float);
    pOverlayDesc->setInputPrimitiveTopology(
        MTL::PrimitiveTopologyClass::PrimitiveTopologyClassLine);

    _pOverlayPSO = _pDevice->newRenderPipelineState(pOverlayDesc, &pError);
    if (!_pOverlayPSO && pError) {
      __builtin_printf("%s\n", pError->localizedDescription()->utf8String());
    }
    pOverlayDesc->release();
  }
  if (pOverlayVertexFn)
    pOverlayVertexFn->release();
  if (pOverlayFragmentFn)
    pOverlayFragmentFn->release();

  pError = nullptr;
  _useAccelerationStructureBindings = false;
  MTL::Function *pPathTraceFn = pLibrary->newFunction(
      NS::String::string("pathTraceKernel", UTF8StringEncoding));
  if (pPathTraceFn) {
    MTL::AutoreleasedComputePipelineReflection reflection = nullptr;
    _pPathTracePSO = _pDevice->newComputePipelineState(
        pPathTraceFn, MTL::PipelineOptionArgumentInfo, &reflection, &pError);
    if (!_pPathTracePSO) {
      if (pError) {
        __builtin_printf("%s\n", pError->localizedDescription()->utf8String());
      }
      pError = nullptr;
      reflection = nullptr;
      _pPathTracePSO =
          _pDevice->newComputePipelineState(pPathTraceFn, &pError);
      if (!_pPathTracePSO && pError) {
        __builtin_printf("%s\n", pError->localizedDescription()->utf8String());
      }
    }
    if (_pPathTracePSO && reflection) {
      NS::Array *arguments = reflection->arguments();
      if (arguments) {
        for (NS::UInteger i = 0; i < arguments->count(); ++i) {
          auto *argument =
              static_cast<MTL::Argument *>(arguments->object(i));
          if (!argument)
            continue;
          if (argument->type() ==
              MTL::ArgumentType::ArgumentTypeInstanceAccelerationStructure) {
            _useAccelerationStructureBindings = true;
            break;
          }
          if (argument->type() == MTL::ArgumentType::ArgumentTypeBuffer &&
              argument->bufferDataType() ==
                  MTL::DataType::DataTypeInstanceAccelerationStructure) {
            _useAccelerationStructureBindings = true;
            break;
          }
        }
      }
    }
    pPathTraceFn->release();
  }

  pError = nullptr;
  MTL::Function *pAdaptiveFn = pLibrary->newFunction(
      NS::String::string("adaptiveSamplingMain", UTF8StringEncoding));
  if (pAdaptiveFn) {
    _pAdaptiveSamplingPSO =
        _pDevice->newComputePipelineState(pAdaptiveFn, &pError);
    if (!_pAdaptiveSamplingPSO && pError) {
      __builtin_printf("%s\n", pError->localizedDescription()->utf8String());
    }
    pAdaptiveFn->release();
  }

  pVertexFn->release();
  pFragFn->release();
  pDesc->release();
  pLibrary->release();
}

Renderer::SceneAccelerationBuildResult
Renderer::buildSceneAccelerationStructures(size_t primitiveCount,
                                           size_t primitiveHitBytes) {
  SceneAccelerationBuildResult result{};

  MTL::CommandBuffer *clearCmd = nullptr;
  if (_pPrimitiveHitBufferGPU && primitiveHitBytes > 0 && _pCommandQueue) {
    clearCmd = _pCommandQueue->commandBuffer();
    if (clearCmd) {
      MTL::BlitCommandEncoder *blit = clearCmd->blitCommandEncoder();
      if (blit) {
        blit->fillBuffer(_pPrimitiveHitBufferGPU,
                         NS::Range::Make(0, primitiveHitBytes), 0);
        blit->endEncoding();
      }
    }
  }

  std::future<SceneAccelerationBuildResult> resultFuture;
  TaskLimiter *limiter = nullptr;

  if (_pScene) {
    TaskLimiter &limiterRef = sceneBvhTaskLimiter();
    limiterRef.acquire();
    limiter = &limiterRef;

    Scene *scene = _pScene;
    size_t primitiveCountCopy = primitiveCount;

    try {
      resultFuture = std::async(
          std::launch::async,
          [scene, primitiveCountCopy, limiter]() {
            struct LimiterGuard {
              TaskLimiter *limiter;
              ~LimiterGuard() {
                if (limiter)
                  limiter->release();
              }
            } guard{limiter};

            SceneAccelerationBuildResult threadResult{};
            if (scene) {
              scene->buildBVH();
              threadResult.blasNodeCount = scene->getBVHNodeCount();
              size_t tlasCount = 0;
              if (primitiveCountCopy > 0) {
                simd::float4 *tmp = scene->createTLASBuffer(tlasCount);
                if (tmp)
                  delete[] tmp;
              }
              threadResult.tlasNodeCount = tlasCount;
            }
            return threadResult;
          });
    } catch (...) {
      limiterRef.release();
      throw;
    }
  }

  if (clearCmd) {
    clearCmd->commit();
    clearCmd->waitUntilCompleted();
  }

  if (resultFuture.valid()) {
    try {
      result = resultFuture.get();
    } catch (...) {
      throw;
    }
  }

  return result;
}

void Renderer::updateVisibleScene() {
  resetProbabilisticResidencyState();
  if (!SceneLoader::LoadSceneFromXML("scene.xml", _pScene)) {
    std::filesystem::path alt =
        std::filesystem::path(__FILE__).parent_path() / "../scene_bunny_room.xml";
    SceneLoader::LoadSceneFromXML(alt.string(), _pScene);
  }

  Camera::screenSize = _pScene->screenSize;
  _animationFrame = 0;
  _observerActive = false;

  if (!_pScene->cameraPath.empty()) {
    const auto &k = _pScene->cameraPath.front();
    _primaryCameraState = makeCameraState(
        k.position, k.lookAt, _primaryCameraState.verticalFov,
        _primaryCameraState.focalLength, _primaryCameraState.forward);
  }

  Camera::applyState(_primaryCameraState);
  _primaryCameraState = Camera::captureState();

  if (_pScene->hasObserverCamera()) {
    const auto &observer = _pScene->getObserverCamera();
    float fov = observer.verticalFov > 0.0f ? observer.verticalFov
                                            : _primaryCameraState.verticalFov;
    _observerCameraState =
        makeCameraState(observer.position, observer.lookAt, fov,
                        _primaryCameraState.focalLength,
                        _primaryCameraState.forward);
  } else {
    _observerCameraState = _primaryCameraState;
  }

  Camera::applyState(_primaryCameraState);

  printf("Scene loaded: %zu total primitives (%zu spheres, %zu triangles, %zu "
         "rectangles)\n",
         _pScene->getPrimitiveCount(), _pScene->getSphereCount(),
         _pScene->getTriangleCount(), _pScene->getRectangleCount());

  _residencyConfig = _pScene->getResidencyParameters();
  _residencyConfig.normalizeEnvironmentDepthSettings();
  bool requiresCachedBlas =
      _pScene->getResidencyStrategy() != ResidencyStrategy::AlwaysResident;
  if (_residencyConfig.buildCachedBlas != requiresCachedBlas) {
    _residencyConfig.buildCachedBlas = requiresCachedBlas;
    _pScene->setResidencyParameters(_residencyConfig);
  }
  _textureResidencyMemoryCapMB =
      std::max(_pScene->getTextureResidencyMemoryCapMB(), 0.0);
  if (_textureResidencyMemoryCapMB <= 0.0)
    _textureResidencyMemoryCapMB = kDefaultTextureResidencyMemoryCapMB;
  printf("Texture residency memory cap: %.1f MB\n",
         _textureResidencyMemoryCapMB);
  _residentCompacted = _pScene->getStartCompacted();
  _compactionCooldown = 0;

  printf("LOD activation threshold: %.1f, deactivation threshold: %.1f (cooldown "
         "%u frames, toggle budget %zu)\n",
         _residencyConfig.lodEnterDistance, _residencyConfig.lodExitDistance,
         _residencyConfig.stateCooldownFrames,
         _residencyConfig.lodMaxTogglesPerFrame);

  const char *strategyName = nullptr;
  switch (_pScene->getResidencyStrategy()) {
  case ResidencyStrategy::EnergyImportance:
    strategyName = "Energy importance";
    break;
  case ResidencyStrategy::RayHitBudget:
    strategyName = "Ray-hit budget";
    break;
  case ResidencyStrategy::ScreenSpaceFootprint:
    strategyName = "Screen-space footprint";
    break;
  case ResidencyStrategy::UnifiedScore:
    strategyName = "Unified score";
    break;
  case ResidencyStrategy::EnvironmentHit:
    strategyName = "Environment hit";
    break;
  case ResidencyStrategy::AlwaysResident:
    strategyName = "Always resident";
    break;
  case ResidencyStrategy::DistanceLOD:
  default:
    strategyName = "Distance-based LOD";
    break;
  }
  printf("Active primitive residency strategy: %s\n", strategyName);

  _alwaysResidentCache.reset();

  ++_cameraVersion;
  _coverageCameraVersion = 0;
  _boundsVersionCounter = 1;

  // Store full primitive list and initialize tracking
  _allPrimitives = _pScene->getPrimitives();
  size_t primCount = _allPrimitives.size();
  _activePrimitive.assign(primCount, false);
  _primitiveCooldown.assign(primCount, 0);
  _primitiveToResidentIndex.assign(primCount, -1);
  _primitiveToObject.assign(primCount, std::numeric_limits<size_t>::max());
  _primitiveBounds.resize(primCount);
  _primitiveBoundsVersion.assign(primCount, _boundsVersionCounter++);
  _primitiveImportance.assign(primCount, 0.0f);
  _objectImportance.clear();
  _objectImportanceHistory.clear();
  _energySortedIndices.clear();
  _primitiveHitScores.assign(primCount, 0.0f);
  _primitiveHitLastFrame.assign(primCount, 0);
  _primitiveRayContributions.assign(primCount, 0.0f);
  _primitiveRaysTestedLastFrame.assign(primCount, 0);
  _primitiveHitAlpha.assign(primCount, 1.0f);
  _primitiveHitBeta.assign(primCount, 1.0f);
  _primitiveHitProbability.assign(primCount, 0.5f);
  _primitiveHitVariance.assign(primCount, 1.0f / 12.0f);
  _primitivePosteriorMass.assign(primCount, 2.0f);
  _primitiveVisible.assign(primCount, 0);
  _rayHitSortedIndices.resize(primCount);
  _probabilitySortedIndices.resize(primCount);
  _primitiveScreenCoverage.assign(primCount, 0.0f);
  _primitiveDistanceFalloffCache.assign(primCount, 0.0f);
  _primitiveCoverageDirty.assign(primCount, 1);
  _primitiveCoverageBoundsVersion.assign(primCount, 0);
  _primitiveCoverageVisibilityKey.assign(primCount, 0xFF);
  _screenCoverageSortedIndices.resize(primCount);
  _totalPrimitiveImportance = 0.0f;

  _residentPrimitives.clear();
  _residentRemap.clear();
  _recentlyActivated.clear();
  _recentlyDeactivated.clear();
  _residentBuffersInitialized = false;
  _cachedPrimitiveData.clear();
  _cachedMaterialData.clear();
  _cachedPrimitiveIndices.clear();
  _cachedBVHNodes.clear();
  _cachedTLASNodes.clear();
  _cachedTriangleVertices.clear();
  _cachedTriangleIndices.clear();
  _cachedLightIndices.clear();
  _cachedLightCdf.clear();
  _cpuActiveMask.clear();

  size_t totalTriangleCount = _pScene->getTriangleCount();
  _maxPrimitiveCount = std::max<size_t>(primCount, 1);
  _maxTriangleVertexCount = std::max<size_t>(totalTriangleCount * 3, 1);
  _maxTriangleIndexCount = std::max<size_t>(totalTriangleCount, 1);
  std::vector<float> importancePartials;
  const unsigned int partialThreadCount =
      std::max(1u, std::thread::hardware_concurrency());
  importancePartials.reserve(partialThreadCount * 2);

  std::mutex partialMutex;
  parallelChunkedAsync(0, primCount, [&](size_t chunkBegin, size_t chunkEnd) {
    float localImportance = 0.0f;
    for (size_t i = chunkBegin; i < chunkEnd; ++i) {
      const Primitive &p = _allPrimitives[i];
      if (p.type == PrimitiveType::Sphere) {
        _primitiveBounds[i] =
            BoundingSphere{p.sphere.center, p.sphere.radius};
      } else if (p.type == PrimitiveType::Triangle) {
        simd::float3 c =
            (p.triangle.v0 + p.triangle.v1 + p.triangle.v2) / 3.0f;
        float r = simd::length(p.triangle.v0 - c);
        r = std::max(r, (float)simd::length(p.triangle.v1 - c));
        r = std::max(r, (float)simd::length(p.triangle.v2 - c));
        _primitiveBounds[i] = {c, r};
      } else {
        float r = simd::length(p.rectangle.u) + simd::length(p.rectangle.v);
        _primitiveBounds[i] = {p.rectangle.center, r};
      }

      _primitiveImportance[i] = primitiveImportance(p);
      if (i < _rayHitSortedIndices.size())
        _rayHitSortedIndices[i] = i;
      if (i < _probabilitySortedIndices.size())
        _probabilitySortedIndices[i] = i;
      if (i < _screenCoverageSortedIndices.size())
        _screenCoverageSortedIndices[i] = i;
      localImportance += std::max(_primitiveImportance[i], 0.0f);
    }

    std::lock_guard<std::mutex> lock(partialMutex);
    importancePartials.push_back(localImportance);
  });

  _totalPrimitiveImportance =
      std::accumulate(importancePartials.begin(), importancePartials.end(),
                      0.0f);

  constexpr size_t kStatsPerPrimitive = 2;
  size_t hitCount = std::max<size_t>(_maxPrimitiveCount, 1);
  size_t hitBytes = hitCount * kStatsPerPrimitive * sizeof(uint32_t);
  flushRayHitCopy();
  ensureBufferCapacity(_pPrimitiveHitBufferGPU, hitBytes,
                       _primitiveHitBufferCapacity, false,
                       MTL::ResourceStorageModePrivate);
  ensureBufferCapacity(_pPrimitiveHitReadback, hitBytes,
                       _primitiveHitReadbackCapacity, false,
                       MTL::ResourceStorageModeShared);
  if (uint32_t *hitPtr =
          _pPrimitiveHitReadback
              ? static_cast<uint32_t *>(_pPrimitiveHitReadback->contents())
              : nullptr) {
    std::memset(hitPtr, 0, hitBytes);
  }
  SceneAccelerationBuildResult accelerationBuild =
      buildSceneAccelerationStructures(primCount, hitBytes);

  _rayHitRebuildCooldown = 0;

  _maxBlasNodeCount =
      std::max<size_t>(accelerationBuild.blasNodeCount, size_t(1));
  _maxTlasNodeCount =
      std::max<size_t>(accelerationBuild.tlasNodeCount, size_t(1));
  _totalNodeCount = _maxBlasNodeCount + _maxTlasNodeCount;

  _allSceneObjects = _pScene->getObjects();
  size_t objectCount = _allSceneObjects.size();
  _objectBounds.resize(objectCount);
  _objectBoundsVersion.assign(objectCount, _boundsVersionCounter++);
  _objectActive.assign(objectCount, false);
  _objectCooldown.assign(objectCount, 0);
  _objectLastToggleFrame.assign(objectCount, 0);
  _objectImportance.assign(objectCount, 0.0f);
  _energySortedIndices.resize(objectCount);
  std::iota(_energySortedIndices.begin(), _energySortedIndices.end(), size_t(0));
  _residentObjectGpuResources.resize(objectCount);
  _objectHitAlpha.assign(objectCount, 1.0f);
  _objectHitBeta.assign(objectCount, 1.0f);
  _objectHitProbability.assign(objectCount, 0.5f);
  _objectHitVariance.assign(objectCount, 1.0f / 12.0f);
  _objectPosteriorMass.assign(objectCount, 2.0f);
  _objectExplorationScore.assign(objectCount, 0.0f);
  _objectHitLastFrame.assign(objectCount, 0);
  _objectRaysTestedLastFrame.assign(objectCount, 0);
  _objectVisible.assign(objectCount, 0);
  _objectVisibilityEvidence.assign(objectCount, 0.0f);
  _objectProbabilitySortedIndices.resize(objectCount);
  std::iota(_objectProbabilitySortedIndices.begin(),
            _objectProbabilitySortedIndices.end(), size_t(0));

  _meshGroups.clear();
  _meshGroups.reserve(objectCount);
  _objectPrimitiveCounts.assign(objectCount, 0);
  _objectActivePrimitiveCounts.assign(objectCount, 0);
  _anyMeshGroups = false;
  std::unordered_map<int, size_t> meshGroupLookup;
  meshGroupLookup.reserve(objectCount);

  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    auto &resident = _residentObjectGpuResources[objectIndex];
    resident.clearPendingCommand();
    resident.byteSize = 0;
    resident.state = ResidentObjectGpuResources::ResidencyState::Cold;
    resident.lastStateChange = std::chrono::steady_clock::now();
    resident.resources.initialize(_pDevice);
  }

  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    const SceneObject &obj = _allSceneObjects[objectIndex];
    simd::float3 boundsMin = obj.boundsMin;
    simd::float3 boundsMax = obj.boundsMax;
    simd::float3 center = (boundsMin + boundsMax) * 0.5f;
    float radius = simd::length(boundsMax - center);
    _objectBounds[objectIndex] = {center, radius};

    size_t first = obj.firstPrimitive;
    size_t last = std::min(first + obj.primitiveCount, _primitiveToObject.size());
    size_t declaredCount = obj.primitiveCount;
    size_t actualCount = (last > first) ? (last - first) : 0;
    size_t primitiveCount = declaredCount > 0 ? declaredCount : actualCount;
    _objectPrimitiveCounts[objectIndex] = primitiveCount;

    size_t groupIndex = std::numeric_limits<size_t>::max();
    if (obj.meshGroupId >= 0) {
      _anyMeshGroups = true;
      auto it = meshGroupLookup.find(obj.meshGroupId);
      if (it == meshGroupLookup.end()) {
        groupIndex = _meshGroups.size();
        meshGroupLookup.emplace(obj.meshGroupId, groupIndex);
        _meshGroups.emplace_back();
      } else {
        groupIndex = it->second;
      }
    } else {
      groupIndex = _meshGroups.size();
      _meshGroups.emplace_back();
    }

    if (groupIndex < _meshGroups.size()) {
      auto &group = _meshGroups[groupIndex];
      if (group.objectIndices.empty())
        group.meshGroupId = obj.meshGroupId;
      group.objectIndices.push_back(objectIndex);
      group.primitiveCount += primitiveCount;
    }

    for (size_t prim = first; prim < last && prim < _primitiveToObject.size();
         ++prim)
      _primitiveToObject[prim] = objectIndex;
  }

  updateResidency(true, true);
  rebuildEnvironmentTexture();
}


std::array<simd::float3, 8>
Renderer::buildFrustumCorners(const Camera::State &state, float nearDistance,
                              float farDistance) const {
  std::array<simd::float3, 8> corners{};
  if (nearDistance <= 0.0f || farDistance <= nearDistance)
    return corners;

  float aspectRatio = Camera::screenSize.y > 0.0f
                           ? Camera::screenSize.x / Camera::screenSize.y
                           : 1.0f;
  float fovRad = state.verticalFov * static_cast<float>(M_PI) / 180.0f;
  float tanHalfFov = std::tan(fovRad * 0.5f);

  simd::float3 forward = state.forward;
  if (simd::length_squared(forward) < 1e-6f)
    forward = {0.0f, 0.0f, -1.0f};
  forward = simd::normalize(forward);

  simd::float3 up = state.up;
  if (simd::length_squared(up) < 1e-6f)
    up = {0.0f, 1.0f, 0.0f};
  up = simd::normalize(up);

  simd::float3 right = simd::cross(forward, up);
  if (simd::length_squared(right) < 1e-6f)
    right = {1.0f, 0.0f, 0.0f};
  right = simd::normalize(right);
  up = simd::normalize(simd::cross(right, forward));

  auto buildPlane = [&](float distance, float halfWidth, float halfHeight,
                        size_t indexBase) {
    simd::float3 center = state.position + forward * distance;
    simd::float3 offsetRight = right * halfWidth;
    simd::float3 offsetUp = up * halfHeight;
    corners[indexBase + 0] = center + offsetUp - offsetRight;
    corners[indexBase + 1] = center + offsetUp + offsetRight;
    corners[indexBase + 2] = center - offsetUp + offsetRight;
    corners[indexBase + 3] = center - offsetUp - offsetRight;
  };

  float nearHalfHeight = tanHalfFov * nearDistance;
  float nearHalfWidth = nearHalfHeight * aspectRatio;
  float farHalfHeight = tanHalfFov * farDistance;
  float farHalfWidth = farHalfHeight * aspectRatio;

  buildPlane(nearDistance, nearHalfWidth, nearHalfHeight, 0);
  buildPlane(farDistance, farHalfWidth, farHalfHeight, 4);

  return corners;
}

void Renderer::recalculateViewport() {

  float aspectRatio = Camera::screenSize.x / Camera::screenSize.y;
  float fovRad = Camera::verticalFov * (M_PI / 180.0f);
  float halfHeight = tanf(fovRad * 0.5f);
  float halfWidth = aspectRatio * halfHeight;

  simd::float3 w = simd::normalize(-Camera::forward);
  simd::float3 u = simd::normalize(simd::cross(Camera::up, w));
  simd::float3 v = simd::cross(w, u);

  simd::float3 viewportU = u * (2.0f * halfWidth);
  simd::float3 viewportV = -v * (2.0f * halfHeight);

  simd::float3 firstPixelPosition =
      Camera::position - w - (viewportU * 0.5f) - (viewportV * 0.5f);

  simd::float3 rayDx = viewportU / Camera::screenSize.x;
  simd::float3 rayDy = viewportV / Camera::screenSize.y;

  UniformsData *uData = (UniformsData *)_pUniformsBuffer->contents();
  uData->cameraPosition = Camera::position;
  uData->viewportU = viewportU;
  uData->viewportV = viewportV;
  uData->firstPixelPosition = firstPixelPosition;
  uData->rayDx = rayDx;
  uData->rayDy = rayDy;
  uData->screenSize = Camera::screenSize;

  markBufferModified(_pUniformsBuffer, NS::Range::Make(0, sizeof(UniformsData)));

}

bool Renderer::buildObjectBlas(size_t objectIndex, const SceneObject &object,
                               ResidentObjectGpuResources &resident) {
  if (!_pDevice || !_pCommandQueue)
    return false;

  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

  auto cleanupPool = [&]() {
    if (pool) {
      pool->release();
      pool = nullptr;
    }
  };

  size_t first = object.firstPrimitive;
  size_t last = std::min(first + object.primitiveCount, _allPrimitives.size());

  if (last <= first) {
    resident.resources.ensureAccelerationStructure(0, nullptr);
    resident.byteSize = 0;
    resident.triangleCount = 0;
    resident.vertexCount = 0;
    resident.vertexBufferOffset = 0;
    resident.indexBufferOffset = 0;
    resident.geometryValid = false;
    cleanupPool();
    return true;
  }

  const size_t primCount = last - first;
  struct ChunkGeometry {
    size_t offset = 0;
    std::vector<simd::float3> vertices;
    std::vector<uint32_t> indices;
  };

  std::vector<ChunkGeometry> chunks;
  std::mutex chunkMutex;

  parallelChunkedAsync(0, primCount, [&](size_t chunkBegin, size_t chunkEnd) {
    ChunkGeometry local{};
    local.offset = chunkBegin;
    local.vertices.reserve((chunkEnd - chunkBegin) * 3);
    local.indices.reserve((chunkEnd - chunkBegin) * 3);

    for (size_t prim = first + chunkBegin; prim < first + chunkEnd &&
                                        prim < _allPrimitives.size();
         ++prim) {
      const Primitive &p = _allPrimitives[prim];
      if (p.type != PrimitiveType::Triangle)
        continue;

      uint32_t baseIndex = static_cast<uint32_t>(local.vertices.size());
      local.vertices.push_back(p.triangle.v0);
      local.vertices.push_back(p.triangle.v1);
      local.vertices.push_back(p.triangle.v2);
      local.indices.push_back(baseIndex + 0);
      local.indices.push_back(baseIndex + 1);
      local.indices.push_back(baseIndex + 2);
    }

    if (!local.indices.empty()) {
      std::lock_guard<std::mutex> lock(chunkMutex);
      chunks.emplace_back(std::move(local));
    }
  });

  std::sort(chunks.begin(), chunks.end(),
            [](const ChunkGeometry &a, const ChunkGeometry &b) {
              return a.offset < b.offset;
            });

  size_t totalVertices = 0;
  size_t totalIndices = 0;
  for (const auto &chunk : chunks) {
    totalVertices += chunk.vertices.size();
    totalIndices += chunk.indices.size();
  }

  std::vector<simd::float3> vertices;
  std::vector<uint32_t> indices;
  vertices.reserve(totalVertices);
  indices.reserve(totalIndices);

  bool geometryValid = true;
  size_t expectedTriangles = totalIndices / 3;

  for (const auto &chunk : chunks) {
    if (chunk.indices.size() % 3 != 0) {
      geometryValid = false;
      break;
    }
    uint32_t baseOffset = static_cast<uint32_t>(vertices.size());

    uint32_t maxIndex = 0;
    for (uint32_t idx : chunk.indices)
      maxIndex = std::max(maxIndex, idx);

    if (chunk.vertices.empty() && !chunk.indices.empty()) {
      geometryValid = false;
      break;
    }
    if (maxIndex >= chunk.vertices.size()) {
      geometryValid = false;
      break;
    }

    vertices.insert(vertices.end(), chunk.vertices.begin(), chunk.vertices.end());
    indices.reserve(indices.size() + chunk.indices.size());
    for (uint32_t idx : chunk.indices)
      indices.push_back(baseOffset + idx);
  }

  size_t triangleCount = indices.size() / 3;
  if (!geometryValid || triangleCount != expectedTriangles) {
    cleanupPool();
    return false;
  }

  if (triangleCount == 0) {
    resident.resources.ensureAccelerationStructure(0, nullptr);
    resident.byteSize = 0;
    resident.triangleCount = 0;
    resident.vertexCount = 0;
    resident.vertexBufferOffset = 0;
    resident.indexBufferOffset = 0;
    resident.geometryValid = false;
    cleanupPool();
    return true;
  }

  auto buildRequest = std::make_shared<PendingBlasBuild>();
  if (!buildRequest) {
    cleanupPool();
    return false;
  }

  buildRequest->renderer = this;
  buildRequest->resident = &resident;
  buildRequest->objectIndex = objectIndex;
  buildRequest->vertices = std::move(vertices);
  buildRequest->indices = std::move(indices);
  buildRequest->triangleCount = triangleCount;
  buildRequest->vertexCount = buildRequest->vertices.size();

  enqueueBlasBuild(buildRequest);

  cleanupPool();
  return true;
}

void Renderer::enqueueBlasBuild(
    const std::shared_ptr<PendingBlasBuild> &buildRequest) {
  if (!buildRequest)
    return;

  _pendingBlasBuilds.push_back(buildRequest);
  processBlasBuildQueue();
}

void Renderer::processBlasBuildQueue() {
  if (!_pDevice || !_pCommandQueue)
    return;

  while (!_pendingBlasBuilds.empty() &&
         _activeBlasBuilds.size() < kMaxBlasBuildsInFlight) {
    auto buildRequest = _pendingBlasBuilds.front();
    if (!startBlasBuild(buildRequest)) {
      _pendingBlasBuilds.pop_front();
      if (buildRequest && buildRequest->resident &&
          buildRequest->objectIndex < _instanceRecords.size()) {
        transitionResidentToCold(buildRequest->objectIndex,
                                 *buildRequest->resident,
                                 _instanceRecords[buildRequest->objectIndex]);
      }
      continue;
    }

    _pendingBlasBuilds.pop_front();
    _activeBlasBuilds.push_back(buildRequest);
  }
}

bool Renderer::startBlasBuild(
    const std::shared_ptr<PendingBlasBuild> &buildRequest) {
  if (!buildRequest || !buildRequest->resident || !_pDevice || !_pCommandQueue)
    return false;

  auto &resident = *buildRequest->resident;

  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  auto cleanupPool = [&]() {
    if (pool) {
      pool->release();
      pool = nullptr;
    }
  };

  const NS::UInteger vertexBytes = static_cast<NS::UInteger>(
      buildRequest->vertices.size() * sizeof(simd::float3));
  const NS::UInteger indexBytes = static_cast<NS::UInteger>(
      buildRequest->indices.size() * sizeof(uint32_t));

  std::string blasLabel = "ObjectBLAS_" + std::to_string(buildRequest->objectIndex);
  std::string vertexLabel =
      "ObjectVertices_" + std::to_string(buildRequest->objectIndex);
  std::string indexLabel =
      "ObjectIndices_" + std::to_string(buildRequest->objectIndex);

  auto geometryDesc = MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()
                          ->init();
  if (!geometryDesc) {
    cleanupPool();
    return false;
  }
  geometryDesc->setOpaque(true);

  NS::Object *descriptorObjects[] = {geometryDesc};
  auto geometryArray = NS::Array::alloc()->init(descriptorObjects, 1);
  auto accelDesc =
      MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
  if (!geometryArray || !accelDesc) {
    if (geometryArray)
      geometryArray->release();
    geometryDesc->release();
    if (accelDesc)
      accelDesc->release();
    cleanupPool();
    return false;
  }

  accelDesc->setGeometryDescriptors(geometryArray);

  NS::UInteger alignedVertexSize =
      vertexBytes > 0 ? resident.resources.alignedHeapSize(vertexBytes) : 0;
  NS::UInteger alignedIndexSize =
      indexBytes > 0 ? resident.resources.alignedHeapSize(indexBytes) : 0;

  MTL::Buffer *vertexBuffer = nullptr;
  MTL::Buffer *indexBuffer = nullptr;

  auto requestGeometryBuffers = [&]() -> bool {
    NS::UInteger initialHeapBytes = alignedVertexSize + alignedIndexSize;
    if (initialHeapBytes > 0)
      resident.resources.ensureHeapCapacity(initialHeapBytes);

    vertexBuffer = resident.resources.ensureVertexBuffer(vertexBytes,
                                                        vertexLabel.c_str());
    indexBuffer = resident.resources.ensureIndexBuffer(indexBytes,
                                                      indexLabel.c_str());
    return vertexBuffer && indexBuffer;
  };

  if (!requestGeometryBuffers()) {
    geometryArray->release();
    geometryDesc->release();
    accelDesc->release();
    cleanupPool();
    return false;
  }

  auto configureGeometryDescriptor = [&]() {
    geometryDesc->setVertexBuffer(vertexBuffer);
    geometryDesc->setVertexBufferOffset(0);
    geometryDesc->setVertexStride(sizeof(simd::float3));
    geometryDesc->setVertexFormat(
        MTL::AttributeFormat::AttributeFormatFloat3);
    geometryDesc->setIndexBuffer(indexBuffer);
    geometryDesc->setIndexBufferOffset(0);
    geometryDesc->setIndexType(MTL::IndexType::IndexTypeUInt32);
    geometryDesc->setTriangleCount(
        static_cast<NS::UInteger>(buildRequest->triangleCount));
  };

  NS::UInteger totalHeapBytes = 0;
  NS::UInteger alignedAccelerationSize = 0;
  MTL::AccelerationStructureSizes sizes{};
  MTL::SizeAndAlign heapAlign{};

  while (true) {
    configureGeometryDescriptor();

    sizes = _pDevice->accelerationStructureSizes(accelDesc);
    heapAlign = _pDevice->heapAccelerationStructureSizeAndAlign(accelDesc);

    alignedAccelerationSize = static_cast<NS::UInteger>(
        alignTo(static_cast<size_t>(heapAlign.size),
                static_cast<size_t>(heapAlign.align)));
    alignedAccelerationSize =
        resident.resources.alignedHeapSize(alignedAccelerationSize);

    NS::UInteger requiredTotal = alignedAccelerationSize + alignedVertexSize +
                                 alignedIndexSize;

    if (requiredTotal > totalHeapBytes) {
      totalHeapBytes = requiredTotal;
      resident.resources.ensureHeapCapacity(totalHeapBytes);

      if (!requestGeometryBuffers()) {
        geometryArray->release();
        geometryDesc->release();
        accelDesc->release();
        cleanupPool();
        return false;
      }
      continue;
    }

    totalHeapBytes = requiredTotal;
    break;
  }

  configureGeometryDescriptor();

  auto accelerationStructure = resident.resources.ensureAccelerationStructure(
      sizes.accelerationStructureSize, blasLabel.c_str());

  if (!accelerationStructure) {
    geometryArray->release();
    geometryDesc->release();
    accelDesc->release();
    cleanupPool();
    return false;
  }

  MTL::Buffer *vertexStaging = nullptr;
  MTL::Buffer *indexStaging = nullptr;
  if (vertexBytes > 0) {
    vertexStaging =
        _pDevice->newBuffer(vertexBytes, MTL::ResourceStorageModeShared);
    if (vertexStaging) {
      std::memcpy(vertexStaging->contents(), buildRequest->vertices.data(),
                  vertexBytes);
      markBufferModified(vertexStaging, NS::Range::Make(0, vertexBytes));
    }
  }
  if (indexBytes > 0) {
    indexStaging =
        _pDevice->newBuffer(indexBytes, MTL::ResourceStorageModeShared);
    if (indexStaging) {
      std::memcpy(indexStaging->contents(), buildRequest->indices.data(),
                  indexBytes);
      markBufferModified(indexStaging, NS::Range::Make(0, indexBytes));
    }
  }

  if ((vertexBytes > 0 && !vertexStaging) ||
      (indexBytes > 0 && !indexStaging)) {
    if (indexStaging)
      indexStaging->release();
    if (vertexStaging)
      vertexStaging->release();
    geometryArray->release();
    geometryDesc->release();
    accelDesc->release();
    cleanupPool();
    return false;
  }

  NS::UInteger requestedScratchSize = sizes.buildScratchBufferSize;
  MTL::Buffer *scratchBuffer = nullptr;
  buildRequest->scratchBuffer = nullptr;
  buildRequest->scratchSize = 0;
  if (requestedScratchSize > 0) {
    bool scratchReused = false;
    NS::UInteger allocatedScratchSize = requestedScratchSize;
    scratchBuffer = acquireBlasScratchBuffer(requestedScratchSize,
                                            allocatedScratchSize,
                                            scratchReused);
    if (!scratchBuffer) {
      if (indexStaging)
        indexStaging->release();
      if (vertexStaging)
        vertexStaging->release();
      geometryArray->release();
      geometryDesc->release();
      accelDesc->release();
      cleanupPool();
      return false;
    }

    buildRequest->scratchBuffer = scratchBuffer;
    buildRequest->scratchSize = allocatedScratchSize;
    (void)scratchReused;
  }

  auto commandBuffer = _pCommandQueue->commandBuffer();
  if (!commandBuffer) {
    if (scratchBuffer) {
      recycleBlasScratchBuffer(scratchBuffer, buildRequest->scratchSize);
      buildRequest->scratchBuffer = nullptr;
      buildRequest->scratchSize = 0;
    }
    if (indexStaging)
      indexStaging->release();
    if (vertexStaging)
      vertexStaging->release();
    geometryArray->release();
    geometryDesc->release();
    accelDesc->release();
    cleanupPool();
    return false;
  }

  MTL::BlitCommandEncoder *blitEncoder = nullptr;
  auto ensureBlitEncoder = [&]() -> MTL::BlitCommandEncoder * {
    if (!blitEncoder)
      blitEncoder = commandBuffer->blitCommandEncoder();
    return blitEncoder;
  };

  if (vertexStaging && vertexBytes > 0)
    ensureBlitEncoder()->copyFromBuffer(vertexStaging, 0, vertexBuffer, 0,
                                        vertexBytes);
  if (indexStaging && indexBytes > 0)
    ensureBlitEncoder()->copyFromBuffer(indexStaging, 0, indexBuffer, 0,
                                        indexBytes);

  if (blitEncoder)
    blitEncoder->endEncoding();

  auto asEncoder = commandBuffer->accelerationStructureCommandEncoder();
  if (!asEncoder) {
    if (scratchBuffer) {
      recycleBlasScratchBuffer(scratchBuffer, buildRequest->scratchSize);
      buildRequest->scratchBuffer = nullptr;
      buildRequest->scratchSize = 0;
    }
    if (indexStaging)
      indexStaging->release();
    if (vertexStaging)
      vertexStaging->release();
    geometryArray->release();
    geometryDesc->release();
    accelDesc->release();
    cleanupPool();
    return false;
  }

  asEncoder->buildAccelerationStructure(accelerationStructure, accelDesc,
                                        scratchBuffer, 0);
  asEncoder->endEncoding();

  buildRequest->geometryDesc = geometryDesc;
  buildRequest->geometryArray = geometryArray;
  buildRequest->accelDesc = accelDesc;
  buildRequest->accelerationStructure = accelerationStructure;
  buildRequest->vertexStaging = vertexStaging;
  buildRequest->indexStaging = indexStaging;
  buildRequest->scratchBuffer = scratchBuffer;
  buildRequest->commandBuffer = commandBuffer;
  buildRequest->totalHeapBytes = totalHeapBytes;

  // Release CPU copies now that staging buffers are populated.
  buildRequest->vertices.clear();
  buildRequest->vertices.shrink_to_fit();
  buildRequest->indices.clear();
  buildRequest->indices.shrink_to_fit();

  resident.transitionToStreaming(commandBuffer);

  auto completion = [this, buildRequest](bool success) {
    this->handleCompletedBlasBuild(buildRequest, success);
  };

  if (!submitAsyncCommandBuffer(commandBuffer, completion)) {
    resident.clearPendingCommand();
    buildRequest->releaseResources();
    cleanupPool();
    return false;
  }

  cleanupPool();
  return true;
}

void Renderer::handleCompletedBlasBuild(
    const std::shared_ptr<PendingBlasBuild> &buildRequest, bool success) {
  if (!buildRequest || !buildRequest->resident)
    return;

  auto &resident = *buildRequest->resident;

  resident.clearPendingCommand();

  if (success) {
    resident.byteSize = buildRequest->totalHeapBytes;
    resident.triangleCount = buildRequest->triangleCount;
    resident.vertexCount = buildRequest->vertexCount;
    resident.vertexBufferOffset = 0;
    resident.indexBufferOffset = 0;
    resident.geometryValid = buildRequest->triangleCount > 0;
    resident.state = ResidentObjectGpuResources::ResidencyState::Resident;
    resident.lastStateChange = std::chrono::steady_clock::now();
  } else if (buildRequest->objectIndex < _instanceRecords.size()) {
    transitionResidentToCold(buildRequest->objectIndex, resident,
                             _instanceRecords[buildRequest->objectIndex]);
  } else {
    resident.geometryValid = false;
  }

  buildRequest->releaseResources();

  std::printf(
      "[BLAS] Build %s for object %zu complete. Scratch pool in-flight=%zu "
      "bytes, available=%zu bytes (created=%zu, reused=%zu).\n",
      success ? "succeeded" : "failed", buildRequest->objectIndex,
      _blasScratchPoolInUseBytes, _blasScratchPoolAvailableBytes,
      _blasScratchPoolCreatedCount, _blasScratchPoolReusedCount);

  auto it = std::find(_activeBlasBuilds.begin(), _activeBlasBuilds.end(),
                      buildRequest);
  if (it != _activeBlasBuilds.end())
    _activeBlasBuilds.erase(it);

  processBlasBuildQueue();
}

bool Renderer::transitionResidentToCold(size_t objectIndex,
                                        ResidentObjectGpuResources &resident,
                                        BlasInstanceRecord &instanceRecord,
                                        bool removePending) {
  std::chrono::steady_clock::time_point frameWaitSnapshot;
  if (!waitForPendingFrameCommands(kFrameCommandBufferWaitTimeout,
                                   &frameWaitSnapshot))
    return false;

  waitForPendingTlasBuild();

  std::unique_lock<std::mutex> lock(_frameCommandBufferMutex);
  auto newerSubmission = std::find_if(
      _frameCommandBuffers.begin(), _frameCommandBuffers.end(),
      [frameWaitSnapshot](const FrameCommandBufferRecord &record) {
        return record.trackedSince >= frameWaitSnapshot;
      });
  if (newerSubmission != _frameCommandBuffers.end()) {
    lock.unlock();
    return false;
  }

  resident.transitionToCold(instanceRecord);
  lock.unlock();

  if (removePending)
    _pendingBlasEvictions.cancel(objectIndex, resident);

  return true;
}

void Renderer::requestResidentEviction(size_t objectIndex,
                                       ResidentObjectGpuResources &resident,
                                       BlasInstanceRecord &instanceRecord) {
  if (objectIndex >= _residentObjectGpuResources.size())
    return;

  bool hasPendingCommand = resident.hasPendingCommands();
  if (!resident.isResident() && !hasPendingCommand) {
    transitionResidentToCold(objectIndex, resident, instanceRecord);
    return;
  }

  resident.transitionToStreaming();
  resident.geometryValid = false;
  _pendingBlasEvictions.enqueue(objectIndex, resident);
}

void Renderer::cancelPendingResidentEviction(size_t objectIndex,
                                             ResidentObjectGpuResources &resident) {
  _pendingBlasEvictions.cancel(objectIndex, resident);
}

void Renderer::handleDeferredBlasEvictions(MTL::CommandBuffer *commandBuffer,
                                           bool success) {
  if (!commandBuffer)
    return;

  std::chrono::steady_clock::time_point frameWaitSnapshot;
  bool ready =
      success && waitForPendingFrameCommands(kFrameCommandBufferWaitTimeout,
                                             &frameWaitSnapshot);

  _pendingBlasEvictions.complete(
      commandBuffer, success && ready,
      [this](ResidentObjectGpuResources &resident, size_t objectIndex) {
        if (objectIndex >= _instanceRecords.size())
          return;
        auto &instanceRecord = _instanceRecords[objectIndex];
        if (!transitionResidentToCold(objectIndex, resident, instanceRecord,
                                      false)) {
          _pendingBlasEvictions.enqueue(objectIndex, resident);
        }
      });
}

bool Renderer::submitAsyncCommandBuffer(
    MTL::CommandBuffer *commandBuffer,
    std::function<void(bool)> completion) {
  if (!commandBuffer)
    return false;

  if (completion) {
    commandBuffer->addCompletedHandler(
        [completion = std::move(completion)](MTL::CommandBuffer *cmd) {
          bool success =
              cmd->status() == MTL::CommandBufferStatusCompleted;
          auto completionCopy = completion;
          dispatch_async(dispatch_get_main_queue(), ^{
            if (completionCopy)
              completionCopy(success);
          });
        });
  }

  commandBuffer->commit();
  return true;
}

bool Renderer::ensureDummyBlas() {
  if (_pDummyBlas)
    return true;

  if (!_pDevice || !_pCommandQueue)
    return false;

  if (_dummyBlasBuildInFlight)
    return false;

  _dummyBlasResources.initialize(_pDevice);

  MTL::AccelerationStructureTriangleGeometryDescriptor *geometryDesc =
      MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
  geometryDesc->setOpaque(true);
  const NS::UInteger dummyVertexSize =
      static_cast<NS::UInteger>(sizeof(simd::float3) * 3);
  const NS::UInteger dummyIndexSize =
      static_cast<NS::UInteger>(sizeof(uint32_t) * 3);
  MTL::Buffer *dummyVertexBuffer = _dummyBlasResources.ensureVertexBuffer(
      dummyVertexSize, "DummyBLASVertices");
  MTL::Buffer *dummyIndexBuffer = _dummyBlasResources.ensureIndexBuffer(
      dummyIndexSize, "DummyBLASIndices");
  if (!dummyVertexBuffer || !dummyIndexBuffer) {
    geometryDesc->release();
    return false;
  }
  geometryDesc->setVertexStride(sizeof(simd::float3));
  geometryDesc->setVertexFormat(MTL::AttributeFormat::AttributeFormatFloat3);
  geometryDesc->setIndexType(MTL::IndexType::IndexTypeUInt32);
  geometryDesc->setTriangleCount(0);
  geometryDesc->setVertexBuffer(dummyVertexBuffer);
  geometryDesc->setVertexBufferOffset(0);
  geometryDesc->setIndexBuffer(dummyIndexBuffer);
  geometryDesc->setIndexBufferOffset(0);

  struct ZeroRequest {
    MTL::Buffer *target = nullptr;
    MTL::Buffer *staging = nullptr;
    NS::UInteger size = 0;
  };

  auto releaseZeroRequest = [](ZeroRequest &request) {
    if (request.staging) {
      request.staging->release();
      request.staging = nullptr;
    }
    request.target = nullptr;
    request.size = 0;
  };

  ZeroRequest vertexZeroRequest;
  ZeroRequest indexZeroRequest;

  auto prepareZeroRequest = [&](MTL::Buffer *buffer, NS::UInteger size,
                                ZeroRequest &request) -> bool {
    if (!buffer || size == 0)
      return true;

    MTL::StorageMode storageMode = buffer->storageMode();
    if (storageMode == MTL::StorageMode::StorageModeShared ||
        storageMode == MTL::StorageMode::StorageModeManaged) {
      if (void *contents = buffer->contents())
        std::memset(contents, 0, size);
      return true;
    }

    MTL::Buffer *staging =
        _pDevice->newBuffer(size, MTL::ResourceStorageModeShared);
    if (!staging)
      return false;
    if (void *contents = staging->contents())
      std::memset(contents, 0, size);
    request.target = buffer;
    request.staging = staging;
    request.size = size;
    return true;
  };

  if (!prepareZeroRequest(dummyVertexBuffer, dummyVertexSize,
                          vertexZeroRequest) ||
      !prepareZeroRequest(dummyIndexBuffer, dummyIndexSize, indexZeroRequest)) {
    releaseZeroRequest(vertexZeroRequest);
    releaseZeroRequest(indexZeroRequest);
    geometryDesc->release();
    return false;
  }

  NS::Object *geometryObjects[] = {geometryDesc};
  NS::Array *geometryArray = NS::Array::alloc()->init(geometryObjects, 1);

  MTL::PrimitiveAccelerationStructureDescriptor *accelDesc =
      MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
  accelDesc->setGeometryDescriptors(geometryArray);

  auto releaseDescriptors = [&]() {
    if (geometryArray) {
      geometryArray->release();
      geometryArray = nullptr;
    }
    if (geometryDesc) {
      geometryDesc->release();
      geometryDesc = nullptr;
    }
    if (accelDesc) {
      accelDesc->release();
      accelDesc = nullptr;
    }
  };

  MTL::AccelerationStructureSizes sizes =
      _pDevice->accelerationStructureSizes(accelDesc);

  NS::UInteger alignedAccelerationSize =
      _dummyBlasResources.alignedHeapSize(sizes.accelerationStructureSize);
  NS::UInteger totalRequestedBytes =
      alignedAccelerationSize + dummyVertexSize + dummyIndexSize;
  constexpr NS::UInteger kDummyBlasMemoryBudgetBytes = 1024ull * 1024ull;
  static bool loggedDummyUsage = false;
  if (totalRequestedBytes > kDummyBlasMemoryBudgetBytes) {
    std::printf(
        "Dummy BLAS placeholder exceeded memory budget: %llu bytes used (limit %llu bytes).\n",
        static_cast<unsigned long long>(totalRequestedBytes),
        static_cast<unsigned long long>(kDummyBlasMemoryBudgetBytes));
  } else if (!loggedDummyUsage &&
             totalRequestedBytes > kDummyBlasMemoryBudgetBytes / 2) {
    std::printf(
        "Dummy BLAS placeholder using %llu / %llu bytes of budget.\n",
        static_cast<unsigned long long>(totalRequestedBytes),
        static_cast<unsigned long long>(kDummyBlasMemoryBudgetBytes));
    loggedDummyUsage = true;
  }
  assert(totalRequestedBytes <= kDummyBlasMemoryBudgetBytes &&
         "Dummy BLAS memory usage exceeds placeholder budget");

  MTL::AccelerationStructure *structure =
      _dummyBlasResources.ensureAccelerationStructure(
          sizes.accelerationStructureSize, "DummyBLAS");

  if (!structure) {
    releaseZeroRequest(vertexZeroRequest);
    releaseZeroRequest(indexZeroRequest);
    releaseDescriptors();
    return false;
  }

  NS::UInteger scratchSize = sizes.buildScratchBufferSize;
  MTL::Buffer *scratchBuffer = nullptr;
  if (scratchSize > 0)
    scratchBuffer =
        _pDevice->newBuffer(scratchSize, MTL::ResourceStorageModePrivate);

  MTL::CommandBuffer *commandBuffer = _pCommandQueue->commandBuffer();
  if (!commandBuffer) {
    if (scratchBuffer)
      scratchBuffer->release();
    releaseZeroRequest(vertexZeroRequest);
    releaseZeroRequest(indexZeroRequest);
    releaseDescriptors();
    return false;
  }

  if (vertexZeroRequest.staging || indexZeroRequest.staging) {
    MTL::BlitCommandEncoder *blitEncoder = commandBuffer->blitCommandEncoder();
    if (!blitEncoder) {
      if (scratchBuffer)
        scratchBuffer->release();
      releaseZeroRequest(vertexZeroRequest);
      releaseZeroRequest(indexZeroRequest);
      releaseDescriptors();
      return false;
    }
    if (vertexZeroRequest.staging) {
      blitEncoder->copyFromBuffer(vertexZeroRequest.staging, 0,
                                  vertexZeroRequest.target, 0,
                                  vertexZeroRequest.size);
    }
    if (indexZeroRequest.staging) {
      blitEncoder->copyFromBuffer(indexZeroRequest.staging, 0,
                                  indexZeroRequest.target, 0,
                                  indexZeroRequest.size);
    }
    blitEncoder->endEncoding();
  }

  MTL::AccelerationStructureCommandEncoder *encoder =
      commandBuffer->accelerationStructureCommandEncoder();
  if (!encoder) {
    if (scratchBuffer)
      scratchBuffer->release();
    releaseZeroRequest(vertexZeroRequest);
    releaseZeroRequest(indexZeroRequest);
    releaseDescriptors();
    return false;
  }

  encoder->buildAccelerationStructure(structure, accelDesc, scratchBuffer, 0);
  encoder->endEncoding();

  struct DummyBlasBuildContext {
    MTL::Buffer *vertexStaging = nullptr;
    MTL::Buffer *indexStaging = nullptr;
    MTL::Buffer *scratchBuffer = nullptr;
  };

  auto buildContext = std::make_shared<DummyBlasBuildContext>();
  buildContext->vertexStaging = vertexZeroRequest.staging;
  buildContext->indexStaging = indexZeroRequest.staging;
  buildContext->scratchBuffer = scratchBuffer;

  vertexZeroRequest.staging = nullptr;
  indexZeroRequest.staging = nullptr;
  scratchBuffer = nullptr;

  _dummyBlasBuildInFlight = true;

  auto completion = [this, context = buildContext, structure,
                     totalRequestedBytes](bool success) {
    if (context->vertexStaging) {
      context->vertexStaging->release();
      context->vertexStaging = nullptr;
    }
    if (context->indexStaging) {
      context->indexStaging->release();
      context->indexStaging = nullptr;
    }
    if (context->scratchBuffer) {
      context->scratchBuffer->release();
      context->scratchBuffer = nullptr;
    }

    if (success) {
      _pDummyBlas = structure;
      std::printf("Dummy BLAS placeholder ready (%llu bytes).\n",
                  static_cast<unsigned long long>(totalRequestedBytes));
    } else {
      std::printf(
          "Dummy BLAS build failed; releasing placeholder resources.\n");
      _dummyBlasResources.ensureAccelerationStructure(0, nullptr);
    }

    _dummyBlasBuildInFlight = false;
  };

  bool submitted = submitAsyncCommandBuffer(commandBuffer, completion);

  releaseDescriptors();

  if (!submitted) {
    _dummyBlasBuildInFlight = false;
    if (buildContext->vertexStaging) {
      buildContext->vertexStaging->release();
      buildContext->vertexStaging = nullptr;
    }
    if (buildContext->indexStaging) {
      buildContext->indexStaging->release();
      buildContext->indexStaging = nullptr;
    }
    if (buildContext->scratchBuffer) {
      buildContext->scratchBuffer->release();
      buildContext->scratchBuffer = nullptr;
    }
    return false;
  }

  return false;
}

void Renderer::ensureTlasBuildEvent() {
  if (_pTlasBuildEvent || !_pDevice)
    return;

  _pTlasBuildEvent = _pDevice->newSharedEvent();
  if (_pTlasBuildEvent) {
    _pTlasBuildEvent->setLabel(
        NS::String::string("RendererTLASFence", NS::UTF8StringEncoding));
    uint64_t initialValue = _pTlasBuildEvent->signaledValue();
    _tlasCompletedEventValue.store(initialValue, std::memory_order_relaxed);
  }
}

bool Renderer::hasPendingTlasBuild() const {
  if (!_pTlasBuildEvent || _tlasBuildEventValue == 0)
    return false;

  return _tlasCompletedEventValue.load(std::memory_order_acquire) <
         _tlasBuildEventValue;
}

void Renderer::waitForPendingTlasBuild() {
  if (!hasPendingTlasBuild())
    return;

  uint64_t targetValue = _tlasBuildEventValue;
  uint64_t completedValue =
      _tlasCompletedEventValue.load(std::memory_order_acquire);
  if (completedValue >= targetValue)
    return;

  uint64_t signaled = _pTlasBuildEvent->signaledValue();
  if (signaled >= targetValue) {
    _tlasCompletedEventValue.store(signaled, std::memory_order_release);
    return;
  }

  dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
  if (!semaphore) {
    while (_pTlasBuildEvent->signaledValue() < targetValue)
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    _tlasCompletedEventValue.store(targetValue, std::memory_order_release);
    return;
  }

  MTL::SharedEventListener *listener =
      MTL::SharedEventListener::alloc()->init(
          dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0));

  if (!listener) {
    dispatch_release(semaphore);
    while (_pTlasBuildEvent->signaledValue() < targetValue)
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    _tlasCompletedEventValue.store(targetValue, std::memory_order_release);
    return;
  }

  _pTlasBuildEvent->notifyListener(
      listener, targetValue,
      ^(MTL::SharedEvent *, uint64_t) { dispatch_semaphore_signal(semaphore); });

  dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

  listener->release();
  dispatch_release(semaphore);

  uint64_t latest = _pTlasBuildEvent->signaledValue();
  if (latest < targetValue)
    latest = targetValue;
  _tlasCompletedEventValue.store(latest, std::memory_order_release);

  finalizePendingTlasScratchResize(true);
}

void Renderer::finalizePendingTlasScratchResize(bool force) {
  if (!_tlasScratchTracker.hasPendingResize())
    return;

  bool ready = force;
  if (!ready) {
    uint64_t completed = _tlasCompletedEventValue.load(std::memory_order_acquire);
    ready = _tlasScratchTracker.ready(completed);
  }

  if (!ready)
    return;

  size_t previousCapacity = static_cast<size_t>(_tlasScratchCapacity);
  size_t desiredSize = _tlasScratchTracker.targetRefitSize();
  size_t retiredBytes = _tlasScratchTracker.retiredBytes();

  if (_pTlasScratchBuffer) {
    _pTlasScratchBuffer->release();
    _pTlasScratchBuffer = nullptr;
    _tlasScratchCapacity = 0;
  }

  if (desiredSize > 0 && _pDevice) {
    _pTlasScratchBuffer = _pDevice->newBuffer(
        static_cast<NS::UInteger>(desiredSize), MTL::ResourceStorageModePrivate);
    _tlasScratchCapacity =
        _pTlasScratchBuffer ? static_cast<NS::UInteger>(desiredSize)
                            : static_cast<NS::UInteger>(0);
  }

  _tlasScratchTracker.noteAllocation(_tlasScratchCapacity);
  _tlasScratchTracker.finalizeResize();
  updateTlasScratchResidentBytes(_tlasScratchCapacity);

  if (retiredBytes > 0) {
    std::printf("[TLAS] Shrunk scratch buffer from %zu to %zu bytes (retired %zu bytes).\n",
                previousCapacity, static_cast<size_t>(_tlasScratchCapacity), retiredBytes);
  }
}

void Renderer::updateTlasScratchResidentBytes(NS::UInteger bytes) {
  _tlasScratchResidentBytes = static_cast<size_t>(bytes);
  _residencyBudget.setTlasScratchBytes(_tlasScratchResidentBytes);
}

void Renderer::updateTopLevelAccelerationStructure(
    const std::vector<MTL::AccelerationStructureInstanceDescriptor> &descriptors,
    const std::vector<MTL::AccelerationStructure *> &structures) {
  if (!_pDevice || !_pCommandQueue)
    return;

  finalizePendingTlasScratchResize();

  if (!ensureDummyBlas())
    return;

  _tlasHeap.initialize(_pDevice);

  std::vector<MTL::AccelerationStructure *> instancedStructures;
  instancedStructures.reserve(structures.size() + 1);
  instancedStructures.push_back(_pDummyBlas);
  for (MTL::AccelerationStructure *structure : structures) {
    instancedStructures.push_back(structure ? structure : _pDummyBlas);
  }

  bool structureListChanged =
      instancedStructures.size() != _cachedInstancedAccelerationStructures.size();
  if (!structureListChanged) {
    for (size_t i = 0; i < instancedStructures.size(); ++i) {
      if (instancedStructures[i] != _cachedInstancedAccelerationStructures[i]) {
        structureListChanged = true;
        break;
      }
    }
  }

  bool descriptorCountChanged =
      descriptors.size() != _cachedInstanceDescriptors.size();
  bool descriptorContentChanged = descriptorCountChanged;
  if (!descriptorContentChanged && !descriptors.empty()) {
    descriptorContentChanged =
        std::memcmp(descriptors.data(), _cachedInstanceDescriptors.data(),
                    descriptors.size() *
                        sizeof(MTL::AccelerationStructureInstanceDescriptor)) != 0;
  }

  bool needsRebuild = structureListChanged || descriptorCountChanged ||
                      (_pTlasStructure == nullptr);
  bool needsDescriptorUpload = descriptorContentChanged || needsRebuild;

  size_t descriptorCount = descriptors.size();
  std::vector<RangeUpdate> descriptorRanges;
  bool uploadFullDescriptors = needsRebuild || descriptorCountChanged;
  if (needsDescriptorUpload && !uploadFullDescriptors && descriptorCount > 0 &&
      !_cachedInstanceDescriptors.empty()) {
    descriptorRanges = computeChangedRanges(
        _cachedInstanceDescriptors.data(), descriptors.data(),
        sizeof(MTL::AccelerationStructureInstanceDescriptor), descriptorCount);
    if (descriptorRanges.empty())
      needsDescriptorUpload = false;
  }

  size_t descriptorBytes =
      descriptorCount * sizeof(MTL::AccelerationStructureInstanceDescriptor);
  size_t uploadBytes =
      std::max<size_t>(descriptorCount, size_t(1)) *
      sizeof(MTL::AccelerationStructureInstanceDescriptor);

  MTL::Buffer *instanceBuffer = _tlasHeap.ensureVertexBuffer(
      static_cast<NS::UInteger>(uploadBytes),
      "TLASInstanceDescriptors", MTL::ResourceStorageModePrivate,
      static_cast<MTL::ResourceUsage>(MTL::ResourceUsageRead));
  if (!instanceBuffer)
    return;

  _pTlasInstanceDescriptorBuffer = instanceBuffer;

  MTL::Buffer *stagingBuffer = nullptr;
  if (needsDescriptorUpload && descriptorBytes > 0) {
    if (!_pTlasDescriptorStaging ||
        _tlasDescriptorStagingCapacity < descriptorBytes) {
      if (_pTlasDescriptorStaging)
        _pTlasDescriptorStaging->release();
      _pTlasDescriptorStaging =
          _pDevice->newBuffer(descriptorBytes, MTL::ResourceStorageModeShared);
      _tlasDescriptorStagingCapacity =
          _pTlasDescriptorStaging ? descriptorBytes : 0;
    }

    stagingBuffer = _pTlasDescriptorStaging;
    if (!stagingBuffer)
      return;

    uint8_t *dstBytes = static_cast<uint8_t *>(stagingBuffer->contents());
    const uint8_t *srcBytes =
        reinterpret_cast<const uint8_t *>(descriptors.data());

    if (uploadFullDescriptors || descriptorRanges.empty()) {
      std::memcpy(dstBytes, srcBytes, descriptorBytes);
      descriptorRanges.clear();
      descriptorRanges.push_back({0, descriptorBytes});
    } else {
      for (const RangeUpdate &range : descriptorRanges) {
        if (range.length == 0)
          continue;
        std::memcpy(dstBytes + range.offset, srcBytes + range.offset,
                    range.length);
      }
    }

    markBufferModified(stagingBuffer,
                       NS::Range::Make(0, static_cast<NS::UInteger>(descriptorBytes)));
  } else {
    descriptorRanges.clear();
  }

  MTL::InstanceAccelerationStructureDescriptor *instanceDesc =
      MTL::InstanceAccelerationStructureDescriptor::alloc()->init();
  instanceDesc->setInstanceCount(static_cast<NS::UInteger>(descriptorCount));
  instanceDesc->setInstanceDescriptorBuffer(instanceBuffer);
  instanceDesc->setInstanceDescriptorBufferOffset(0);
  instanceDesc->setInstanceDescriptorStride(
      sizeof(MTL::AccelerationStructureInstanceDescriptor));
  instanceDesc->setInstanceDescriptorType(
      MTL::AccelerationStructureInstanceDescriptorTypeDefault);

  if (!instancedStructures.empty()) {
    std::vector<NS::Object *> nsStructures(instancedStructures.size());
    for (size_t i = 0; i < instancedStructures.size(); ++i)
      nsStructures[i] = instancedStructures[i];
    NS::Array *structuresArray =
        NS::Array::alloc()->init(nsStructures.data(), nsStructures.size());
    instanceDesc->setInstancedAccelerationStructures(structuresArray);
    structuresArray->release();
  }

  MTL::AccelerationStructureSizes sizes =
      _pDevice->accelerationStructureSizes(instanceDesc);

  if (needsRebuild) {
    MTL::AccelerationStructure *structure =
        _tlasHeap.ensureAccelerationStructure(
            sizes.accelerationStructureSize, "SceneTLAS");
    if (!structure) {
      instanceDesc->release();
      return;
    }
    _pTlasStructure = structure;
  }

  NS::UInteger scratchSize = needsRebuild ? sizes.buildScratchBufferSize
                                          : sizes.refitScratchBufferSize;
  size_t buildScratchBytes = static_cast<size_t>(scratchSize);
  size_t refitScratchBytes =
      static_cast<size_t>(sizes.refitScratchBufferSize);
  MTL::Buffer *scratchBuffer = nullptr;
  if (scratchSize > 0) {
    if (!_pTlasScratchBuffer || _tlasScratchCapacity < scratchSize) {
      if (_pTlasScratchBuffer)
        _pTlasScratchBuffer->release();
      _pTlasScratchBuffer =
          _pDevice->newBuffer(scratchSize, MTL::ResourceStorageModePrivate);
      _tlasScratchCapacity =
          _pTlasScratchBuffer ? scratchSize : static_cast<NS::UInteger>(0);
    }
    scratchBuffer = _pTlasScratchBuffer;
    if (!scratchBuffer) {
      instanceDesc->release();
      return;
    }
  }

  _tlasScratchTracker.noteAllocation(_tlasScratchCapacity);
  updateTlasScratchResidentBytes(_tlasScratchCapacity);

  MTL::CommandBuffer *commandBuffer = _pCommandQueue->commandBuffer();
  if (!commandBuffer) {
    instanceDesc->release();
    return;
  }

  MTL::BlitCommandEncoder *blit = nullptr;
  auto ensureBlitEncoder = [&]() -> MTL::BlitCommandEncoder * {
    if (!blit)
      blit = commandBuffer->blitCommandEncoder();
    return blit;
  };

  if (stagingBuffer && !descriptorRanges.empty()) {
    MTL::BlitCommandEncoder *enc = ensureBlitEncoder();
    if (!enc) {
      instanceDesc->release();
      return;
    }
    for (const RangeUpdate &range : descriptorRanges) {
      if (range.length == 0)
        continue;
      enc->copyFromBuffer(stagingBuffer, static_cast<NS::UInteger>(range.offset),
                          instanceBuffer, static_cast<NS::UInteger>(range.offset),
                          static_cast<NS::UInteger>(range.length));
    }
  } else if (needsDescriptorUpload && descriptorBytes == 0 && instanceBuffer &&
             instanceBuffer->length() > 0) {
    MTL::BlitCommandEncoder *enc = ensureBlitEncoder();
    if (!enc) {
      instanceDesc->release();
      return;
    }
    enc->fillBuffer(instanceBuffer,
                    NS::Range::Make(0, instanceBuffer->length()), 0);
  }

  if (blit)
    blit->endEncoding();

  MTL::AccelerationStructureCommandEncoder *encoder =
      commandBuffer->accelerationStructureCommandEncoder();
  if (!encoder) {
    instanceDesc->release();
    return;
  }

  if (needsRebuild) {
    encoder->buildAccelerationStructure(_pTlasStructure, instanceDesc,
                                        scratchBuffer, 0);
  } else {
    encoder->refitAccelerationStructure(_pTlasStructure, instanceDesc,
                                        _pTlasStructure, scratchBuffer, 0);
  }
  encoder->endEncoding();

  ensureTlasBuildEvent();
  uint64_t signalValue = 0;
  if (_pTlasBuildEvent) {
    signalValue = ++_tlasBuildEventValue;
    commandBuffer->encodeSignalEvent(_pTlasBuildEvent, signalValue);
  }

  if (needsRebuild)
    _tlasScratchTracker.registerRebuild(signalValue, buildScratchBytes,
                                        refitScratchBytes);

  if (signalValue > 0) {
    commandBuffer->addCompletedHandler(
        [this, signalValue](MTL::CommandBuffer *cmd) {
          bool success =
              cmd->status() == MTL::CommandBufferStatusCompleted;
          this->handleDeferredBlasEvictions(cmd, success);
          this->_tlasCompletedEventValue.store(signalValue,
                                               std::memory_order_release);
        });
  } else {
    commandBuffer->addCompletedHandler(
        [this](MTL::CommandBuffer *cmd) {
          bool success =
              cmd->status() == MTL::CommandBufferStatusCompleted;
          this->handleDeferredBlasEvictions(cmd, success);
          this->finalizePendingTlasScratchResize(true);
        });
  }

  _pendingBlasEvictions.assign(commandBuffer);

  commandBuffer->commit();

  instanceDesc->release();

  _cachedInstanceDescriptors = descriptors;
  _cachedInstancedAccelerationStructures = instancedStructures;
}

// Repack GPU-facing caches to match the most recent CPU residency state.
// This routine refreshes the primitive/material buffers, BLAS/TLAS bindings,
// light sampling tables, active masks and any compaction remap tables.  When
// possible it reuses cached data so only modified ranges are uploaded.
void Renderer::rebuildResidentResources(bool forceFullRebuild) {
  size_t totalPrimitiveCount = _allPrimitives.size();
  size_t cachedTotalPrimitiveCount = _cachedTotalPrimitiveCount;

  if (forceFullRebuild) {
    bool startCompacted = _pScene ? _pScene->getStartCompacted() : false;
    _residentCompacted = startCompacted;
    _compactionCooldown = 0;
  }

  const size_t uniformsDataSize = sizeof(UniformsData);
  if (!_pUniformsBuffer) {
    _pUniformsBuffer =
        _pDevice->newBuffer(uniformsDataSize, MTL::ResourceStorageModeManaged);
    if (_pUniformsBuffer)
      markBufferModified(_pUniformsBuffer,
                         NS::Range::Make(0, uniformsDataSize));
  }

  if (_primitiveToResidentIndex.size() < totalPrimitiveCount)
    _primitiveToResidentIndex.resize(totalPrimitiveCount, -1);
  std::fill(_primitiveToResidentIndex.begin(), _primitiveToResidentIndex.end(),
            -1);

  bool sizeChanged = cachedTotalPrimitiveCount != totalPrimitiveCount;
  bool needFullUpload =
      forceFullRebuild || !_residentBuffersInitialized || sizeChanged;

  const auto &sceneObjects = _pScene->getObjects();
  if (_objectResidentState.size() != sceneObjects.size()) {
    _objectResidentState.assign(sceneObjects.size(), false);
    needFullUpload = true;
  }

  std::vector<std::pair<size_t, size_t>> dirtyPrimitiveRanges;

  if (needFullUpload) {
    _cachedPrimitiveData.assign(totalPrimitiveCount * kPrimitiveFloat4Count,
                                simd::float4{0.0f, 0.0f, 0.0f, 0.0f});
    _cachedMaterialData.assign(totalPrimitiveCount * kMaterialFloat4Count,
                               simd::float4{0.0f, 0.0f, 0.0f, 0.0f});

    parallelFor(totalPrimitiveCount, [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        const Primitive &p = _allPrimitives[i];
        simd::float4 *primBase =
            &_cachedPrimitiveData[kPrimitiveFloat4Count * i];
        simd::float4 *matBase =
            &_cachedMaterialData[kMaterialFloat4Count * i];

        primBase[4] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        primBase[5] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        primBase[6] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};

        switch (p.type) {
        case PrimitiveType::Sphere: {
          primBase[0] =
              simd::make_float4(p.sphere.center, static_cast<float>(p.type));
          primBase[1] = simd::make_float4(
              simd::make_float3(p.sphere.radius, 0.0f, 0.0f), 0.0f);
          primBase[2] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          primBase[3] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          primBase[6] = simd::make_float4(0.0f, 0.0f, 1.0f, 0.0f);
          break;
        }
        case PrimitiveType::Rectangle: {
          primBase[0] =
              simd::make_float4(p.rectangle.center, static_cast<float>(p.type));
          primBase[1] = simd::make_float4(p.rectangle.u, 0.0f);
          primBase[2] = simd::make_float4(p.rectangle.v, 0.0f);
          primBase[3] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          primBase[4] = simd::make_float4(p.rectangle.tangent, 0.0f);
          primBase[5] = simd::make_float4(p.rectangle.bitangent, 0.0f);
          primBase[6] = simd::make_float4(p.rectangle.normal,
                                          p.rectangle.supportsNormalMap ? 1.0f
                                                                        : 0.0f);
          break;
        }
        case PrimitiveType::Triangle: {
          primBase[0] =
              simd::make_float4(p.triangle.v0, static_cast<float>(p.type));
          primBase[1] = simd::make_float4(p.triangle.v1, p.triangle.uv0.x);
          primBase[2] = simd::make_float4(p.triangle.v2, p.triangle.uv0.y);
          primBase[3] = simd::make_float4(p.triangle.uv1.x, p.triangle.uv1.y,
                                          p.triangle.uv2.x, p.triangle.uv2.y);
          primBase[4] = simd::make_float4(p.triangle.tangent, 0.0f);
          primBase[5] = simd::make_float4(p.triangle.bitangent, 0.0f);
          primBase[6] = simd::make_float4(p.triangle.normal, 1.0f);
          break;
        }
        }

        const Material &m = p.material;
        auto packed = encodeMaterial(m);
        for (size_t j = 0; j < kMaterialFloat4Count; ++j) {
          matBase[j] = packed[j];
        }
      }
    }, primitivePackingConfig(totalPrimitiveCount));

    _cachedTextureInfos.clear();
    _cachedTextureData.clear();
    if (_pScene) {
      const auto &sceneTextures = _pScene->getTextures();
      const size_t textureCount = sceneTextures.size();
      std::vector<size_t> textureOffsets(textureCount, 0);
      size_t totalTexelCount = 0;
      for (size_t texIndex = 0; texIndex < textureCount; ++texIndex) {
        textureOffsets[texIndex] = totalTexelCount;
        totalTexelCount += static_cast<size_t>(sceneTextures[texIndex].width) *
                           sceneTextures[texIndex].height;
      }

      _cachedTextureInfos.resize(textureCount);
      _cachedTextureData.resize(totalTexelCount);

      parallelFor(textureCount, [&](size_t begin, size_t end) {
        for (size_t texIndex = begin; texIndex < end; ++texIndex) {
          const auto &tex = sceneTextures[texIndex];
          TextureInfo info{};
          info.offset = static_cast<uint32_t>(textureOffsets[texIndex]);
          info.width = tex.width;
          info.height = tex.height;
          _cachedTextureInfos[texIndex] = info;

          size_t texelCount =
              static_cast<size_t>(tex.width) * static_cast<size_t>(tex.height);
          size_t base = textureOffsets[texIndex];
          simd::float4 *dst = _cachedTextureData.data() + base;

          for (size_t t = 0; t < texelCount; ++t) {
            size_t idx = t * 4;
            float r = (idx < tex.pixels.size()) ? tex.pixels[idx + 0] : 0.0f;
            float g = (idx + 1 < tex.pixels.size()) ? tex.pixels[idx + 1] : 0.0f;
            float b = (idx + 2 < tex.pixels.size()) ? tex.pixels[idx + 2] : 0.0f;
            float a = (idx + 3 < tex.pixels.size()) ? tex.pixels[idx + 3] : 1.0f;
            dst[t] = simd::make_float4(r, g, b, a);
          }
        }
      }, textureUploadConfig(textureCount, totalTexelCount));
    }

    rebuildMaterialTextures();
    rebuildEnvironmentTexture();

    _pScene->createTriangleBuffers(_cachedTriangleVertices,
                                   _cachedTriangleIndices);

    const auto &sceneIndices = _pScene->getPrimitiveIndices();
    _cachedPrimitiveIndices.resize(sceneIndices.size());
    for (size_t i = 0; i < sceneIndices.size(); ++i)
      _cachedPrimitiveIndices[i] = static_cast<int>(sceneIndices[i]);

    _cachedBVHNodes.clear();
    _blasNodeCount = _pScene ? _pScene->getBVHNodeCount() : 0;
    if (_blasNodeCount > 0) {
      simd::float4 *bvhRaw = _pScene->createBVHBuffer();
      if (bvhRaw) {
        _cachedBVHNodes.assign(bvhRaw, bvhRaw + _blasNodeCount * 2);
        delete[] bvhRaw;
      }
    }

    _cachedTLASNodes.clear();
    size_t tlasCount = 0;
    if (totalPrimitiveCount > 0) {
      simd::float4 *tlasRaw = _pScene->createTLASBuffer(tlasCount);
      if (tlasRaw) {
        _cachedTLASNodes.assign(tlasRaw, tlasRaw + tlasCount * 2);
        delete[] tlasRaw;
      }
    }
    _tlasNodeCount = tlasCount;
    _cachedTotalPrimitiveCount = totalPrimitiveCount;
  }

  _instanceRecords.resize(sceneObjects.size());
  std::vector<bool> objectShouldBeResident(sceneObjects.size(), false);

  if (_cpuActiveMask.size() < totalPrimitiveCount)
    _cpuActiveMask.resize(totalPrimitiveCount, 0);
  if (needFullUpload)
    std::fill(_cpuActiveMask.begin(), _cpuActiveMask.end(), 0);

  auto deduplicate = [](std::vector<size_t> &values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
  };

  if (needFullUpload) {
    _dirtyResidentObjects.clear();
  } else {
    deduplicate(_recentlyActivated);
    deduplicate(_recentlyDeactivated);
    deduplicate(_dirtyResidentObjects);
    if (!_recentlyActivated.empty() && !_recentlyDeactivated.empty()) {
      for (size_t idx : _recentlyDeactivated) {
        auto it = std::lower_bound(_recentlyActivated.begin(),
                                   _recentlyActivated.end(), idx);
        if (it != _recentlyActivated.end() && *it == idx)
          _recentlyActivated.erase(it);
      }
    }
  }

  if (!needFullUpload) {
    auto addDirtyRange = [&](size_t start, size_t count) {
      if (count == 0 || start >= totalPrimitiveCount)
        return;
      size_t clampedCount = std::min(count, totalPrimitiveCount - start);
      dirtyPrimitiveRanges.emplace_back(start, clampedCount);
    };

    for (size_t idx : _recentlyActivated)
      addDirtyRange(idx, 1);
    for (size_t idx : _recentlyDeactivated)
      addDirtyRange(idx, 1);
    for (size_t objectIndex : _dirtyResidentObjects) {
      if (objectIndex >= sceneObjects.size())
        continue;
      const SceneObject &obj = sceneObjects[objectIndex];
      addDirtyRange(obj.firstPrimitive, obj.primitiveCount);
    }

    if (!dirtyPrimitiveRanges.empty()) {
      std::sort(dirtyPrimitiveRanges.begin(), dirtyPrimitiveRanges.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });
      std::vector<std::pair<size_t, size_t>> mergedRanges;
      mergedRanges.reserve(dirtyPrimitiveRanges.size());
      for (const auto &range : dirtyPrimitiveRanges) {
        if (mergedRanges.empty()) {
          mergedRanges.push_back(range);
          continue;
        }
        auto &back = mergedRanges.back();
        size_t backEnd = back.first + back.second;
        if (range.first <= backEnd) {
          size_t newEnd = std::max(backEnd, range.first + range.second);
          back.second = newEnd - back.first;
        } else {
          mergedRanges.push_back(range);
        }
      }
      dirtyPrimitiveRanges.swap(mergedRanges);
    }
  }

  if (!needFullUpload && !dirtyPrimitiveRanges.empty()) {
    auto packPrimitiveRange = [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        if (i >= _allPrimitives.size())
          break;
        const Primitive &p = _allPrimitives[i];
        simd::float4 *primBase =
            &_cachedPrimitiveData[kPrimitiveFloat4Count * i];
        simd::float4 *matBase = &_cachedMaterialData[kMaterialFloat4Count * i];

        primBase[4] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        primBase[5] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        primBase[6] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};

        switch (p.type) {
        case PrimitiveType::Sphere: {
          primBase[0] =
              simd::make_float4(p.sphere.center, static_cast<float>(p.type));
          primBase[1] = simd::make_float4(
              simd::make_float3(p.sphere.radius, 0.0f, 0.0f), 0.0f);
          primBase[2] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          primBase[3] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          primBase[6] = simd::make_float4(0.0f, 0.0f, 1.0f, 0.0f);
          break;
        }
        case PrimitiveType::Rectangle: {
          primBase[0] =
              simd::make_float4(p.rectangle.center, static_cast<float>(p.type));
          primBase[1] = simd::make_float4(p.rectangle.u, 0.0f);
          primBase[2] = simd::make_float4(p.rectangle.v, 0.0f);
          primBase[3] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          primBase[4] = simd::make_float4(p.rectangle.tangent, 0.0f);
          primBase[5] = simd::make_float4(p.rectangle.bitangent, 0.0f);
          primBase[6] = simd::make_float4(p.rectangle.normal,
                                          p.rectangle.supportsNormalMap ? 1.0f
                                                                        : 0.0f);
          break;
        }
        case PrimitiveType::Triangle: {
          primBase[0] =
              simd::make_float4(p.triangle.v0, static_cast<float>(p.type));
          primBase[1] = simd::make_float4(p.triangle.v1, p.triangle.uv0.x);
          primBase[2] = simd::make_float4(p.triangle.v2, p.triangle.uv0.y);
          primBase[3] = simd::make_float4(p.triangle.uv1.x, p.triangle.uv1.y,
                                          p.triangle.uv2.x, p.triangle.uv2.y);
          primBase[4] = simd::make_float4(p.triangle.tangent, 0.0f);
          primBase[5] = simd::make_float4(p.triangle.bitangent, 0.0f);
          primBase[6] = simd::make_float4(p.triangle.normal, 1.0f);
          break;
        }
        }

        const Material &m = p.material;
        auto packed = encodeMaterial(m);
        for (size_t j = 0; j < kMaterialFloat4Count; ++j) {
          matBase[j] = packed[j];
        }
      }
    };

    for (const auto &range : dirtyPrimitiveRanges) {
      parallelFor(range.second,
                  [&](size_t begin, size_t end) {
                    packPrimitiveRange(range.first + begin,
                                       range.first + end);
                  },
                  primitivePackingConfig(range.second));
    }
  }

  _cachedLightIndices.clear();
  _cachedLightCdf.clear();
  float totalLightWeight = 0.0f;
  std::vector<size_t> activeIndices;
  activeIndices.reserve(totalPrimitiveCount);
  size_t activeTriangleCount = 0;

  for (size_t i = 0; i < totalPrimitiveCount; ++i) {
    bool active = i < _activePrimitive.size() && _activePrimitive[i];
    if (active) {
      activeIndices.push_back(i);
      if (_allPrimitives[i].type == PrimitiveType::Triangle)
        ++activeTriangleCount;

      const Material &m = _allPrimitives[i].material;
      float emissionStrength = m.emissionPower * luminance(m.emissionColor);
      if (emissionStrength > 0.0f) {
        float area = primitiveArea(_allPrimitives[i]);
        if (area > 0.0f) {
          float weight = area * emissionStrength;
          if (weight > 0.0f) {
            totalLightWeight += weight;
            _cachedLightIndices.push_back(static_cast<uint32_t>(i));
            _cachedLightCdf.push_back(totalLightWeight);
          }
        }
      }
    }

    if (needFullUpload || _residentCompacted)
      _cpuActiveMask[i] = active ? 1 : 0;
  }

  _activePrimitiveCount = activeIndices.size();
  _activeTriangleCount = activeTriangleCount;
  _lightCount = _cachedLightIndices.size();
  _lightTotalWeight = totalLightWeight;

  float activeRatio = (totalPrimitiveCount > 0)
                          ? static_cast<float>(_activePrimitiveCount) /
                                static_cast<float>(totalPrimitiveCount)
                          : 1.0f;
  activeRatio = std::clamp(activeRatio, 0.0f, 1.0f);
  _lastActivePrimitiveRatio = activeRatio;

  constexpr float kCompactionEnterRatio = 0.45f;
  constexpr float kCompactionExitRatio = 0.7f;
  constexpr uint32_t kCompactionCooldownFrames = 30;

  bool useCompaction = _residentCompacted;
  if (!_residentCompacted) {
    if (_compactionCooldown == 0 && totalPrimitiveCount > 0 &&
        _activePrimitiveCount < totalPrimitiveCount) {
      float occupancy = static_cast<float>(_activePrimitiveCount) /
                        static_cast<float>(totalPrimitiveCount);
      if (_activePrimitiveCount == 0 || occupancy <= kCompactionEnterRatio)
        useCompaction = true;
    }
  } else {
    float occupancy = (totalPrimitiveCount > 0)
                          ? static_cast<float>(_activePrimitiveCount) /
                                static_cast<float>(totalPrimitiveCount)
                          : 1.0f;
    if (_activePrimitiveCount == 0)
      useCompaction = true;
    else if (_activePrimitiveCount == totalPrimitiveCount ||
             occupancy >= kCompactionExitRatio)
      useCompaction = false;
  }

  bool compactionStateChanged = (useCompaction != _residentCompacted);
  if (compactionStateChanged) {
    _residentCompacted = useCompaction;
    _compactionCooldown = kCompactionCooldownFrames;
  }

  float shrinkTarget = std::clamp(_residencyConfig.bufferShrinkActiveRatio, 0.0f, 1.0f);
  bool occupancyShrinkActive = _residencyConfig.enableBufferShrink &&
                               activeRatio <= shrinkTarget &&
                               totalPrimitiveCount > 0;

  bool allowShrink = useCompaction;
  const char *shrinkContext = nullptr;
  if (allowShrink)
    shrinkContext = occupancyShrinkActive ? "compaction+low-occupancy"
                                          : "compaction";
  else if (occupancyShrinkActive) {
    allowShrink = true;
    shrinkContext = "low-occupancy";
  }

  std::vector<uint32_t> remapUpload;
  std::vector<uint8_t> compactActiveMask;
  std::vector<simd::float4> compactPrimitiveData;
  std::vector<simd::float4> compactMaterialData;
  std::vector<int> compactPrimitiveIndices;
  std::vector<simd::float4> compactBVHNodes;
  std::vector<simd::float3> compactTriangleVertices;
  std::vector<simd::uint3> compactTriangleIndices;

  const std::vector<simd::float4> *primitiveSource = &_cachedPrimitiveData;
  const std::vector<simd::float4> *materialSource = &_cachedMaterialData;
  const std::vector<int> *primitiveIndexSource = &_cachedPrimitiveIndices;
  const std::vector<simd::float4> *bvhSource = &_cachedBVHNodes;
  const std::vector<simd::float4> *tlasSource = &_cachedTLASNodes;
  const std::vector<simd::float3> *triangleVertexSource =
      &_cachedTriangleVertices;
  const std::vector<simd::uint3> *triangleIndexSource =
      &_cachedTriangleIndices;
  if (!useCompaction) {
    remapUpload.resize(totalPrimitiveCount);
    for (size_t i = 0; i < totalPrimitiveCount; ++i) {
      remapUpload[i] = static_cast<uint32_t>(i);
      if (i < _primitiveToResidentIndex.size())
        _primitiveToResidentIndex[i] = static_cast<int32_t>(i);
    }
    _residentPrimitiveCount = totalPrimitiveCount;
    _residentTriangleCount = _cachedTriangleIndices.size();
    _blasNodeCount = _cachedBVHNodes.size() / 2;
    _tlasNodeCount = _cachedTLASNodes.size() / 2;

    if (needFullUpload) {
      for (size_t objectIndex = 0; objectIndex < sceneObjects.size();
           ++objectIndex) {
        const SceneObject &obj = sceneObjects[objectIndex];
        BlasInstanceRecord record{};
        record.primitiveBase = static_cast<uint32_t>(obj.firstPrimitive);
        record.primitiveCount = static_cast<uint32_t>(obj.primitiveCount);
        record.primitiveIndexBase = static_cast<uint32_t>(obj.firstPrimitive);
        bool anyActive = false;
        size_t first = obj.firstPrimitive;
        size_t last = first + obj.primitiveCount;
        for (size_t prim = first;
             prim < last && prim < _activePrimitive.size(); ++prim) {
          if (_activePrimitive[prim]) {
            anyActive = true;
            break;
          }
        }
        record.blasRootIndex = anyActive ? obj.blasRootIndex : -1;
        objectShouldBeResident[objectIndex] = anyActive;
        _instanceRecords[objectIndex] = record;
      }
    } else {
      for (size_t objectIndex = 0; objectIndex < sceneObjects.size();
           ++objectIndex) {
        const SceneObject &obj = sceneObjects[objectIndex];
        BlasInstanceRecord record{};
        record.primitiveBase = static_cast<uint32_t>(obj.firstPrimitive);
        record.primitiveCount = static_cast<uint32_t>(obj.primitiveCount);
        record.primitiveIndexBase = static_cast<uint32_t>(obj.firstPrimitive);
        bool cachedResident =
            (objectIndex < _objectResidentState.size())
                ? _objectResidentState[objectIndex]
                : false;
        record.blasRootIndex = cachedResident ? obj.blasRootIndex : -1;
        _instanceRecords[objectIndex] = record;
        objectShouldBeResident[objectIndex] = cachedResident;
      }

      for (size_t dirtyIndex : _dirtyResidentObjects) {
        if (dirtyIndex >= sceneObjects.size())
          continue;
        const SceneObject &obj = sceneObjects[dirtyIndex];
        bool anyActive =
            (dirtyIndex < _objectActive.size()) ? _objectActive[dirtyIndex]
                                                : false;
        BlasInstanceRecord &record = _instanceRecords[dirtyIndex];
        record.primitiveBase = static_cast<uint32_t>(obj.firstPrimitive);
        record.primitiveCount = static_cast<uint32_t>(obj.primitiveCount);
        record.primitiveIndexBase = static_cast<uint32_t>(obj.firstPrimitive);
        record.blasRootIndex = anyActive ? obj.blasRootIndex : -1;
        objectShouldBeResident[dirtyIndex] = anyActive;
      }
    }
  } else {
    remapUpload.clear();
    compactPrimitiveData.clear();
    compactMaterialData.clear();
    compactPrimitiveIndices.clear();
    compactBVHNodes.clear();
    compactTriangleVertices.clear();
    compactTriangleIndices.clear();
    compactActiveMask.clear();

    compactPrimitiveData.reserve(_activePrimitiveCount *
                                 kPrimitiveFloat4Count);
    compactMaterialData.reserve(_activePrimitiveCount * kMaterialFloat4Count);
    compactPrimitiveIndices.reserve(_activePrimitiveCount);
    compactActiveMask.reserve(_activePrimitiveCount);

    std::vector<int32_t> cachedRemap;
    std::vector<size_t> activeLocalPrims;
    std::vector<Primitive> subset;
    std::vector<int> localPrimitiveIndices;
    std::vector<BVHNode> localNodes;

    size_t blasCacheReused = 0;
    size_t blasCacheFallback = 0;
    size_t blasCacheSkipped = 0;

    for (size_t objectIndex = 0; objectIndex < sceneObjects.size();
         ++objectIndex) {
      const SceneObject &obj = sceneObjects[objectIndex];
      BlasInstanceRecord record{};
      record.blasRootIndex = -1;
      record.primitiveBase = static_cast<uint32_t>(remapUpload.size());
      record.primitiveIndexBase =
          static_cast<uint32_t>(compactPrimitiveIndices.size());

      bool hasCache = !obj.cachedBlasNodes.empty() &&
                      !obj.cachedPrimitiveIndices.empty() &&
                      obj.cachedBlasRootIndex >= 0;

      if (hasCache) {
        size_t remapStart = remapUpload.size();
        size_t compactPrimitiveDataStart = compactPrimitiveData.size();
        size_t compactMaterialDataStart = compactMaterialData.size();
        size_t compactPrimitiveIndicesStart = compactPrimitiveIndices.size();
        size_t compactActiveMaskStart = compactActiveMask.size();
        size_t compactBVHNodesStart = compactBVHNodes.size();
        size_t compactTriangleVerticesStart = compactTriangleVertices.size();
        size_t compactTriangleIndicesStart = compactTriangleIndices.size();

        std::vector<std::pair<size_t, int32_t>> residentIndexEdits;
        residentIndexEdits.reserve(obj.cachedPrimitiveIndices.size());

        cachedRemap.assign(obj.cachedPrimitiveIndices.size(), -1);
        size_t activeCount = 0;
        for (size_t localIdx = 0; localIdx < obj.cachedPrimitiveIndices.size();
             ++localIdx) {
          size_t globalIndex = obj.cachedPrimitiveIndices[localIdx];
          if (globalIndex >= _activePrimitive.size() ||
              !_activePrimitive[globalIndex])
            continue;

          cachedRemap[localIdx] = static_cast<int32_t>(activeCount);
          remapUpload.push_back(static_cast<uint32_t>(globalIndex));
          if (globalIndex < _primitiveToResidentIndex.size()) {
            residentIndexEdits.emplace_back(globalIndex,
                                            _primitiveToResidentIndex[globalIndex]);
            _primitiveToResidentIndex[globalIndex] =
                static_cast<int32_t>(record.primitiveBase + activeCount);
          }

          const Primitive &p = _allPrimitives[globalIndex];
          simd::float4 prim0;
          simd::float4 prim1;
          simd::float4 prim2;
          simd::float4 prim3 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          simd::float4 prim4 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          simd::float4 prim5 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          simd::float4 prim6 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          switch (p.type) {
          case PrimitiveType::Sphere: {
            prim0 =
                simd::make_float4(p.sphere.center, static_cast<float>(p.type));
            prim1 = simd::make_float4(
                simd::make_float3(p.sphere.radius, 0.0f, 0.0f), 0.0f);
            prim2 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
            prim6 = simd::make_float4(0.0f, 0.0f, 1.0f, 0.0f);
            break;
          }
          case PrimitiveType::Rectangle: {
            prim0 = simd::make_float4(p.rectangle.center,
                                      static_cast<float>(p.type));
            prim1 = simd::make_float4(p.rectangle.u, 0.0f);
            prim2 = simd::make_float4(p.rectangle.v, 0.0f);
            prim4 = simd::make_float4(p.rectangle.tangent, 0.0f);
            prim5 = simd::make_float4(p.rectangle.bitangent, 0.0f);
            prim6 = simd::make_float4(p.rectangle.normal,
                                      p.rectangle.supportsNormalMap ? 1.0f
                                                                    : 0.0f);
            break;
          }
          case PrimitiveType::Triangle: {
            prim0 =
                simd::make_float4(p.triangle.v0, static_cast<float>(p.type));
            prim1 = simd::make_float4(p.triangle.v1, p.triangle.uv0.x);
            prim2 = simd::make_float4(p.triangle.v2, p.triangle.uv0.y);
            prim3 = simd::make_float4(p.triangle.uv1.x, p.triangle.uv1.y,
                                      p.triangle.uv2.x, p.triangle.uv2.y);
            prim4 = simd::make_float4(p.triangle.tangent, 0.0f);
            prim5 = simd::make_float4(p.triangle.bitangent, 0.0f);
            prim6 = simd::make_float4(p.triangle.normal, 1.0f);
            size_t baseVertex = compactTriangleVertices.size();
            compactTriangleVertices.push_back(p.triangle.v0);
            compactTriangleVertices.push_back(p.triangle.v1);
            compactTriangleVertices.push_back(p.triangle.v2);
            compactTriangleIndices.push_back(simd::make_uint3(
                static_cast<uint32_t>(baseVertex),
                static_cast<uint32_t>(baseVertex + 1),
                static_cast<uint32_t>(baseVertex + 2)));
            break;
          }
          }

          compactPrimitiveData.push_back(prim0);
          compactPrimitiveData.push_back(prim1);
          compactPrimitiveData.push_back(prim2);
          compactPrimitiveData.push_back(prim3);
          compactPrimitiveData.push_back(prim4);
          compactPrimitiveData.push_back(prim5);
          compactPrimitiveData.push_back(prim6);

          const Material &m = p.material;
          auto packedMaterial = encodeMaterial(m);
          for (size_t j = 0; j < kMaterialFloat4Count; ++j) {
            compactMaterialData.push_back(packedMaterial[j]);
          }
          compactActiveMask.push_back(1);
          compactPrimitiveIndices.push_back(
              static_cast<int>(record.primitiveBase + activeCount));

          ++activeCount;
        }

        record.primitiveCount = static_cast<uint32_t>(activeCount);
        if (activeCount == 0) {
          for (const auto &edit : residentIndexEdits)
            if (edit.first < _primitiveToResidentIndex.size())
              _primitiveToResidentIndex[edit.first] = edit.second;
          objectShouldBeResident[objectIndex] = false;
          _instanceRecords[objectIndex] = record;
          ++blasCacheSkipped;
          continue;
        }

        size_t nodeBase = compactBVHNodes.size() / 2;

        std::vector<BVHNode> rebuiltNodeStructs;
        rebuiltNodeStructs.reserve(obj.cachedBlasNodes.size());

        std::function<int(int)> rebuildNode;
        rebuildNode = [&](int nodeIdx) -> int {
          if (nodeIdx < 0 ||
              static_cast<size_t>(nodeIdx) >= obj.cachedBlasNodes.size())
            return -1;

          BVHNode adjusted = obj.cachedBlasNodes[nodeIdx];
          if (adjusted.count > 0) {
            int newLeftFirst = -1;
            int newCount = 0;
            int originalCount = adjusted.count;
            int localFirst = adjusted.leftFirst;
            for (int i = 0; i < originalCount; ++i) {
              size_t idx = static_cast<size_t>(localFirst + i);
              if (idx >= cachedRemap.size())
                continue;
              int32_t remapped = cachedRemap[idx];
              if (remapped < 0)
                continue;
              if (newLeftFirst < 0)
                newLeftFirst =
                    static_cast<int>(record.primitiveIndexBase + remapped);
              ++newCount;
            }
            if (newCount == 0)
              return -1;
            if (newLeftFirst < 0)
              newLeftFirst = static_cast<int>(record.primitiveIndexBase);
            adjusted.leftFirst = newLeftFirst;
            adjusted.count = newCount;
          } else {
            int leftChild = rebuildNode(adjusted.leftFirst);
            int rightChild = rebuildNode(-adjusted.count);
            if (leftChild < 0 && rightChild < 0)
              return -1;
            if (leftChild < 0)
              return rightChild;
            if (rightChild < 0)
              return leftChild;
            adjusted.leftFirst = leftChild;
            adjusted.count = -rightChild;
          }

          int newIndex = static_cast<int>(rebuiltNodeStructs.size());
          rebuiltNodeStructs.push_back(adjusted);
          return newIndex;
        };

        int rebuiltRoot = rebuildNode(obj.cachedBlasRootIndex);
        if (rebuiltRoot >= 0) {
          record.blasRootIndex =
              static_cast<int32_t>(nodeBase + rebuiltRoot);
          for (const BVHNode &node : rebuiltNodeStructs) {
            BVHNode adjusted = node;
            if (adjusted.count > 0) {
              // Leaf nodes already reference compact primitive indices.
            } else {
              int leftChild = adjusted.leftFirst + static_cast<int>(nodeBase);
              int rightChild = -adjusted.count + static_cast<int>(nodeBase);
              adjusted.leftFirst = leftChild;
              adjusted.count = -rightChild;
            }
            float leftBits = 0.0f;
            float rightBits = 0.0f;
            std::memcpy(&leftBits, &adjusted.leftFirst, sizeof(int));
            std::memcpy(&rightBits, &adjusted.count, sizeof(int));
            compactBVHNodes.push_back(
                simd::make_float4(adjusted.boundsMin, leftBits));
            compactBVHNodes.push_back(
                simd::make_float4(adjusted.boundsMax, rightBits));
          }

          objectShouldBeResident[objectIndex] = true;
          _instanceRecords[objectIndex] = record;
          ++blasCacheReused;
          continue;
        }

        printf("Cached BLAS leaf remapped to zero primitives for object %zu, rebuilding fallback BVH.\n",
               objectIndex);
        remapUpload.resize(remapStart);
        compactPrimitiveData.resize(compactPrimitiveDataStart);
        compactMaterialData.resize(compactMaterialDataStart);
        compactPrimitiveIndices.resize(compactPrimitiveIndicesStart);
        compactActiveMask.resize(compactActiveMaskStart);
        compactBVHNodes.resize(compactBVHNodesStart);
        compactTriangleVertices.resize(compactTriangleVerticesStart);
        compactTriangleIndices.resize(compactTriangleIndicesStart);

        for (const auto &edit : residentIndexEdits)
          if (edit.first < _primitiveToResidentIndex.size())
            _primitiveToResidentIndex[edit.first] = edit.second;

        record.blasRootIndex = -1;
      }

      activeLocalPrims.reserve(obj.primitiveCount);
      activeLocalPrims.clear();
      size_t first = obj.firstPrimitive;
      size_t last = first + obj.primitiveCount;
      for (size_t prim = first;
           prim < last && prim < _activePrimitive.size(); ++prim) {
        if (_activePrimitive[prim])
          activeLocalPrims.push_back(prim);
      }

      record.primitiveCount = static_cast<uint32_t>(activeLocalPrims.size());
      if (activeLocalPrims.empty()) {
        objectShouldBeResident[objectIndex] = false;
        _instanceRecords[objectIndex] = record;
        continue;
      }

      ++blasCacheFallback;

      subset.clear();
      subset.reserve(activeLocalPrims.size());
      for (size_t idx : activeLocalPrims)
        subset.push_back(_allPrimitives[idx]);

      localPrimitiveIndices.resize(subset.size());
      std::iota(localPrimitiveIndices.begin(), localPrimitiveIndices.end(), 0);

      localNodes.clear();
      size_t localNodeCount = 0;
      simd::float4 *localBVHRaw = _pScene->createBVHBuffer(
          subset, localPrimitiveIndices, localNodeCount, localNodes);
      if (localBVHRaw)
        delete[] localBVHRaw;

      if (localNodeCount == 0 || localNodes.empty()) {
        record.primitiveCount = 0;
        objectShouldBeResident[objectIndex] = false;
        _instanceRecords[objectIndex] = record;
        continue;
      }

      for (size_t local = 0; local < subset.size(); ++local) {
        size_t globalIndex = activeLocalPrims[local];
        remapUpload.push_back(static_cast<uint32_t>(globalIndex));
        if (globalIndex < _primitiveToResidentIndex.size())
          _primitiveToResidentIndex[globalIndex] =
              static_cast<int32_t>(record.primitiveBase + local);

        const Primitive &p = subset[local];
        simd::float4 prim0;
        simd::float4 prim1;
        simd::float4 prim2;
        simd::float4 prim3 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        simd::float4 prim4 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        simd::float4 prim5 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        simd::float4 prim6 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        switch (p.type) {
        case PrimitiveType::Sphere: {
          prim0 = simd::make_float4(p.sphere.center, static_cast<float>(p.type));
          prim1 = simd::make_float4(
              simd::make_float3(p.sphere.radius, 0.0f, 0.0f), 0.0f);
          prim2 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          prim6 = simd::make_float4(0.0f, 0.0f, 1.0f, 0.0f);
          break;
        }
        case PrimitiveType::Rectangle: {
          prim0 = simd::make_float4(p.rectangle.center, static_cast<float>(p.type));
          prim1 = simd::make_float4(p.rectangle.u, 0.0f);
          prim2 = simd::make_float4(p.rectangle.v, 0.0f);
          prim4 = simd::make_float4(p.rectangle.tangent, 0.0f);
          prim5 = simd::make_float4(p.rectangle.bitangent, 0.0f);
          prim6 = simd::make_float4(p.rectangle.normal,
                                    p.rectangle.supportsNormalMap ? 1.0f
                                                                  : 0.0f);
          break;
        }
        case PrimitiveType::Triangle: {
          prim0 = simd::make_float4(p.triangle.v0, static_cast<float>(p.type));
          prim1 = simd::make_float4(p.triangle.v1, p.triangle.uv0.x);
          prim2 = simd::make_float4(p.triangle.v2, p.triangle.uv0.y);
          prim3 = simd::make_float4(p.triangle.uv1.x, p.triangle.uv1.y,
                                    p.triangle.uv2.x, p.triangle.uv2.y);
          prim4 = simd::make_float4(p.triangle.tangent, 0.0f);
          prim5 = simd::make_float4(p.triangle.bitangent, 0.0f);
          prim6 = simd::make_float4(p.triangle.normal, 1.0f);
          size_t baseVertex = compactTriangleVertices.size();
          compactTriangleVertices.push_back(p.triangle.v0);
          compactTriangleVertices.push_back(p.triangle.v1);
          compactTriangleVertices.push_back(p.triangle.v2);
          compactTriangleIndices.push_back(simd::make_uint3(
              static_cast<uint32_t>(baseVertex),
              static_cast<uint32_t>(baseVertex + 1),
              static_cast<uint32_t>(baseVertex + 2)));
          break;
        }
        }

        compactPrimitiveData.push_back(prim0);
        compactPrimitiveData.push_back(prim1);
        compactPrimitiveData.push_back(prim2);
        compactPrimitiveData.push_back(prim3);
        compactPrimitiveData.push_back(prim4);
        compactPrimitiveData.push_back(prim5);
        compactPrimitiveData.push_back(prim6);

        const Material &m = p.material;
        auto packedMaterial = encodeMaterial(m);
        for (size_t j = 0; j < kMaterialFloat4Count; ++j) {
          compactMaterialData.push_back(packedMaterial[j]);
        }
        compactActiveMask.push_back(1);
      }

      for (int &idx : localPrimitiveIndices)
        idx = static_cast<int>(record.primitiveBase + idx);
      compactPrimitiveIndices.insert(compactPrimitiveIndices.end(),
                                     localPrimitiveIndices.begin(),
                                     localPrimitiveIndices.end());

      size_t nodeBase = compactBVHNodes.size() / 2;
      for (const BVHNode &node : localNodes) {
        BVHNode adjusted = node;
        if (adjusted.count > 0) {
          adjusted.leftFirst += static_cast<int>(record.primitiveIndexBase);
        } else {
          int leftChild = adjusted.leftFirst + static_cast<int>(nodeBase);
          int rightChild = -adjusted.count + static_cast<int>(nodeBase);
          adjusted.leftFirst = leftChild;
          adjusted.count = -rightChild;
        }
        float leftBits = 0.0f;
        float rightBits = 0.0f;
        std::memcpy(&leftBits, &adjusted.leftFirst, sizeof(int));
        std::memcpy(&rightBits, &adjusted.count, sizeof(int));
        compactBVHNodes.push_back(
            simd::make_float4(adjusted.boundsMin, leftBits));
        compactBVHNodes.push_back(
            simd::make_float4(adjusted.boundsMax, rightBits));
      }

      record.blasRootIndex = static_cast<int32_t>(nodeBase);
      objectShouldBeResident[objectIndex] = record.primitiveCount > 0;
      _instanceRecords[objectIndex] = record;
    }

    printf("BLAS cache stats: reused %zu, rebuilt %zu, skipped %zu\n",
           blasCacheReused, blasCacheFallback, blasCacheSkipped);

    primitiveSource = &compactPrimitiveData;
    materialSource = &compactMaterialData;
    primitiveIndexSource = &compactPrimitiveIndices;
    bvhSource = &compactBVHNodes;
    tlasSource = &_cachedTLASNodes;
    triangleVertexSource = &compactTriangleVertices;
    triangleIndexSource = &compactTriangleIndices;

    _residentPrimitiveCount = remapUpload.size();
    _residentTriangleCount = compactTriangleIndices.size();
    _blasNodeCount = compactBVHNodes.size() / 2;
    _tlasNodeCount = _cachedTLASNodes.size() / 2;

    std::vector<uint32_t> remappedLights;
    remappedLights.reserve(_cachedLightIndices.size());
    for (uint32_t globalIndex : _cachedLightIndices) {
      if (globalIndex < _primitiveToResidentIndex.size()) {
        int32_t local = _primitiveToResidentIndex[globalIndex];
        if (local >= 0)
          remappedLights.push_back(static_cast<uint32_t>(local));
      }
    }
    _cachedLightIndices.swap(remappedLights);
    _lightCount = _cachedLightIndices.size();
  }

  if (_frameStrategy == ResidencyStrategy::AlwaysResident) {
    for (size_t objectIndex = 0; objectIndex < sceneObjects.size();
         ++objectIndex) {
      const SceneObject &obj = sceneObjects[objectIndex];
      if (obj.primitiveCount > 0) {
        objectShouldBeResident[objectIndex] = true;
      }
    }
  }

  for (size_t objectIndex = 0; objectIndex < sceneObjects.size(); ++objectIndex) {
    bool shouldBeResident = objectShouldBeResident[objectIndex];
    auto &gpuResident = _residentObjectGpuResources[objectIndex];
    auto &instanceRecord = _instanceRecords[objectIndex];

    if (shouldBeResident) {
      bool built = gpuResident.ensureResident(
          *this, objectIndex, sceneObjects[objectIndex], instanceRecord,
          needFullUpload);
      if (!built) {
        shouldBeResident = false;
        objectShouldBeResident[objectIndex] = false;
      }
    }

    if (!shouldBeResident &&
        _frameStrategy != ResidencyStrategy::AlwaysResident) {
      requestResidentEviction(objectIndex, gpuResident, instanceRecord);
    }
  }

  std::vector<MTL::AccelerationStructureInstanceDescriptor> instanceDescriptors(
      sceneObjects.size());
  std::vector<MTL::AccelerationStructure *> instancedStructures(
      sceneObjects.size(), nullptr);
  std::vector<GeometryHandle> geometryHandles(sceneObjects.size() + 1);
  if (!geometryHandles.empty())
    geometryHandles[0] = GeometryHandle{};
  MTL::PackedFloat4x3 identityMatrix;
  identityMatrix[0] = MTL::PackedFloat3(1.0f, 0.0f, 0.0f);
  identityMatrix[1] = MTL::PackedFloat3(0.0f, 1.0f, 0.0f);
  identityMatrix[2] = MTL::PackedFloat3(0.0f, 0.0f, 1.0f);
  identityMatrix[3] = MTL::PackedFloat3(0.0f, 0.0f, 0.0f);

  for (size_t objectIndex = 0; objectIndex < sceneObjects.size();
       ++objectIndex) {
    auto &desc = instanceDescriptors[objectIndex];
    desc.transformationMatrix = identityMatrix;
    desc.options = MTL::AccelerationStructureInstanceOptionNone;
    desc.intersectionFunctionTableOffset = 0;
    desc.accelerationStructureIndex =
        static_cast<uint32_t>(objectIndex + 1);

    bool resident = false;
    if (objectIndex < _residentObjectGpuResources.size()) {
      const auto &gpuResident = _residentObjectGpuResources[objectIndex];
      resident = gpuResident.isResident() && objectShouldBeResident[objectIndex];
      if (resident) {
        MTL::AccelerationStructure *structure =
            gpuResident.resources.accelerationStructure();
        if (structure) {
          instancedStructures[objectIndex] = structure;
        } else {
          resident = false;
        }
      }
    }

    desc.mask = resident ? 0xFFu : 0u;

    GeometryHandle handle{};
    if (resident && objectIndex < _residentObjectGpuResources.size()) {
      const auto &gpuResident = _residentObjectGpuResources[objectIndex];
      if (gpuResident.geometryValid) {
        MTL::Buffer *vertexBuffer = gpuResident.resources.vertexBuffer();
        MTL::Buffer *indexBuffer = gpuResident.resources.indexBuffer();
        if (vertexBuffer && indexBuffer) {
          handle.vertexBufferAddress =
              vertexBuffer->gpuAddress() + gpuResident.vertexBufferOffset;
          handle.indexBufferAddress =
              indexBuffer->gpuAddress() + gpuResident.indexBufferOffset;
          handle.vertexStride = static_cast<uint32_t>(sizeof(simd::float3));
          handle.indexStride = static_cast<uint32_t>(sizeof(uint32_t));
          handle.vertexCount =
              static_cast<uint32_t>(gpuResident.vertexCount);
          handle.indexCount =
              static_cast<uint32_t>(gpuResident.triangleCount * 3);
          handle.instanceSlot = static_cast<uint32_t>(objectIndex + 1);
        }
      }
    }
    geometryHandles[objectIndex + 1] = handle;
  }

  updateTopLevelAccelerationStructure(instanceDescriptors, instancedStructures);

  _residentRemap = remapUpload;

  bool hasDirtyPrimitiveRanges = !dirtyPrimitiveRanges.empty();
  bool uploadAll = needFullUpload || useCompaction || compactionStateChanged;

  if (uploadAll) {
    size_t primitiveFloat4Count = primitiveSource->size();
    size_t primitiveCount =
        (kPrimitiveFloat4Count > 0)
            ? ((primitiveFloat4Count + kPrimitiveFloat4Count - 1) /
               kPrimitiveFloat4Count)
            : 0;
    size_t primitiveBytes = static_cast<size_t>(
        GpuHeapResources::primitiveDataSize(primitiveCount));
    if (primitiveBytes == 0)
      primitiveBytes = sizeof(simd::float4);
    ensureBufferCapacity(_pSphereBuffer, primitiveBytes, _sphereBufferCapacity,
                         allowShrink, MTL::ResourceStorageModeManaged,
                         "PrimitiveData", shrinkContext);
    if (_pSphereBuffer) {
      simd::float4 *dst =
          static_cast<simd::float4 *>(_pSphereBuffer->contents());
      size_t modifiedBytes = primitiveBytes;
      if (primitiveFloat4Count > 0)
        std::memcpy(dst, primitiveSource->data(),
                    primitiveFloat4Count * sizeof(simd::float4));
      else
        dst[0] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
      if (primitiveFloat4Count > 0)
        modifiedBytes = primitiveFloat4Count * sizeof(simd::float4);
      markBufferModified(_pSphereBuffer, NS::Range::Make(0, modifiedBytes));
    }

    size_t materialFloat4Count = materialSource->size();
    size_t materialBytes =
        std::max<size_t>(materialFloat4Count, size_t(1)) *
        sizeof(simd::float4);
    ensureBufferCapacity(_pSphereMaterialBuffer, materialBytes,
                         _sphereMaterialBufferCapacity, allowShrink,
                         MTL::ResourceStorageModeManaged, "PrimitiveMaterials",
                         shrinkContext);
    if (_pSphereMaterialBuffer) {
      simd::float4 *dst = static_cast<simd::float4 *>(
          _pSphereMaterialBuffer->contents());
      if (materialFloat4Count > 0)
        std::memcpy(dst, materialSource->data(),
                    materialFloat4Count * sizeof(simd::float4));
      else {
        dst[0] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (materialBytes >= 2 * sizeof(simd::float4))
          dst[1] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
      markBufferModified(_pSphereMaterialBuffer,
                         NS::Range::Make(0, materialBytes));
    }

    size_t primitiveIndexCount = primitiveIndexSource->size();
    size_t primitiveIndexBytes =
        std::max<size_t>(primitiveIndexCount, size_t(1)) * sizeof(int);
    ensureBufferCapacity(_pPrimitiveIndexBuffer, primitiveIndexBytes,
                         _primitiveIndexBufferCapacity, allowShrink,
                         MTL::ResourceStorageModeManaged, "PrimitiveIndices",
                         shrinkContext);
    if (_pPrimitiveIndexBuffer) {
      int *dst = static_cast<int *>(_pPrimitiveIndexBuffer->contents());
      if (primitiveIndexCount > 0)
        std::memcpy(dst, primitiveIndexSource->data(),
                    primitiveIndexCount * sizeof(int));
      else
        dst[0] = 0;
      markBufferModified(_pPrimitiveIndexBuffer,
                         NS::Range::Make(0, primitiveIndexBytes));
    }

    size_t blasFloat4Count = bvhSource->size();
    size_t bvhBytes =
        std::max<size_t>(blasFloat4Count, size_t(1)) * sizeof(simd::float4);
    ensureBufferCapacity(_pBVHBuffer, bvhBytes, _bvhBufferCapacity, allowShrink,
                         MTL::ResourceStorageModeManaged, "BLASNodes",
                         shrinkContext);
    if (_pBVHBuffer) {
      simd::float4 *dst =
          static_cast<simd::float4 *>(_pBVHBuffer->contents());
      if (blasFloat4Count > 0)
        std::memcpy(dst, bvhSource->data(),
                    blasFloat4Count * sizeof(simd::float4));
      else {
        dst[0] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (bvhBytes >= 2 * sizeof(simd::float4))
          dst[1] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
      markBufferModified(_pBVHBuffer, NS::Range::Make(0, bvhBytes));
    }

    size_t tlasFloat4Count = tlasSource->size();
    size_t tlasBytes =
        std::max<size_t>(tlasFloat4Count, size_t(1)) * sizeof(simd::float4);
    ensureBufferCapacity(_pTLASBuffer, tlasBytes, _tlasBufferCapacity,
                         allowShrink, MTL::ResourceStorageModeManaged,
                         "TLASNodes", shrinkContext);
    if (_pTLASBuffer) {
      simd::float4 *dst =
          static_cast<simd::float4 *>(_pTLASBuffer->contents());
      if (tlasFloat4Count > 0)
        std::memcpy(dst, tlasSource->data(),
                    tlasFloat4Count * sizeof(simd::float4));
      else {
        dst[0] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (tlasBytes >= 2 * sizeof(simd::float4))
          dst[1] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
      markBufferModified(_pTLASBuffer, NS::Range::Make(0, tlasBytes));
    }

    size_t remapCount = std::max<size_t>(_residentRemap.size(), size_t(1));
    ensureBufferCapacity(_pPrimitiveRemapBuffer,
                         remapCount * sizeof(uint32_t),
                         _primitiveRemapBufferCapacity, allowShrink,
                         MTL::ResourceStorageModeManaged, "PrimitiveRemap",
                         shrinkContext);
    if (_pPrimitiveRemapBuffer) {
      uint32_t *dst = static_cast<uint32_t *>(
          _pPrimitiveRemapBuffer->contents());
      if (_residentRemap.empty()) {
        dst[0] = 0;
      } else {
        std::memcpy(dst, _residentRemap.data(),
                    _residentRemap.size() * sizeof(uint32_t));
        if (_residentRemap.size() < remapCount)
          dst[_residentRemap.size()] = 0;
      }
      markBufferModified(_pPrimitiveRemapBuffer,
                         NS::Range::Make(0, remapCount * sizeof(uint32_t)));
    }

    size_t vertexCount = triangleVertexSource->size();
    size_t vertexBytes =
        std::max<size_t>(vertexCount, size_t(1)) * sizeof(simd::float3);
    ensureBufferCapacity(_pTriangleVertexBuffer, vertexBytes,
                         _triangleVertexBufferCapacity, allowShrink,
                         MTL::ResourceStorageModeManaged, "TriangleVertices",
                         shrinkContext);
    if (_pTriangleVertexBuffer) {
      simd::float3 *dst = static_cast<simd::float3 *>(
          _pTriangleVertexBuffer->contents());
      if (vertexCount > 0)
        std::memcpy(dst, triangleVertexSource->data(),
                    vertexCount * sizeof(simd::float3));
      else
        dst[0] = simd::float3{0.0f, 0.0f, 0.0f};
      markBufferModified(_pTriangleVertexBuffer,
                         NS::Range::Make(0, vertexBytes));
    }

    size_t indexCount = triangleIndexSource->size();
    size_t indexBytes =
        std::max<size_t>(indexCount, size_t(1)) * sizeof(simd::uint3);
    ensureBufferCapacity(_pTriangleIndexBuffer, indexBytes,
                         _triangleIndexBufferCapacity, allowShrink,
                         MTL::ResourceStorageModeManaged, "TriangleIndices",
                         shrinkContext);
    if (_pTriangleIndexBuffer) {
      simd::uint3 *dst = static_cast<simd::uint3 *>(
          _pTriangleIndexBuffer->contents());
      if (indexCount > 0)
        std::memcpy(dst, triangleIndexSource->data(),
                    indexCount * sizeof(simd::uint3));
      else
        dst[0] = simd::make_uint3(0, 0, 0);
      markBufferModified(_pTriangleIndexBuffer,
                         NS::Range::Make(0, indexBytes));
    }
    _residentBuffersInitialized = true;
  } else if (hasDirtyPrimitiveRanges) {
    if (_pSphereBuffer && !_cachedPrimitiveData.empty()) {
      simd::float4 *dst =
          static_cast<simd::float4 *>(_pSphereBuffer->contents());
      for (const auto &range : dirtyPrimitiveRanges) {
        size_t float4Offset = range.first * kPrimitiveFloat4Count;
        size_t float4Count = range.second * kPrimitiveFloat4Count;
        size_t byteOffset = float4Offset * sizeof(simd::float4);
        size_t byteLength = float4Count * sizeof(simd::float4);
        std::memcpy(dst + float4Offset,
                    _cachedPrimitiveData.data() + float4Offset, byteLength);
        markBufferModified(_pSphereBuffer,
                           NS::Range::Make(byteOffset, byteLength));
      }
    }

    if (_pSphereMaterialBuffer && !_cachedMaterialData.empty()) {
      simd::float4 *dst = static_cast<simd::float4 *>(
          _pSphereMaterialBuffer->contents());
      for (const auto &range : dirtyPrimitiveRanges) {
        size_t float4Offset = range.first * kMaterialFloat4Count;
        size_t float4Count = range.second * kMaterialFloat4Count;
        size_t byteOffset = float4Offset * sizeof(simd::float4);
        size_t byteLength = float4Count * sizeof(simd::float4);
        std::memcpy(dst + float4Offset,
                    _cachedMaterialData.data() + float4Offset, byteLength);
        markBufferModified(_pSphereMaterialBuffer,
                           NS::Range::Make(byteOffset, byteLength));
      }
    }
  }

  size_t instanceCount = std::max<size_t>(_instanceRecords.size(), size_t(1));
  size_t instanceBytes = instanceCount * sizeof(BlasInstanceRecord);
  ensureBufferCapacity(_pInstanceBuffer, instanceBytes, _instanceBufferCapacity,
                       allowShrink, MTL::ResourceStorageModeManaged,
                       "InstanceRecords", shrinkContext);
  if (_pInstanceBuffer) {
    auto *dst = static_cast<BlasInstanceRecord *>(
        _pInstanceBuffer->contents());
    if (_instanceRecords.empty()) {
      dst[0] = BlasInstanceRecord{};
    } else {
      std::memcpy(dst, _instanceRecords.data(),
                  _instanceRecords.size() * sizeof(BlasInstanceRecord));
      if (_instanceRecords.size() < instanceCount)
        dst[_instanceRecords.size()] = BlasInstanceRecord{};
    }
    markBufferModified(_pInstanceBuffer, NS::Range::Make(0, instanceBytes));
  }

  size_t geometryHandleCount = geometryHandles.size();
  size_t geometryHandleBytes =
      std::max<size_t>(geometryHandleCount, size_t(1)) *
      sizeof(GeometryHandle);
  ensureBufferCapacity(_pGeometryHandleBuffer, geometryHandleBytes,
                       _geometryHandleBufferCapacity, allowShrink,
                       MTL::ResourceStorageModeManaged, "GeometryHandles",
                       shrinkContext);
  if (_pGeometryHandleBuffer) {
    auto *dst = static_cast<GeometryHandle *>(
        _pGeometryHandleBuffer->contents());
    if (!geometryHandles.empty()) {
      std::memcpy(dst, geometryHandles.data(),
                  geometryHandles.size() * sizeof(GeometryHandle));
    } else {
      *dst = GeometryHandle{};
    }
    markBufferModified(_pGeometryHandleBuffer,
                       NS::Range::Make(0, geometryHandleBytes));
  }

  size_t activeMaskCount =
      useCompaction ? _residentPrimitiveCount : totalPrimitiveCount;
  size_t activeBytes =
      std::max<size_t>(activeMaskCount, size_t(1)) * sizeof(uint8_t);
  ensureBufferCapacity(_pActiveBuffer, activeBytes, _activeBufferCapacity,
                       allowShrink, MTL::ResourceStorageModeManaged,
                       "ActiveMask", shrinkContext);
  if (_pActiveBuffer) {
    uint8_t *activePtr =
        static_cast<uint8_t *>(_pActiveBuffer->contents());
    if (useCompaction) {
      if (_residentPrimitiveCount > 0) {
        std::memcpy(activePtr, compactActiveMask.data(),
                    _residentPrimitiveCount * sizeof(uint8_t));
      } else {
        activePtr[0] = 0;
      }
      markBufferModified(_pActiveBuffer, NS::Range::Make(0, activeBytes));
    } else if (uploadAll) {
      if (totalPrimitiveCount > 0)
        std::memcpy(activePtr, _cpuActiveMask.data(),
                    totalPrimitiveCount * sizeof(uint8_t));
      else
        activePtr[0] = 0;
      markBufferModified(_pActiveBuffer, NS::Range::Make(0, activeBytes));
    } else {
      auto updateMask = [&](const std::vector<size_t> &indices) {
        for (size_t idx : indices) {
          if (idx >= totalPrimitiveCount)
            continue;
          bool active = idx < _activePrimitive.size() && _activePrimitive[idx];
          uint8_t value = active ? 1 : 0;
          _cpuActiveMask[idx] = value;
          activePtr[idx] = value;
          markBufferModified(_pActiveBuffer, NS::Range::Make(idx, 1));
        }
      };
      updateMask(_recentlyActivated);
      updateMask(_recentlyDeactivated);
    }
  }

  size_t lightIndexBytes =
      std::max<size_t>(_cachedLightIndices.size(), size_t(1)) *
      sizeof(uint32_t);
  ensureBufferCapacity(_pLightIndexBuffer, lightIndexBytes,
                       _lightIndexBufferCapacity, allowShrink,
                       MTL::ResourceStorageModeManaged, "LightIndices",
                       shrinkContext);
  if (_pLightIndexBuffer) {
    uint32_t *dst = static_cast<uint32_t *>(_pLightIndexBuffer->contents());
    if (_cachedLightIndices.empty())
      dst[0] = 0;
    else
      std::memcpy(dst, _cachedLightIndices.data(),
                  _cachedLightIndices.size() * sizeof(uint32_t));
    markBufferModified(_pLightIndexBuffer,
                       NS::Range::Make(0, lightIndexBytes));
  }

  size_t lightCdfBytes =
      std::max<size_t>(_cachedLightCdf.size(), size_t(1)) * sizeof(float);
  ensureBufferCapacity(_pLightCdfBuffer, lightCdfBytes, _lightCdfBufferCapacity,
                       allowShrink, MTL::ResourceStorageModeManaged,
                       "LightCdf", shrinkContext);
  if (_pLightCdfBuffer) {
    float *dst = static_cast<float *>(_pLightCdfBuffer->contents());
    if (_cachedLightCdf.empty())
      dst[0] = 0.0f;
    else
      std::memcpy(dst, _cachedLightCdf.data(),
                  _cachedLightCdf.size() * sizeof(float));
    markBufferModified(_pLightCdfBuffer, NS::Range::Make(0, lightCdfBytes));
  }

  recalculateNodeCounters(objectShouldBeResident);

  _objectResidentState = objectShouldBeResident;
  _dirtyResidentObjects.clear();
  _recentlyActivated.clear();
  _recentlyDeactivated.clear();

  _needsAccumulationReset = true;
}

std::vector<bool> Renderer::buildResidentMaskFromGpuResources() const {
  std::vector<bool> residentMask;
  if (_residentObjectGpuResources.empty())
    return residentMask;

  size_t objectCount =
      std::max(_residentObjectGpuResources.size(), _objectResidentState.size());
  residentMask.assign(objectCount, false);
  for (size_t objectIndex = 0; objectIndex < _residentObjectGpuResources.size();
       ++objectIndex) {
    residentMask[objectIndex] =
        _residentObjectGpuResources[objectIndex].isResident();
  }
  return residentMask;
}

void Renderer::recalculateNodeCounters(
    const std::vector<bool> &residentMask) {
  bool hasBlasData =
      _blasNodeCount > 0 && _cachedBVHNodes.size() >= _blasNodeCount * 2;
  bool hasTlasData =
      _tlasNodeCount > 0 && _cachedTLASNodes.size() >= _tlasNodeCount * 2;

  if (_blasNodeCount == 0 && _tlasNodeCount == 0) {
    _residentNodeCount = 0;
    _activeNodeCount = 0;
    return;
  }

  const std::vector<SceneObject> *sceneObjects = nullptr;
  if (!_allSceneObjects.empty()) {
    sceneObjects = &_allSceneObjects;
  } else if (_pScene) {
    sceneObjects = &_pScene->getObjects();
  }

  size_t totalPrimitiveCount = _allPrimitives.size();
  size_t residentBlasNodes = 0;
  size_t residentTlasNodes = 0;

  if (_residentCompacted) {
    residentBlasNodes = _blasNodeCount;
    residentTlasNodes = _tlasNodeCount;
  } else {
    if (hasBlasData) {
      std::vector<uint8_t> primitiveResident(totalPrimitiveCount, 0);
      size_t activeLimit =
          std::min(totalPrimitiveCount, _activePrimitive.size());
      for (size_t i = 0; i < activeLimit; ++i) {
        if (_activePrimitive[i])
          primitiveResident[i] = 1;
      }

      if (sceneObjects) {
        for (size_t objectIndex = 0;
             objectIndex < sceneObjects->size(); ++objectIndex) {
          if (objectIndex >= residentMask.size() ||
              !residentMask[objectIndex])
            continue;
          const SceneObject &obj = (*sceneObjects)[objectIndex];
          size_t first = obj.firstPrimitive;
          size_t last = first + obj.primitiveCount;
          for (size_t prim = first;
               prim < last && prim < totalPrimitiveCount; ++prim) {
            primitiveResident[prim] = 1;
          }
        }
      }

      const auto &bvhNodes = _cachedBVHNodes;
      const auto &primitiveIndices = _cachedPrimitiveIndices;
      std::vector<uint8_t> processed(_blasNodeCount, 0);
      std::vector<uint8_t> nodeResident(_blasNodeCount, 0);
      std::vector<size_t> stack;
      stack.reserve(_blasNodeCount);
      stack.push_back(0);

      auto decodeNode = [](const simd::float4 &minVec,
                           const simd::float4 &maxVec, int &leftFirst,
                           int &count) {
        const auto *minWords = reinterpret_cast<const int *>(&minVec);
        const auto *maxWords = reinterpret_cast<const int *>(&maxVec);
        leftFirst = minWords[3];
        count = maxWords[3];
      };

      while (!stack.empty()) {
        size_t nodeIdx = stack.back();
        if (nodeIdx >= _blasNodeCount) {
          stack.pop_back();
          continue;
        }

        if (processed[nodeIdx]) {
          stack.pop_back();
          continue;
        }

        int leftFirst = 0;
        int count = 0;
        decodeNode(bvhNodes[2 * nodeIdx], bvhNodes[2 * nodeIdx + 1], leftFirst,
                   count);

        if (count > 0) {
          bool leafResident = false;
          size_t start = static_cast<size_t>(std::max(leftFirst, 0));
          size_t end = start + static_cast<size_t>(std::max(count, 0));
          for (size_t idx = start; idx < end; ++idx) {
            size_t primitiveIndex =
                (idx < primitiveIndices.size())
                    ? static_cast<size_t>(primitiveIndices[idx])
                    : idx;
            if (primitiveIndex < primitiveResident.size() &&
                primitiveResident[primitiveIndex]) {
              leafResident = true;
              break;
            }
          }
          nodeResident[nodeIdx] = leafResident ? 1 : 0;
          processed[nodeIdx] = 1;
          stack.pop_back();
        } else {
          size_t leftChild =
              static_cast<size_t>(leftFirst >= 0 ? leftFirst : 0);
          size_t rightChild = static_cast<size_t>(-count);
          bool leftDone = leftChild >= _blasNodeCount || processed[leftChild];
          bool rightDone = rightChild >= _blasNodeCount || processed[rightChild];
          if (!leftDone) {
            stack.push_back(leftChild);
            continue;
          }
          if (!rightDone) {
            stack.push_back(rightChild);
            continue;
          }
          bool anyResident =
              (leftChild < _blasNodeCount && nodeResident[leftChild]) ||
              (rightChild < _blasNodeCount && nodeResident[rightChild]);
          nodeResident[nodeIdx] = anyResident ? 1 : 0;
          processed[nodeIdx] = 1;
          stack.pop_back();
        }
      }

      residentBlasNodes =
          std::accumulate(nodeResident.begin(), nodeResident.end(), size_t(0));
    } else {
      residentBlasNodes = _blasNodeCount;
    }

    if (hasTlasData) {
      const auto &tlasNodes = _cachedTLASNodes;
      std::vector<uint8_t> processed(_tlasNodeCount, 0);
      std::vector<uint8_t> nodeResident(_tlasNodeCount, 0);
      std::vector<size_t> stack;
      stack.reserve(_tlasNodeCount);
      stack.push_back(0);

      while (!stack.empty()) {
        size_t nodeIdx = stack.back();
        if (nodeIdx >= _tlasNodeCount) {
          stack.pop_back();
          continue;
        }

        if (processed[nodeIdx]) {
          stack.pop_back();
          continue;
        }

        int leftChild = 0;
        int rightChild = 0;
        const auto *leftWords =
            reinterpret_cast<const int *>(&tlasNodes[2 * nodeIdx]);
        const auto *rightWords =
            reinterpret_cast<const int *>(&tlasNodes[2 * nodeIdx + 1]);
        leftChild = leftWords[3];
        rightChild = rightWords[3];

        if (leftChild < 0) {
          size_t objectIndex = static_cast<size_t>(-leftChild - 1);
          bool resident = objectIndex < residentMask.size() &&
                          residentMask[objectIndex];
          nodeResident[nodeIdx] = resident ? 1 : 0;
          processed[nodeIdx] = 1;
          stack.pop_back();
          continue;
        }

        size_t leftIndex = static_cast<size_t>(leftChild);
        size_t rightIndex = static_cast<size_t>(rightChild);
        bool leftDone = leftIndex >= _tlasNodeCount || processed[leftIndex];
        bool rightDone = rightIndex >= _tlasNodeCount || processed[rightIndex];
        if (!leftDone) {
          stack.push_back(leftIndex);
          continue;
        }
        if (!rightDone) {
          stack.push_back(rightIndex);
          continue;
        }
        bool anyResident =
            (leftIndex < _tlasNodeCount && nodeResident[leftIndex]) ||
            (rightIndex < _tlasNodeCount && nodeResident[rightIndex]);
        nodeResident[nodeIdx] = anyResident ? 1 : 0;
        processed[nodeIdx] = 1;
        stack.pop_back();
      }

      residentTlasNodes =
          std::accumulate(nodeResident.begin(), nodeResident.end(), size_t(0));
    } else {
      residentTlasNodes = _tlasNodeCount;
    }
  }

  _residentNodeCount = residentBlasNodes + residentTlasNodes;

  size_t activeBlasNodes = 0;
  if (_residentCompacted) {
    activeBlasNodes = _blasNodeCount;
  } else if (hasBlasData) {
    const auto &bvhNodes = _cachedBVHNodes;
    const auto &primitiveIndices = _cachedPrimitiveIndices;
    std::vector<uint8_t> processed(_blasNodeCount, 0);
    std::vector<uint8_t> nodeActive(_blasNodeCount, 0);
    std::vector<size_t> stack;
    stack.reserve(_blasNodeCount);
    stack.push_back(0);

    auto decodeNode = [](const simd::float4 &minVec,
                         const simd::float4 &maxVec, int &leftFirst,
                         int &count) {
      const auto *minWords = reinterpret_cast<const int *>(&minVec);
      const auto *maxWords = reinterpret_cast<const int *>(&maxVec);
      leftFirst = minWords[3];
      count = maxWords[3];
    };

    while (!stack.empty()) {
      size_t nodeIdx = stack.back();
      if (nodeIdx >= _blasNodeCount) {
        stack.pop_back();
        continue;
      }

      if (processed[nodeIdx]) {
        stack.pop_back();
        continue;
      }

      int leftFirst = 0;
      int count = 0;
      decodeNode(bvhNodes[2 * nodeIdx], bvhNodes[2 * nodeIdx + 1], leftFirst,
                 count);

      if (count > 0) {
        bool leafActive = false;
        size_t start = static_cast<size_t>(std::max(leftFirst, 0));
        size_t end = start + static_cast<size_t>(std::max(count, 0));
        for (size_t idx = start; idx < end; ++idx) {
          size_t primitiveIndex =
              (idx < primitiveIndices.size())
                  ? static_cast<size_t>(primitiveIndices[idx])
                  : idx;
          if (primitiveIndex < _activePrimitive.size() &&
              _activePrimitive[primitiveIndex]) {
            leafActive = true;
            break;
          }
        }
        nodeActive[nodeIdx] = leafActive ? 1 : 0;
        processed[nodeIdx] = 1;
        stack.pop_back();
      } else {
        size_t leftChild = static_cast<size_t>(leftFirst >= 0 ? leftFirst : 0);
        size_t rightChild = static_cast<size_t>(-count);
        bool leftDone = leftChild >= _blasNodeCount || processed[leftChild];
        bool rightDone = rightChild >= _blasNodeCount || processed[rightChild];
        if (!leftDone) {
          stack.push_back(leftChild);
          continue;
        }
        if (!rightDone) {
          stack.push_back(rightChild);
          continue;
        }
        bool anyActive =
            (leftChild < _blasNodeCount && nodeActive[leftChild]) ||
            (rightChild < _blasNodeCount && nodeActive[rightChild]);
        nodeActive[nodeIdx] = anyActive ? 1 : 0;
        processed[nodeIdx] = 1;
        stack.pop_back();
      }
    }

    activeBlasNodes =
        std::accumulate(nodeActive.begin(), nodeActive.end(), size_t(0));
  }

  size_t activeTlasNodes = 0;
  if (hasTlasData) {
    const auto &tlasNodes = _cachedTLASNodes;
    std::vector<uint8_t> processed(_tlasNodeCount, 0);
    std::vector<uint8_t> nodeActive(_tlasNodeCount, 0);
    std::vector<size_t> stack;
    stack.reserve(_tlasNodeCount);
    stack.push_back(0);

    while (!stack.empty()) {
      size_t nodeIdx = stack.back();
      if (nodeIdx >= _tlasNodeCount) {
        stack.pop_back();
        continue;
      }

      if (processed[nodeIdx]) {
        stack.pop_back();
        continue;
      }

      int leftChild = 0;
      int rightChild = 0;
      const auto *leftWords =
          reinterpret_cast<const int *>(&tlasNodes[2 * nodeIdx]);
      const auto *rightWords =
          reinterpret_cast<const int *>(&tlasNodes[2 * nodeIdx + 1]);
      leftChild = leftWords[3];
      rightChild = rightWords[3];

      if (leftChild < 0) {
        size_t objectIndex = static_cast<size_t>(-leftChild - 1);
        bool active = objectIndex < _objectActive.size() &&
                      _objectActive[objectIndex];
        nodeActive[nodeIdx] = active ? 1 : 0;
        processed[nodeIdx] = 1;
        stack.pop_back();
        continue;
      }

      size_t leftIndex = static_cast<size_t>(leftChild);
      size_t rightIndex = static_cast<size_t>(rightChild);
      bool leftDone = leftIndex >= _tlasNodeCount || processed[leftIndex];
      bool rightDone = rightIndex >= _tlasNodeCount || processed[rightIndex];
      if (!leftDone) {
        stack.push_back(leftIndex);
        continue;
      }
      if (!rightDone) {
        stack.push_back(rightIndex);
        continue;
      }
      bool anyActive =
          (leftIndex < _tlasNodeCount && nodeActive[leftIndex]) ||
          (rightIndex < _tlasNodeCount && nodeActive[rightIndex]);
      nodeActive[nodeIdx] = anyActive ? 1 : 0;
      processed[nodeIdx] = 1;
      stack.pop_back();
    }

    activeTlasNodes =
        std::accumulate(nodeActive.begin(), nodeActive.end(), size_t(0));
  } else if (_residentCompacted) {
    activeTlasNodes = _tlasNodeCount;
  }

  _activeNodeCount = activeBlasNodes + activeTlasNodes;
}




void Renderer::releaseTextureSlot(ManagedTextureSlot &slot) {
  if (slot.texture) {
    slot.texture->release();
    slot.texture = nullptr;
  }
  if (slot.stagingBuffer) {
    slot.stagingBuffer->release();
    slot.stagingBuffer = nullptr;
    slot.stagingCapacity = 0;
  }
  slot.stagingValid = false;
  clearTextureHistory(slot);
}

const char *Renderer::textureSlotLabel(const ManagedTextureSlot &slot) const {
  if (&slot == &_accumulationSlots[0])
    return "_accumulationSlots[0]";
  if (&slot == &_accumulationSlots[1])
    return "_accumulationSlots[1]";
  if (&slot == &_sampleCountSlot)
    return "_sampleCountSlot";
  if (&slot == &_sampleImportanceSlot)
    return "_sampleImportanceSlot";
  if (&slot == &_albedoSlot)
    return "_albedoSlot";
  if (&slot == &_normalSlot)
    return "_normalSlot";
  return "<unknown texture slot>";
}

void Renderer::configureTextureSlot(ManagedTextureSlot &slot, NS::UInteger width,
                                    NS::UInteger height,
                                    MTL::PixelFormat format,
                                    MTL::TextureUsage usage) {
  bool descriptorChanged =
      !slot.descriptorValid || slot.width != width || slot.height != height ||
      slot.pixelFormat != format || slot.usage != usage;

  if (!descriptorChanged)
    return;

  releaseTextureSlot(slot);

  slot.width = width;
  slot.height = height;
  slot.pixelFormat = format;
  slot.usage = usage;
  slot.textureType = MTL::TextureType::TextureType2D;
  slot.storageMode = MTL::StorageMode::StorageModePrivate;
  slot.descriptorValid = true;
  slot.stagingValid = false;
}

size_t Renderer::textureByteSize(const ManagedTextureSlot &slot) const {
  if (!slot.descriptorValid || slot.width == 0 || slot.height == 0)
    return 0;

  size_t pixelBytes = bytesPerPixel(slot.pixelFormat);
  if (pixelBytes == 0)
    return 0;

  size_t rowBytes = slot.width * pixelBytes;
  size_t alignedRowBytes = alignTo(rowBytes, 256);
  return alignedRowBytes * slot.height;
}

void Renderer::clearTextureHistory(ManagedTextureSlot &slot) {
  slot.historyData.clear();
  slot.historyBacking = ManagedTextureSlot::HistoryBacking::None;
  slot.historyIsProxy = false;
  slot.historyWidth = 0;
  slot.historyHeight = 0;
  slot.historyBytesPerRow = 0;
  slot.needsGpuRefresh = false;
}

bool Renderer::captureCpuHistoryProxy(ManagedTextureSlot &slot) {
  clearTextureHistory(slot);

  if (!slot.texture)
    return false;

  if (slot.width == 0 || slot.height == 0)
    return false;

  size_t pixelBytes = bytesPerPixel(slot.pixelFormat);
  if (pixelBytes == 0)
    return false;

  NS::UInteger proxyWidth = slot.width;
  NS::UInteger proxyHeight = slot.height;
  size_t proxyRowBytes = proxyWidth * pixelBytes;
  size_t proxyTotalBytes = proxyRowBytes * proxyHeight;

  while (proxyTotalBytes > kMaxTextureHistoryBytes && proxyWidth > 1 &&
         proxyHeight > 1) {
    proxyWidth = std::max<NS::UInteger>(1, proxyWidth / 2);
    proxyHeight = std::max<NS::UInteger>(1, proxyHeight / 2);
    proxyRowBytes = proxyWidth * pixelBytes;
    proxyTotalBytes = proxyRowBytes * proxyHeight;
  }

  if (proxyTotalBytes == 0)
    return false;

  std::vector<uint8_t> data(proxyTotalBytes, 0);
  std::vector<uint8_t> pixel(pixelBytes, 0);
  for (NS::UInteger y = 0; y < proxyHeight; ++y) {
    NS::UInteger srcY = static_cast<NS::UInteger>(
        std::min<NS::UInteger>(slot.height - 1,
                                (y * slot.height) / std::max<NS::UInteger>(proxyHeight, 1)));
    uint8_t *dstRow = data.data() + static_cast<size_t>(y) * proxyRowBytes;
    for (NS::UInteger x = 0; x < proxyWidth; ++x) {
      NS::UInteger srcX = static_cast<NS::UInteger>(
          std::min<NS::UInteger>(slot.width - 1,
                                  (x * slot.width) /
                                      std::max<NS::UInteger>(proxyWidth, 1)));
      MTL::Region region = MTL::Region::Make2D(srcX, srcY, 1, 1);
      slot.texture->getBytes(pixel.data(), pixelBytes, region, 0);
      std::memcpy(dstRow + static_cast<size_t>(x) * pixelBytes, pixel.data(),
                  pixelBytes);
    }
  }

  slot.historyBacking = ManagedTextureSlot::HistoryBacking::CpuData;
  slot.historyIsProxy = proxyWidth != slot.width || proxyHeight != slot.height;
  slot.historyWidth = proxyWidth;
  slot.historyHeight = proxyHeight;
  slot.historyBytesPerRow = proxyRowBytes;
  slot.historyData = std::move(data);
  slot.needsGpuRefresh = true;
  slot.stagingValid = false;
  return true;
}

bool Renderer::uploadHistoryToTexture(ManagedTextureSlot &slot,
                                      MTL::CommandBuffer *cmd,
                                      MTL::BlitCommandEncoder *&blit) {
  if (!slot.texture)
    return false;

  size_t pixelBytes = bytesPerPixel(slot.pixelFormat);
  if (pixelBytes == 0)
    return false;

  if (slot.historyBacking == ManagedTextureSlot::HistoryBacking::SharedBuffer) {
    if (!slot.stagingBuffer || !slot.stagingValid)
      return false;

    size_t totalBytes = textureByteSize(slot);
    if (totalBytes == 0 || slot.stagingCapacity < totalBytes)
      return false;

    if (cmd) {
      if (!blit)
        blit = cmd->blitCommandEncoder();
      if (blit) {
        MTL::Size size = MTL::Size::Make(slot.historyWidth, slot.historyHeight, 1);
        MTL::Origin origin = MTL::Origin::Make(0, 0, 0);
        NS::UInteger bytesPerImage = static_cast<NS::UInteger>(
            slot.historyBytesPerRow * slot.historyHeight);
        blit->copyFromBuffer(slot.stagingBuffer, 0, slot.historyBytesPerRow,
                             bytesPerImage, size, slot.texture, 0, 0, origin);
        slot.needsGpuRefresh = false;
        return true;
      }
    }

    if (void *contents = slot.stagingBuffer->contents()) {
      const uint8_t *src = static_cast<const uint8_t *>(contents);
      NS::UInteger uploadWidth = std::max<NS::UInteger>(1, slot.historyWidth);
      NS::UInteger uploadHeight = std::max<NS::UInteger>(1, slot.historyHeight);
      MTL::Region region = MTL::Region::Make2D(0, 0, uploadWidth, uploadHeight);
      slot.texture->replaceRegion(region, 0, src, slot.historyBytesPerRow);
      slot.needsGpuRefresh = false;
      return true;
    }

    return false;
  }

  if (slot.historyBacking == ManagedTextureSlot::HistoryBacking::CpuData) {
    if (slot.historyData.empty() || slot.historyWidth == 0 ||
        slot.historyHeight == 0)
      return false;

    std::vector<uint8_t> expanded;
    const uint8_t *srcData = slot.historyData.data();
    size_t srcRowBytes = slot.historyBytesPerRow;

    if (slot.historyIsProxy &&
        (slot.historyWidth != slot.width || slot.historyHeight != slot.height)) {
      size_t expandedBytes = static_cast<size_t>(slot.width) * slot.height *
                             pixelBytes;
      expanded.resize(expandedBytes);
      for (NS::UInteger y = 0; y < slot.height; ++y) {
        NS::UInteger srcY = static_cast<NS::UInteger>(
            std::min<NS::UInteger>(slot.historyHeight - 1,
                                    (y * slot.historyHeight) /
                                        std::max<NS::UInteger>(slot.height, 1)));
        const uint8_t *srcRow =
            srcData + static_cast<size_t>(srcY) * slot.historyBytesPerRow;
        uint8_t *dstRow = expanded.data() +
                          static_cast<size_t>(y) * slot.width * pixelBytes;
        for (NS::UInteger x = 0; x < slot.width; ++x) {
          NS::UInteger srcX = static_cast<NS::UInteger>(
              std::min<NS::UInteger>(slot.historyWidth - 1,
                                      (x * slot.historyWidth) /
                                          std::max<NS::UInteger>(slot.width, 1)));
          const uint8_t *srcPixel =
              srcRow + static_cast<size_t>(srcX) * pixelBytes;
          std::memcpy(dstRow + static_cast<size_t>(x) * pixelBytes, srcPixel,
                      pixelBytes);
        }
      }
      srcData = expanded.data();
      srcRowBytes = static_cast<size_t>(slot.width) * pixelBytes;
    }

    MTL::Region region =
        MTL::Region::Make2D(0, 0, slot.width, slot.height);
    slot.texture->replaceRegion(region, 0, srcData, srcRowBytes);
    slot.needsGpuRefresh = false;
    return true;
  }

  return false;
}

MTL::Texture *Renderer::requestResidentTexture(ManagedTextureSlot &slot,
                                               MTL::CommandBuffer *cmd,
                                               MTL::BlitCommandEncoder *&blit) {
  if (slot.texture || !slot.descriptorValid || slot.width == 0 ||
      slot.height == 0) {
    if (slot.texture)
      slot.lastUsedFrame = _renderedFrameCount;
    if (slot.texture && slot.needsGpuRefresh &&
        slot.historyBacking != ManagedTextureSlot::HistoryBacking::None) {
      if (uploadHistoryToTexture(slot, cmd, blit)) {
        _historyStreamingActive = true;
        ++_historyStreamingRestoreCount;
        if (slot.historyIsProxy)
          _historyStreamingUsedProxy = true;
        const char *source =
            slot.historyBacking == ManagedTextureSlot::HistoryBacking::SharedBuffer
                ? "staging buffer"
                : (slot.historyIsProxy ? "CPU proxy" : "CPU history");
        std::printf(
            "[TextureResidency] Refreshed slot %s from %s (%ux%u).\n",
            textureSlotLabel(slot), source,
            static_cast<unsigned>(slot.historyWidth),
            static_cast<unsigned>(slot.historyHeight));
      } else {
        std::printf(
            "[TextureResidency] Failed to refresh resident slot %s from history;"
            " history unavailable.\n",
            textureSlotLabel(slot));
        slot.needsGpuRefresh = false;
        if (!slot.stagingValid &&
            slot.historyBacking == ManagedTextureSlot::HistoryBacking::CpuData)
          slot.historyData.clear();
        _needsAccumulationReset = true;
        _accumulationTargetsNeedClear = true;
      }
    }
    return slot.texture;
  }

  const char *label = textureSlotLabel(slot);
  MTL::TextureDescriptor *descriptor =
      MTL::TextureDescriptor::alloc()->init();
  descriptor->setTextureType(slot.textureType);
  descriptor->setWidth(slot.width);
  descriptor->setHeight(slot.height);
  descriptor->setPixelFormat(slot.pixelFormat);
  descriptor->setUsage(slot.usage);
  descriptor->setStorageMode(slot.storageMode);

  slot.texture = _pDevice->newTexture(descriptor);
  descriptor->release();

  if (!slot.texture) {
    std::printf("[TextureResidency] Failed to restore slot %s: texture allocation returned null.\n",
                label);
    _needsAccumulationReset = true;
    _accumulationTargetsNeedClear = true;
    return nullptr;
  }

  bool logged = false;
  if (slot.historyBacking != ManagedTextureSlot::HistoryBacking::None) {
    if (uploadHistoryToTexture(slot, cmd, blit)) {
      logged = true;
      _historyStreamingActive = true;
      ++_historyStreamingRestoreCount;
      if (slot.historyIsProxy)
        _historyStreamingUsedProxy = true;
      const char *source =
          slot.historyBacking == ManagedTextureSlot::HistoryBacking::SharedBuffer
              ? "staging buffer"
              : (slot.historyIsProxy ? "CPU proxy" : "CPU history");
      std::printf(
          "[TextureResidency] Restored slot %s from %s (%ux%u).\n", label,
          source, static_cast<unsigned>(slot.historyWidth),
          static_cast<unsigned>(slot.historyHeight));
    } else {
      std::printf(
          "[TextureResidency] Restored slot %s without history data.\n", label);
      clearTextureHistory(slot);
      _needsAccumulationReset = true;
      _accumulationTargetsNeedClear = true;
    }
  } else {
    std::printf("[TextureResidency] Restored slot %s without staging data.\n",
                label);
  }

  slot.lastUsedFrame = _renderedFrameCount;
  return slot.texture;
}

bool Renderer::evictTextureSlot(ManagedTextureSlot &slot,
                                MTL::CommandBuffer *cmd,
                                MTL::BlitCommandEncoder *&blit) {
  if (!slot.texture)
    return false;

  const char *label = textureSlotLabel(slot);
  size_t totalBytes = textureByteSize(slot);
  bool preservedHistory = false;
  if (totalBytes == 0 || !cmd) {
    std::string reason;
    if (totalBytes == 0)
      reason += "zero-sized texture";
    if (!cmd) {
      if (!reason.empty())
        reason += ", ";
      reason += "no command buffer";
    }

    if (captureCpuHistoryProxy(slot)) {
      preservedHistory = true;
      std::printf(
          "[TextureResidency] Evicting slot %s: captured CPU %s (%ux%u).\n",
          label, slot.historyIsProxy ? "proxy" : "copy",
          static_cast<unsigned>(slot.historyWidth),
          static_cast<unsigned>(slot.historyHeight));
    } else {
      std::printf("[TextureResidency] Evicting slot %s: releasing without staging"
                  " (%s).\n",
                  label, reason.c_str());
    }
  } else {
    ensureBufferCapacity(slot.stagingBuffer, totalBytes, slot.stagingCapacity,
                         false, MTL::ResourceStorageModeShared);
    if (!slot.stagingBuffer) {
      if (captureCpuHistoryProxy(slot)) {
        preservedHistory = true;
        std::printf("[TextureResidency] Evicting slot %s: captured CPU %s (%ux%u)"
                    " after staging allocation failure.\n",
                    label, slot.historyIsProxy ? "proxy" : "copy",
                    static_cast<unsigned>(slot.historyWidth),
                    static_cast<unsigned>(slot.historyHeight));
      } else {
        std::printf("[TextureResidency] Evicting slot %s: releasing without staging"
                    " (failed to allocate staging buffer).\n",
                    label);
      }
    } else {
      if (!blit)
        blit = cmd->blitCommandEncoder();
      if (!blit) {
        if (captureCpuHistoryProxy(slot)) {
          preservedHistory = true;
          std::printf(
              "[TextureResidency] Evicting slot %s: captured CPU %s (%ux%u) after"
              " blit encoder failure.\n",
              label, slot.historyIsProxy ? "proxy" : "copy",
              static_cast<unsigned>(slot.historyWidth),
              static_cast<unsigned>(slot.historyHeight));
        } else {
          std::printf("[TextureResidency] Evicting slot %s: releasing without staging"
                      " (failed to create blit encoder).\n",
                      label);
        }
      } else {
        size_t rowBytes = slot.width * bytesPerPixel(slot.pixelFormat);
        size_t alignedRowBytes = alignTo(rowBytes, 256);
        NS::UInteger bytesPerImage =
            static_cast<NS::UInteger>(alignedRowBytes * slot.height);
        MTL::Size size = MTL::Size::Make(slot.width, slot.height, 1);
        MTL::Origin origin = MTL::Origin::Make(0, 0, 0);
        blit->copyFromTexture(slot.texture, 0, 0, origin, size,
                              slot.stagingBuffer, 0, alignedRowBytes,
                              bytesPerImage);

        slot.stagingValid = true;
        slot.historyBacking = ManagedTextureSlot::HistoryBacking::SharedBuffer;
        slot.historyIsProxy = false;
        slot.historyWidth = slot.width;
        slot.historyHeight = slot.height;
        slot.historyBytesPerRow = alignedRowBytes;
        slot.needsGpuRefresh = true;
        preservedHistory = true;

        std::printf(
            "[TextureResidency] Evicting slot %s: copied %zu bytes to staging buffer"
            " before release.\n",
            label, totalBytes);
      }
    }
  }

  if (!preservedHistory) {
    clearTextureHistory(slot);
    slot.stagingValid = false;
    _needsAccumulationReset = true;
    _accumulationTargetsNeedClear = true;
  }

  if (slot.texture) {
    slot.texture->release();
    slot.texture = nullptr;
  }
  return true;
}

void Renderer::updateTextureResidency(MTL::CommandBuffer *cmd) {
  if (!cmd || _needsAccumulationReset)
    return;

  bool belowBudget = _residentPrimitiveCount < kTextureResidencyPrimitiveBudget;
  double totalMemoryMB = currentGPUMemoryMB();
  double scratchMB = scratchMemoryMB();
  double residencyMB = std::max(0.0, totalMemoryMB - scratchMB);
  bool overMemory = residencyMB > _textureResidencyMemoryCapMB;

  if (!overMemory && totalMemoryMB > _textureResidencyMemoryCapMB &&
      scratchMB > 0.0) {
    std::printf("[TextureResidency] Skipping eviction: total=%.2f MB, scratch=%.2f MB, "
                "residency=%.2f MB (cap=%.2f MB).\n",
                totalMemoryMB, scratchMB, residencyMB, _textureResidencyMemoryCapMB);
  }
  if (!belowBudget && !overMemory)
    return;

  if (overMemory) {
    std::printf("[TextureResidency] Triggering eviction: residency=%.2f MB (total=%.2f MB, "
                "scratch=%.2f MB, cap=%.2f MB).\n",
                residencyMB, totalMemoryMB, scratchMB, _textureResidencyMemoryCapMB);
  }

  std::array<ManagedTextureSlot *, 6> slots = {
      &_accumulationSlots[0], &_accumulationSlots[1], &_sampleCountSlot,
      &_sampleImportanceSlot, &_albedoSlot, &_normalSlot};

  struct ResidencyCandidate {
    ManagedTextureSlot *slot = nullptr;
    double score = 0.0;
    double bytesMB = 0.0;
  };

  auto scoreSlot = [&](ManagedTextureSlot *slot) {
    ResidencyCandidate candidate{};
    candidate.slot = slot;
    size_t bytes = textureByteSize(*slot);
    candidate.bytesMB = static_cast<double>(bytes) / (1024.0 * 1024.0);
    uint64_t frameAge = _renderedFrameCount > slot->lastUsedFrame
                            ? _renderedFrameCount - slot->lastUsedFrame
                            : 0;
    double recency = 1.0 / (1.0 + static_cast<double>(frameAge));
    candidate.score = recency / (1.0 + candidate.bytesMB);
    return candidate;
  };

  std::vector<ResidencyCandidate> candidates;
  candidates.reserve(slots.size());
  for (ManagedTextureSlot *slot : slots) {
    if (!slot->texture)
      continue;
    candidates.push_back(scoreSlot(slot));
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const ResidencyCandidate &a, const ResidencyCandidate &b) {
              if (a.score == b.score)
                return a.bytesMB > b.bytesMB;
              return a.score < b.score;
            });

  bool stopOnMemory = overMemory;
  bool stopOnBudget = !belowBudget;
  bool needMemoryEviction = overMemory;
  bool primitiveBudgetSatisfied = belowBudget;
  MTL::BlitCommandEncoder *blit = nullptr;
  for (const ResidencyCandidate &candidate : candidates) {
    bool needMoreEviction =
        (stopOnMemory && needMemoryEviction) ||
        (stopOnBudget && !primitiveBudgetSatisfied);
    if (!needMoreEviction)
      break;
    if (!evictTextureSlot(*candidate.slot, cmd, blit))
      continue;
    residencyMB = std::max(0.0, residencyMB - candidate.bytesMB);
    needMemoryEviction = residencyMB > _textureResidencyMemoryCapMB;
    primitiveBudgetSatisfied =
        _residentPrimitiveCount < kTextureResidencyPrimitiveBudget;
  }
  if (blit)
    blit->endEncoding();
}

void Renderer::clearMaterialTextures() {
  for (MTL::Texture *texture : _materialTextures) {
    if (texture)
      texture->release();
  }
  _materialTextures.clear();
}

void Renderer::releaseEnvironmentTexture() {
  if (_environmentTexture) {
    _environmentTexture->release();
    _environmentTexture = nullptr;
  }
  _environmentTexturePath.clear();
}

void Renderer::rebuildEnvironmentTexture() {
  std::string desiredPath;
  float desiredBrightness = 1.0f;
  if (_pScene) {
    desiredPath = _pScene->getEnvironmentTexturePath();
    desiredBrightness = _pScene->getEnvironmentBrightness();
  }

  _environmentBrightness = std::max(desiredBrightness, 0.0f);

  if (!_environmentSampler && _pDevice)
    _environmentSampler = CreateEnvironmentSampler(_pDevice);

  if (desiredPath.empty()) {
    releaseEnvironmentTexture();
    return;
  }

  if (!_environmentSampler)
    return;

  if (_environmentTexture && desiredPath == _environmentTexturePath)
    return;

  releaseEnvironmentTexture();

  EnvironmentTextureData data;
  if (!LoadEnvironmentTextureData(desiredPath, data))
    return;

  MTL::Texture *texture = CreateEnvironmentTexture(_pDevice, data);
  if (!texture) {
    std::printf("[Renderer] Failed to allocate environment texture for %s\n",
               desiredPath.c_str());
    return;
  }

  _environmentTexture = texture;
  _environmentTexturePath = desiredPath;
}

void Renderer::rebuildMaterialTextures() {
  clearMaterialTextures();

  if (!_pDevice)
    return;

  size_t availableSlots = static_cast<size_t>(kMaxMaterialTextureSlots);
  size_t requestedTextures = _cachedTextureInfos.size();
  if (requestedTextures > availableSlots) {
    std::printf(
        "[Renderer] Truncating material textures from %zu to %u to fit shader "
        "bind slots.\n",
        requestedTextures, kMaxMaterialTextureSlots);
  }

  size_t maxTextures = std::min(requestedTextures, availableSlots);
  size_t totalTexels = _cachedTextureData.size();
  _materialTextures.reserve(maxTextures);

  auto createFallbackTexture = [&]() -> MTL::Texture * {
    MTL::TextureDescriptor *descriptor = MTL::TextureDescriptor::alloc()->init();
    descriptor->setTextureType(MTL::TextureType::TextureType2D);
    descriptor->setPixelFormat(MTL::PixelFormat::PixelFormatRGBA32Float);
    descriptor->setWidth(1);
    descriptor->setHeight(1);
    descriptor->setUsage(MTL::TextureUsageShaderRead);
    descriptor->setStorageMode(MTL::StorageMode::StorageModeManaged);

    MTL::Texture *texture = _pDevice->newTexture(descriptor);
    descriptor->release();
    if (!texture)
      return nullptr;

    simd::float4 pixel = simd::float4{1.0f, 1.0f, 1.0f, 1.0f};
    NS::UInteger bytesPerRow = sizeof(simd::float4);
    MTL::Region region = MTL::Region::Make2D(0, 0, 1, 1);
    texture->replaceRegion(region, 0, &pixel, bytesPerRow);
    return texture;
  };

  for (size_t texIndex = 0; texIndex < maxTextures; ++texIndex) {
    const TextureInfo &info = _cachedTextureInfos[texIndex];
    size_t texelCount = static_cast<size_t>(info.width) * info.height;
    bool hasValidDimensions = info.width > 0 && info.height > 0;
    bool hasValidRange = info.offset < totalTexels &&
                         texelCount > 0 &&
                         info.offset + texelCount <= totalTexels;

    if (!hasValidDimensions || !hasValidRange) {
      MTL::Texture *fallback = createFallbackTexture();
      _materialTextures.push_back(fallback);
      continue;
    }

    MTL::TextureDescriptor *descriptor = MTL::TextureDescriptor::alloc()->init();
    descriptor->setTextureType(MTL::TextureType::TextureType2D);
    descriptor->setPixelFormat(MTL::PixelFormat::PixelFormatRGBA32Float);
    descriptor->setWidth(info.width);
    descriptor->setHeight(info.height);
    descriptor->setUsage(MTL::TextureUsageShaderRead);
    descriptor->setStorageMode(MTL::StorageMode::StorageModeManaged);

    MTL::Texture *texture = _pDevice->newTexture(descriptor);
    descriptor->release();

    if (!texture) {
      std::printf(
          "[Renderer] Failed to allocate material texture %zu (%ux%u).\n",
          texIndex, info.width, info.height);
      MTL::Texture *fallback = createFallbackTexture();
      _materialTextures.push_back(fallback);
      continue;
    }

    const simd::float4 *src = _cachedTextureData.data() + info.offset;
    NS::UInteger bytesPerRow =
        static_cast<NS::UInteger>(info.width * sizeof(simd::float4));
    MTL::Region region = MTL::Region::Make2D(0, 0, info.width, info.height);
    texture->replaceRegion(region, 0, src, bytesPerRow);

    _materialTextures.push_back(texture);
  }
}

void Renderer::buildTextures() {
  NS::UInteger width = std::max<NS::UInteger>(
      1, static_cast<NS::UInteger>(Camera::screenSize.x));
  NS::UInteger height = std::max<NS::UInteger>(
      1, static_cast<NS::UInteger>(Camera::screenSize.y));

  MTL::TextureUsage usage = static_cast<MTL::TextureUsage>(
      MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);

  for (auto &slot : _accumulationSlots)
    configureTextureSlot(slot, width, height,
                         MTL::PixelFormat::PixelFormatRGBA16Float, usage);
  configureTextureSlot(_sampleCountSlot, width, height,
                       MTL::PixelFormat::PixelFormatR16Float, usage);
  configureTextureSlot(_sampleImportanceSlot, width, height,
                       MTL::PixelFormat::PixelFormatR16Float, usage);
  configureTextureSlot(_albedoSlot, width, height,
                       MTL::PixelFormat::PixelFormatRGBA16Float, usage);
  configureTextureSlot(_normalSlot, width, height,
                       MTL::PixelFormat::PixelFormatRGBA16Float, usage);

  _needsAccumulationReset = true;
  _accumulationTargetsNeedClear = true;
}

bool Renderer::resetAccumulationTargets(MTL::CommandBuffer *cmd) {
  if (!cmd)
    return false;

  std::array<ManagedTextureSlot *, 6> slots = {
      &_accumulationSlots[0], &_accumulationSlots[1], &_sampleCountSlot,
      &_sampleImportanceSlot, &_albedoSlot, &_normalSlot};

  bool anyDescriptors = false;
  bool allResident = true;
  size_t maxBytes = 0;
  for (ManagedTextureSlot *slot : slots) {
    if (!slot->descriptorValid)
      continue;
    anyDescriptors = true;
    if (!slot->texture)
      allResident = false;
    size_t bytes = textureByteSize(*slot);
    maxBytes = std::max(maxBytes, bytes);
  }

  if (!anyDescriptors)
    return true;

  if (maxBytes == 0)
    return allResident;

  ensureBufferCapacity(_pTextureClearBuffer, maxBytes,
                       _textureClearBufferCapacity, false,
                       MTL::ResourceStorageModeShared);
  if (!_pTextureClearBuffer)
    return false;

  if (void *ptr = _pTextureClearBuffer->contents())
    std::memset(ptr, 0, maxBytes);

  MTL::BlitCommandEncoder *blit = cmd->blitCommandEncoder();
  if (!blit)
    return false;

  for (ManagedTextureSlot *slot : slots) {
    if (!slot->texture)
      continue;
    size_t pixelBytes = bytesPerPixel(slot->pixelFormat);
    if (pixelBytes == 0)
      continue;
    size_t rowBytes = slot->width * pixelBytes;
    size_t alignedRowBytes = alignTo(rowBytes, 256);
    size_t totalBytes = alignedRowBytes * slot->height;
    if (totalBytes == 0)
      continue;
    MTL::Size size = MTL::Size::Make(slot->width, slot->height, 1);
    MTL::Origin origin = MTL::Origin::Make(0, 0, 0);
    blit->copyFromBuffer(_pTextureClearBuffer, 0, alignedRowBytes, totalBytes,
                         size, slot->texture, 0, 0, origin);
    slot->stagingValid = false;
  }

  blit->endEncoding();
  return allResident;
}

void Renderer::updateAdaptiveSamplingMaps(MTL::CommandBuffer *pCmd) {
  if (!_pAdaptiveSamplingPSO || !pCmd)
    return;
  MTL::Texture *importance = _sampleImportanceSlot.texture;
  MTL::Texture *accumulation = _accumulationSlots[0].texture;
  if (!importance || !accumulation || !_pUniformsBuffer)
    return;

  NS::UInteger width = importance->width();
  NS::UInteger height = importance->height();
  if (width == 0 || height == 0)
    return;

  MTL::ComputeCommandEncoder *pCompute = pCmd->computeCommandEncoder();
  if (!pCompute)
    return;

  pCompute->setComputePipelineState(_pAdaptiveSamplingPSO);
  pCompute->setTexture(accumulation, 0);
  pCompute->setTexture(importance, 1);
  pCompute->setBuffer(_pUniformsBuffer, 0, 0);

  const NS::UInteger threadWidth = 8;
  const NS::UInteger threadHeight = 8;
  MTL::Size threadsPerThreadgroup =
      MTL::Size::Make(threadWidth, threadHeight, 1);
  MTL::Size threadgroups = MTL::Size::Make(
      (width + threadWidth - 1) / threadWidth,
      (height + threadHeight - 1) / threadHeight, 1);

  pCompute->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
  pCompute->endEncoding();
}

bool Renderer::updateCameraStates() {
  const auto &path = _pScene->cameraPath;
  bool hadObserver = _observerActive;
  Camera::State previousViewState =
      hadObserver ? _observerCameraState : _primaryCameraState;

  bool toggled = false;
  if (InputSystem::observerToggleRequest) {
    _observerActive = !_observerActive;
    InputSystem::observerToggleRequest = false;
    toggled = true;
  }

  double originalDelta = _deltaTimeSeconds;

  if (!path.empty()) {
    Camera::State newState = _primaryCameraState;
    if (_animationFrame <= path.front().frame) {
      const auto &k = path.front();
      newState = makeCameraState(k.position, k.lookAt,
                                 _primaryCameraState.verticalFov,
                                 _primaryCameraState.focalLength,
                                 _primaryCameraState.forward);
    } else if (_animationFrame >= path.back().frame) {
      const auto &k = path.back();
      newState = makeCameraState(k.position, k.lookAt,
                                 _primaryCameraState.verticalFov,
                                 _primaryCameraState.focalLength,
                                 _primaryCameraState.forward);
    } else {
      for (size_t i = 0; i + 1 < path.size(); ++i) {
        const auto &k0 = path[i];
        const auto &k1 = path[i + 1];
        if (_animationFrame >= k0.frame && _animationFrame <= k1.frame) {
          float t =
              float(_animationFrame - k0.frame) / float(k1.frame - k0.frame);
          simd::float3 position =
              k0.position + t * (k1.position - k0.position);
          simd::float3 look = k0.lookAt + t * (k1.lookAt - k0.lookAt);
          newState = makeCameraState(position, look,
                                     _primaryCameraState.verticalFov,
                                     _primaryCameraState.focalLength,
                                     _primaryCameraState.forward);
          break;
        }
      }
    }

    if (cameraStatesDiffer(newState, _primaryCameraState))
      _primaryCameraState = newState;

    _deltaTimeSeconds = 0.0;
    Camera::deltaTime = 0.0f;

    Camera::applyState(_observerCameraState);
    Camera::deltaTime = static_cast<float>(originalDelta);
    if (Camera::transformWithInputs())
      _observerCameraState = Camera::captureState();

    _animationFrame++;
  } else {
    Camera::State *target =
        _observerActive ? &_observerCameraState : &_primaryCameraState;
    Camera::applyState(*target);
    Camera::deltaTime = static_cast<float>(_deltaTimeSeconds);
    if (Camera::transformWithInputs())
      *target = Camera::captureState();
  }

  const Camera::State &activeView =
      _observerActive ? _observerCameraState : _primaryCameraState;
  Camera::applyState(activeView);
  Camera::deltaTime = _observerActive ? 0.0f
                                      : static_cast<float>(_deltaTimeSeconds);

  bool viewChanged = toggled || cameraStatesDiffer(activeView, previousViewState) ||
                     hadObserver != _observerActive;
  if (viewChanged) {
    ++_cameraVersion;
    recalculateViewport();
  }

  return viewChanged;
}

void Renderer::updateUniforms(bool cameraChanged) {
  UniformsData &u = *((UniformsData *)_pUniformsBuffer->contents());

  if (cameraChanged)
    _needsAccumulationReset = true;

  if (_needsAccumulationReset) {
    if (!_accumulationTargetsNeedClear) {
      _accumulationTargetsNeedClear = true;
      u.randomSeed = {randomFloat(), randomFloat(), randomFloat()};
    }
    u.frameCount = 0;
  } else {
    u.frameCount++;
  }

  uint32_t minSamples = std::min(_minSamplesPerPixel, _maxSamplesPerPixel);
  uint32_t maxSamples = std::max(_minSamplesPerPixel, _maxSamplesPerPixel);
  minSamples = std::max<uint32_t>(minSamples, 1);
  maxSamples = std::max(maxSamples, minSamples);

  u.sampleCountTextureIndex = 2;
  u.sampleImportanceTextureIndex = 3;
  u.minSamplesPerPixel = minSamples;
  u.maxSamplesPerPixel = maxSamples;
  size_t boundTextureCount = std::min(
      _materialTextures.size(), static_cast<size_t>(kMaxMaterialTextureSlots));
  u.textureCount = static_cast<uint32_t>(boundTextureCount);
  u.environmentMapEnabled =
      (_environmentTexture && _environmentSampler) ? 1u : 0u;
  u.environmentMapIntensity = _environmentBrightness;
  u.environmentPadding0 = 0.0f;
  u.environmentPadding1 = 0.0f;

  uint64_t residentPrimitiveCount = _residentPrimitiveCount;
  uint64_t residentTriangleCount = _residentTriangleCount;

  if (_useAccelerationStructureBindings && _pTlasStructure &&
      _pGeometryHandleBuffer) {
    residentPrimitiveCount = 0;
    residentTriangleCount = 0;
    size_t sharedCount =
        std::min(_residentObjectGpuResources.size(), _instanceRecords.size());
    for (size_t i = 0; i < sharedCount; ++i) {
      const auto &resident = _residentObjectGpuResources[i];
      if (!resident.isResident() || !resident.geometryValid)
        continue;
      residentPrimitiveCount += _instanceRecords[i].primitiveCount;
      residentTriangleCount += resident.triangleCount;
    }
  }

  uint64_t totalPrimitiveCount = 0;
  for (const auto &record : _instanceRecords)
    totalPrimitiveCount += record.primitiveCount;
  if (totalPrimitiveCount == 0)
    totalPrimitiveCount = _allPrimitives.size();

  u.primitiveCount = residentPrimitiveCount;
  u.triangleCount = residentTriangleCount;
  u.totalPrimitiveCount = totalPrimitiveCount;
  u.tlasNodeCount = _tlasNodeCount;
  u.blasNodeCount = _blasNodeCount;
  u.maxRayDepth = _pScene->maxRayDepth;
  u.debugAS = InputSystem::debugAS;
  u.lightCount = static_cast<uint32_t>(_lightCount);
  u.lightTotalWeight = _lightTotalWeight;

  markBufferModified(_pUniformsBuffer, NS::Range::Make(0, sizeof(UniformsData)));
}

void Renderer::draw(MTK::View *pView) {
  processRayHitCounters();
  bool cameraChanged = updateCameraStates();
  Camera::State viewCamera =
      _observerActive ? _observerCameraState : _primaryCameraState;

  Camera::applyState(_primaryCameraState);
  updateResidency();

  Camera::applyState(viewCamera);
  if (_captureOutputsPending.load(std::memory_order_acquire))
    processPendingCapturedFrames();

  Camera::deltaTime =
      _observerActive ? 0.0f : static_cast<float>(_deltaTimeSeconds);
  updateUniforms(cameraChanged);
  beginFrameMetrics();
  std::swap(_accumulationSlots[0], _accumulationSlots[1]);

  _historyStreamingActive = false;
  _historyStreamingRestoreCount = 0;
  _historyStreamingUsedProxy = false;

  uint64_t frameIndex = _renderedFrameCount;
  bool captureThisFrame = _frameCaptureEnabled && _frameCaptureInterval > 0 &&
                          (frameIndex % _frameCaptureInterval == 0);

  NS::AutoreleasePool *pPool = NS::AutoreleasePool::alloc()->init();

  auto captureList =
      std::make_shared<std::vector<std::shared_ptr<FrameCaptureRequest>>>();

  MTL::CommandBuffer *prepCmd = _pCommandQueue->commandBuffer();
  if (!prepCmd) {
    if (_benchmarkEnabled && !_pendingBenchmarkSamples.empty())
      _pendingBenchmarkSamples.pop_back();
    if (_accumulationTargetsNeedClear) {
      _accumulationTargetsNeedClear = false;
      _needsAccumulationReset = false;
    }
    completeFrameMetrics(nullptr);
    pPool->release();
    return;
  }

  bool belowBudget =
      _residentPrimitiveCount < kTextureResidencyPrimitiveBudget;
  double contentMemory = residencyMemoryMB();
  bool overCap = contentMemory > _textureResidencyMemoryCapMB;

  if (!_needsAccumulationReset && (belowBudget || overCap))
    updateTextureResidency(prepCmd);

  bool allowResidency =
      _needsAccumulationReset || (!belowBudget && !overCap);

  MTL::BlitCommandEncoder *restoreBlit = nullptr;
  MTL::Texture *accum0 = _accumulationSlots[0].texture;
  MTL::Texture *accum1 = _accumulationSlots[1].texture;
  MTL::Texture *sampleCount = _sampleCountSlot.texture;
  MTL::Texture *sampleImportance = _sampleImportanceSlot.texture;
  MTL::Texture *albedoTexture = _albedoSlot.texture;
  MTL::Texture *normalTexture = _normalSlot.texture;
  if (allowResidency) {
    accum0 = requestResidentTexture(_accumulationSlots[0], prepCmd, restoreBlit);
    accum1 = requestResidentTexture(_accumulationSlots[1], prepCmd, restoreBlit);
    sampleCount = requestResidentTexture(_sampleCountSlot, prepCmd, restoreBlit);
    sampleImportance =
        requestResidentTexture(_sampleImportanceSlot, prepCmd, restoreBlit);
    albedoTexture = requestResidentTexture(_albedoSlot, prepCmd, restoreBlit);
    normalTexture = requestResidentTexture(_normalSlot, prepCmd, restoreBlit);
  }

  auto ensureSlotReady = [&](ManagedTextureSlot &slot, MTL::Texture *&handle) {
    if (!handle &&
        slot.historyBacking != ManagedTextureSlot::HistoryBacking::None) {
      handle = requestResidentTexture(slot, prepCmd, restoreBlit);
    } else if (handle && slot.needsGpuRefresh &&
               slot.historyBacking != ManagedTextureSlot::HistoryBacking::None) {
      requestResidentTexture(slot, prepCmd, restoreBlit);
    }
  };

  ensureSlotReady(_accumulationSlots[0], accum0);
  ensureSlotReady(_accumulationSlots[1], accum1);
  ensureSlotReady(_sampleCountSlot, sampleCount);
  ensureSlotReady(_sampleImportanceSlot, sampleImportance);
  ensureSlotReady(_albedoSlot, albedoTexture);
  ensureSlotReady(_normalSlot, normalTexture);

  auto markSlotUsed = [&](ManagedTextureSlot &slot, MTL::Texture *handle) {
    if (handle)
      slot.lastUsedFrame = _renderedFrameCount;
  };

  markSlotUsed(_accumulationSlots[0], accum0);
  markSlotUsed(_accumulationSlots[1], accum1);
  markSlotUsed(_sampleCountSlot, sampleCount);
  markSlotUsed(_sampleImportanceSlot, sampleImportance);
  markSlotUsed(_albedoSlot, albedoTexture);
  markSlotUsed(_normalSlot, normalTexture);

  if (restoreBlit)
    restoreBlit->endEncoding();

  bool haveAllTextures =
      accum0 && accum1 && sampleCount && sampleImportance && albedoTexture &&
      normalTexture;

  if (_accumulationTargetsNeedClear && haveAllTextures) {
    if (resetAccumulationTargets(prepCmd)) {
      _accumulationTargetsNeedClear = false;
      _needsAccumulationReset = false;
    }
  }

  updateAdaptiveSamplingMaps(prepCmd);

  prepCmd->commit();

  if (!haveAllTextures) {
    MTL::CommandBuffer *presentCmd = _pCommandQueue->commandBuffer();
    if (presentCmd) {
      presentCmd->addCompletedHandler(
          [this, captureList](MTL::CommandBuffer *cmd) {
            this->completeFrameMetrics(cmd);
            for (const auto &capture : *captureList)
              this->finalizeFrameCapture(capture);
          });
      MTL::Drawable *drawable = pView->currentDrawable();
      if (drawable)
        presentCmd->presentDrawable(drawable);
      trackFrameCommandBuffer(presentCmd);
      presentCmd->commit();
    } else {
      completeFrameMetrics(nullptr);
    }
    ++_renderedFrameCount;
    pPool->release();
    return;
  }

  uint32_t minSamples = 1;
  uint32_t maxSamples = 1;
  std::vector<TileDispatchRegion> tiles;
  if (_pPathTracePSO && accum1) {
    NS::UInteger width = accum1->width();
    NS::UInteger height = accum1->height();
    if (width > 0 && height > 0) {
      NS::UInteger tileWidth =
          std::max<NS::UInteger>(kPathTraceTileWidth, 1);
      NS::UInteger tileHeight =
          std::max<NS::UInteger>(kPathTraceTileHeight, 1);

      minSamples = std::min(_minSamplesPerPixel, _maxSamplesPerPixel);
      maxSamples = std::max(_minSamplesPerPixel, _maxSamplesPerPixel);
      minSamples = std::max<uint32_t>(minSamples, 1);
      maxSamples = std::max<uint32_t>(maxSamples, minSamples);

      if (maxSamples > 1) {
        double sampleScale =
            std::max<double>(1.0, std::sqrt(static_cast<double>(maxSamples)));
        double scaledWidth =
            std::floor(static_cast<double>(tileWidth) / sampleScale);
        double scaledHeight =
            std::floor(static_cast<double>(tileHeight) / sampleScale);

        tileWidth = std::max<NS::UInteger>(
            1, static_cast<NS::UInteger>(std::max(1.0, scaledWidth)));
        tileHeight = std::max<NS::UInteger>(
            1, static_cast<NS::UInteger>(std::max(1.0, scaledHeight)));
      }
      for (NS::UInteger y = 0; y < height; y += tileHeight) {
        NS::UInteger actualHeight = std::min(tileHeight, height - y);
        for (NS::UInteger x = 0; x < width; x += tileWidth) {
          NS::UInteger actualWidth = std::min(tileWidth, width - x);
          TileDispatchRegion region{};
          region.originX = static_cast<uint32_t>(x);
          region.originY = static_cast<uint32_t>(y);
          region.width = static_cast<uint32_t>(actualWidth);
          region.height = static_cast<uint32_t>(actualHeight);
          tiles.push_back(region);
        }
      }
    }
  }

  if (_pPathTracePSO && !tiles.empty()) {
    if (_useAccelerationStructureBindings && _pTlasStructure &&
        _pGeometryHandleBuffer)
      waitForPendingTlasBuild();

    NS::UInteger tgWidth =
        std::max<NS::UInteger>(1, _pPathTracePSO->threadExecutionWidth());
    NS::UInteger maxThreads = std::max<NS::UInteger>(
        tgWidth, _pPathTracePSO->maxTotalThreadsPerThreadgroup());
    NS::UInteger tgHeight =
        std::max<NS::UInteger>(1, maxThreads / tgWidth);
    MTL::Size threadsPerThreadgroup =
        MTL::Size::Make(tgWidth, tgHeight, 1);

    size_t tileIndex = 0;
    const size_t maxWorkPerCommand =
        std::max<size_t>(kMaxTileSampleWorkPerCommand, 1);
    const size_t effectiveMaxSamples = std::max<size_t>(maxSamples, 1);

    while (tileIndex < tiles.size()) {
      size_t batchStart = tileIndex;
      size_t batchWork = 0;

      while (tileIndex < tiles.size()) {
        const TileDispatchRegion &tile = tiles[tileIndex];
        size_t tileWork = static_cast<size_t>(tile.width) * tile.height;
        tileWork = std::max<size_t>(tileWork, 1);
        tileWork *= effectiveMaxSamples;

        if (tileIndex > batchStart && batchWork + tileWork > maxWorkPerCommand)
          break;

        batchWork += tileWork;
        ++tileIndex;
      }

      size_t batchEnd = std::max<size_t>(batchStart + 1, tileIndex);

      MTL::CommandBuffer *computeCmd = _pCommandQueue->commandBuffer();
      if (!computeCmd)
        break;

      if (_useAccelerationStructureBindings && _pTlasStructure &&
          _pGeometryHandleBuffer && _pTlasBuildEvent &&
          _tlasBuildEventValue > 0)
        computeCmd->encodeWait(_pTlasBuildEvent, _tlasBuildEventValue);

      MTL::ComputeCommandEncoder *pCompute = computeCmd->computeCommandEncoder();
      if (!pCompute) {
        trackFrameCommandBuffer(computeCmd);
        computeCmd->commit();
        break;
      }

      pCompute->setComputePipelineState(_pPathTracePSO);

      bool useAccelerationStructureLayout =
          _useAccelerationStructureBindings && _pTlasStructure &&
          _pGeometryHandleBuffer;

      if (useAccelerationStructureLayout) {
        pCompute->setAccelerationStructure(_pTlasStructure, 0);
        pCompute->setBuffer(_pGeometryHandleBuffer, 0, 1);
        pCompute->setBuffer(_pSphereBuffer, 0, 2);
        pCompute->setBuffer(_pSphereMaterialBuffer, 0, 3);
        pCompute->setBuffer(_pUniformsBuffer, 0, 4);
        pCompute->setBuffer(_pActiveBuffer, 0, 5);
        pCompute->setBuffer(_pLightIndexBuffer, 0, 6);
        pCompute->setBuffer(_pLightCdfBuffer, 0, 7);
        pCompute->setBuffer(_pPrimitiveRemapBuffer, 0, 8);
        pCompute->setBuffer(_pPrimitiveHitBufferGPU, 0, 9);
        pCompute->setBuffer(_pInstanceBuffer, 0, 10);
        pCompute->setBuffer(_pPrimitiveHitBufferGPU, 0, 12);
        pCompute->setBuffer(_pInstanceBuffer, 0, 13);
      } else {
        pCompute->setBuffer(_pBVHBuffer, 0, 0);
        pCompute->setBuffer(_pSphereBuffer, 0, 1);
        pCompute->setBuffer(_pSphereMaterialBuffer, 0, 2);
        pCompute->setBuffer(_pUniformsBuffer, 0, 3);
        pCompute->setBuffer(_pTriangleVertexBuffer, 0, 4);
        pCompute->setBuffer(_pTriangleIndexBuffer, 0, 5);
        pCompute->setBuffer(_pPrimitiveIndexBuffer, 0, 6);
        pCompute->setBuffer(_pTLASBuffer, 0, 7);
        pCompute->setBuffer(_pActiveBuffer, 0, 8);
        pCompute->setBuffer(_pLightIndexBuffer, 0, 9);
        pCompute->setBuffer(_pLightCdfBuffer, 0, 10);
        pCompute->setBuffer(_pPrimitiveRemapBuffer, 0, 11);
        pCompute->setBuffer(_pPrimitiveHitBufferGPU, 0, 12);
        pCompute->setBuffer(_pInstanceBuffer, 0, 13);
      }

      pCompute->setTexture(accum0, 0);
      pCompute->setTexture(accum1, 1);
      pCompute->setTexture(sampleCount, 2);
      pCompute->setTexture(sampleImportance, 3);
      pCompute->setTexture(albedoTexture, 4);
      pCompute->setTexture(normalTexture, 5);

      for (uint32_t texIdx = 0; texIdx < kMaxMaterialTextureSlots; ++texIdx) {
        MTL::Texture *materialTex =
            (texIdx < _materialTextures.size()) ? _materialTextures[texIdx]
                                                : nullptr;
        pCompute->setTexture(materialTex, 6 + texIdx);
      }

      pCompute->setTexture(_environmentTexture,
                            6 + kMaxMaterialTextureSlots);
      if (_environmentSampler)
        pCompute->setSamplerState(_environmentSampler, 0);

      for (size_t batchIdx = batchStart; batchIdx < batchEnd; ++batchIdx) {
        const TileDispatchRegion &tile = tiles[batchIdx];
        TileDispatchRegion tileParams = tile;
        pCompute->setBytes(&tileParams, sizeof(TileDispatchRegion), 16);

        NS::UInteger localWidth = static_cast<NS::UInteger>(tile.width);
        NS::UInteger localHeight = static_cast<NS::UInteger>(tile.height);
        MTL::Size threadgroups = MTL::Size::Make(
            (localWidth + threadsPerThreadgroup.width - 1) /
                threadsPerThreadgroup.width,
            (localHeight + threadsPerThreadgroup.height - 1) /
                threadsPerThreadgroup.height,
            1);

        pCompute->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
      }

      pCompute->endEncoding();
      trackFrameCommandBuffer(computeCmd);
      computeCmd->commit();
    }
  }

  MTL::CommandBuffer *presentCmd = _pCommandQueue->commandBuffer();
  if (!presentCmd) {
    completeFrameMetrics(nullptr);
    ++_renderedFrameCount;
    pPool->release();
    return;
  }

  presentCmd->addCompletedHandler([this, captureList](MTL::CommandBuffer *cmd) {
    this->completeFrameMetrics(cmd);
    for (const auto &capture : *captureList)
      this->finalizeFrameCapture(capture);
  });

  MTL::RenderPassDescriptor *pRpd = pView->currentRenderPassDescriptor();
  MTL::RenderCommandEncoder *pEnc = presentCmd->renderCommandEncoder(pRpd);

  pEnc->setRenderPipelineState(_pPSO);
  pEnc->setFragmentTexture(accum1, 0);

  pEnc->drawPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle,
                       NS::UInteger(0), NS::UInteger(6));

  if (_pOverlayPSO && _observerActive) {
    float nearDistance = kFrustumDebugNear;
    float baseDistance = std::max(_primaryCameraState.focalLength, 1.0f);
    float farDistance =
        std::max(baseDistance * kFrustumDebugFarMultiplier, nearDistance * 2.0f);

    auto corners =
        buildFrustumCorners(_primaryCameraState, nearDistance, farDistance);
    std::array<simd::float3, kFrustumEdges.size() * 2> frustumVertices{};
    size_t vertexCount = 0;
    for (const auto &edge : kFrustumEdges) {
      size_t firstIndex = static_cast<size_t>(edge.first);
      size_t secondIndex = static_cast<size_t>(edge.second);
      if (firstIndex >= corners.size() || secondIndex >= corners.size())
        continue;
      frustumVertices[vertexCount++] = corners[firstIndex];
      frustumVertices[vertexCount++] = corners[secondIndex];
    }

    size_t requiredBytes = vertexCount * sizeof(simd::float3);
    ensureBufferCapacity(_pFrustumVertexBuffer, requiredBytes,
                         _frustumVertexCapacity, false,
                         MTL::ResourceStorageModeShared);

    if (_pFrustumVertexBuffer && requiredBytes > 0) {
      void *dst = _pFrustumVertexBuffer->contents();
      if (dst)
        std::memcpy(dst, frustumVertices.data(), requiredBytes);
      if (_pFrustumVertexBuffer->storageMode() == MTL::StorageModeManaged)
        markBufferModified(_pFrustumVertexBuffer,
                           NS::Range::Make(0, requiredBytes));

      float aspect = Camera::screenSize.y > 0.0f
                         ? Camera::screenSize.x / Camera::screenSize.y
                         : 1.0f;
      constexpr float kViewNear = 0.1f;
      constexpr float kViewFar = 1000.0f;
      OverlayUniforms uniforms{};
      uniforms.viewProjection =
          simd_mul(makePerspectiveMatrix(viewCamera.verticalFov, aspect,
                                         kViewNear, kViewFar),
                   makeViewMatrix(viewCamera));

      pEnc->setRenderPipelineState(_pOverlayPSO);
      pEnc->setVertexBuffer(_pFrustumVertexBuffer, 0, 0);
      pEnc->setVertexBytes(&uniforms, sizeof(uniforms), 1);
      pEnc->drawPrimitives(MTL::PrimitiveTypeLine, NS::UInteger(0),
                           NS::UInteger(vertexCount));

      pEnc->setRenderPipelineState(_pPSO);
      pEnc->setFragmentTexture(accum1, 0);
    }
  }

  pEnc->endEncoding();

  MTL::BlitCommandEncoder *pBlit = nullptr;
  bool performedRayHitReadback = false;

  if (captureThisFrame && accum1) {
    if (auto capture = encodeFrameCapture(accum1, albedoTexture, normalTexture,
                                          frameIndex, presentCmd, pBlit))
      captureList->push_back(capture);
  }

  if (_pPrimitiveHitBufferGPU && _pPrimitiveHitReadback) {
    if (!pBlit)
      pBlit = presentCmd->blitCommandEncoder();
    if (pBlit) {
      size_t bytes =
          std::min(_pPrimitiveHitBufferGPU->length(),
                   _pPrimitiveHitReadback->length());
      if (bytes > 0) {
        pBlit->copyFromBuffer(_pPrimitiveHitBufferGPU, 0,
                              _pPrimitiveHitReadback, 0, bytes);
        pBlit->fillBuffer(_pPrimitiveHitBufferGPU,
                          NS::Range::Make(0, bytes), 0);
        performedRayHitReadback = true;
      }
    }
  }
  if (pBlit)
    pBlit->endEncoding();

  MTL::Drawable *drawable = pView->currentDrawable();
  if (drawable)
    presentCmd->presentDrawable(drawable);
  trackFrameCommandBuffer(presentCmd);
  presentCmd->commit();

  if (_historyStreamingActive) {
    std::printf("[TextureResidency] History streaming active: restored %zu slot%s%s.\n",
                _historyStreamingRestoreCount,
                (_historyStreamingRestoreCount == 1) ? "" : "s",
                _historyStreamingUsedProxy ? " (proxy data involved)" : "");
  }

  if (!_frameCaptureEnabled &&
      _captureOutputsPending.load(std::memory_order_acquire))
    processPendingCapturedFrames();

  if (performedRayHitReadback) {
    if (_lastRayHitCommandBuffer)
      _lastRayHitCommandBuffer->release();
    _lastRayHitCommandBuffer = presentCmd;
    _lastRayHitCommandBuffer->retain();
    _rayHitCopyError = false;
  }

  ++_renderedFrameCount;
  pPool->release();
}

void Renderer::updateResidency(bool forceAllToggles, bool forceFullRebuild) {
  if (!_pScene)
    return;

  _framePrimitiveActivations = 0;
  _framePrimitiveDeactivations = 0;
  _frameObjectActivations = 0;
  _frameObjectDeactivations = 0;
  _frameProbabilisticToggles = 0;
  ResidencyStrategy strategy = _pScene->getResidencyStrategy();
  if (strategy != _lastResidencyStrategy) {
    if (strategy == ResidencyStrategy::AlwaysResident)
      _alwaysResidentCache.markDirty();
    _lastResidencyStrategy = strategy;
  }
  _frameStrategy = strategy;

  for (uint32_t &cooldown : _primitiveCooldown)
    if (cooldown > 0)
      --cooldown;
  for (uint32_t &cooldown : _objectCooldown)
    if (cooldown > 0)
      --cooldown;
  if (_compactionCooldown > 0)
    --_compactionCooldown;

  bool changed = false;
  switch (strategy) {
  case ResidencyStrategy::EnergyImportance:
    changed = updateEnergyImportance(forceAllToggles);
    break;
  case ResidencyStrategy::UnifiedScore:
    changed = updateUnifiedResidency(forceAllToggles);
    break;
  case ResidencyStrategy::RayHitBudget:
    changed = updateRayHitBudget(forceAllToggles);
    break;
  case ResidencyStrategy::Probabilistic:
    changed = updateProbabilisticResidency(forceAllToggles);
    break;
  case ResidencyStrategy::ScreenSpaceFootprint:
    changed = updateScreenSpaceFootprint(forceAllToggles);
    break;
  case ResidencyStrategy::EnvironmentHit:
    changed = updateEnvironmentHitResidency(forceAllToggles);
    break;
  case ResidencyStrategy::AlwaysResident:
    changed = updateAlwaysResident(forceAllToggles);
    break;
  case ResidencyStrategy::DistanceLOD:
  default:
    changed = updateLODByDistance(forceAllToggles);
    break;
  }

  if (changed || forceFullRebuild)
    flushResidencyChanges(forceFullRebuild);
}

bool Renderer::updateAlwaysResident(bool forceAllToggles) {
  size_t primitiveCount = _activePrimitive.size();
  size_t objectCount = _allSceneObjects.size();
  bool hasRecentChanges = !_recentlyActivated.empty() || !_recentlyDeactivated.empty();

  bool changed = false;

  if (_alwaysResidentCache.needsUpdate(forceAllToggles, primitiveCount,
                                       objectCount, hasRecentChanges)) {
    for (size_t objectIndex = 0; objectIndex < _allSceneObjects.size();
         ++objectIndex) {
      size_t toggled = setObjectActive(objectIndex, true);
      if (toggled > 0)
        changed = true;
    }

    for (size_t primIndex = 0; primIndex < _activePrimitive.size(); ++primIndex) {
      if (setPrimitiveActive(primIndex, true))
        changed = true;
    }

    _alwaysResidentCache.markUpdated(primitiveCount, objectCount);
  }

  bool anyInactivePrimitive =
      std::any_of(_activePrimitive.begin(), _activePrimitive.end(),
                  [](bool active) { return !active; });
  bool anyInactiveObject = false;
  for (size_t objectIndex = 0; objectIndex < _allSceneObjects.size(); ++objectIndex) {
    const SceneObject &obj = _allSceneObjects[objectIndex];
    if (obj.primitiveCount == 0)
      continue;
    bool active =
        (objectIndex < _objectActive.size()) ? _objectActive[objectIndex] : false;
    if (!active) {
      anyInactiveObject = true;
      break;
    }
  }

  if (anyInactivePrimitive || anyInactiveObject || !_recentlyDeactivated.empty()) {
    printf("Always-resident strategy detected attempted eviction (%zu primitives pending).\n",
           _recentlyDeactivated.size());
  }

  assert(!anyInactivePrimitive &&
         "Always-resident strategy should keep all primitives active");
  assert(!anyInactiveObject &&
         "Always-resident strategy should keep populated objects active");
  assert(_recentlyDeactivated.empty() &&
         "Always-resident strategy should not queue primitive deactivations");

  return changed;
}

bool Renderer::updateLODByDistance(bool forceAllToggles) {
  // Use hysteresis so objects do not flicker when hovering near the activation
  // boundary. Inactive objects only become active once the camera is closer
  // than LOD_ENTER_DISTANCE, while active objects stay active until the camera
  // has moved beyond LOD_EXIT_DISTANCE. Objects fully behind the camera are
  // treated as infinitely far away and immediately culled regardless of
  // distance thresholds. Additionally, objects must satisfy angular frustum
  // checks with separate entry/exit margins so camera rotations do not
  // immediately thrash residency state.

  size_t toggles = 0;
  bool changed = false;

  const size_t objectCount = _allSceneObjects.size();
  std::vector<float> objectDistances(objectCount,
                                     std::numeric_limits<float>::max());
  std::vector<bool> objectBehind(objectCount, false);
  std::vector<bool> objectViewEnter(objectCount, true);
  std::vector<bool> objectViewExit(objectCount, true);
  std::vector<size_t> sortedIndices(objectCount);
  simd::float3 forward = Camera::forward;
  float forwardLenSq = simd::length_squared(forward);
  bool forwardValid = forwardLenSq >= 1e-6f;
  if (forwardValid)
    forward /= std::sqrt(forwardLenSq);
  simd::float3 up = Camera::up;
  float upLenSq = simd::length_squared(up);
  if (upLenSq < 1e-6f) {
    up = {0.0f, 1.0f, 0.0f};
    upLenSq = 1.0f;
  }
  up /= std::sqrt(upLenSq);

  simd::float3 right = simd::make_float3(1.0f, 0.0f, 0.0f);
  if (forwardValid) {
    right = simd::cross(forward, up);
    float rightLenSq = simd::length_squared(right);
    if (rightLenSq < 1e-6f) {
      right = {1.0f, 0.0f, 0.0f};
    } else {
      right /= std::sqrt(rightLenSq);
    }
  }

  if (forwardValid)
    up = simd::normalize(simd::cross(right, forward));

  float verticalHalfFov =
      Camera::verticalFov * static_cast<float>(M_PI) / 180.0f * 0.5f;
  if (verticalHalfFov <= 0.0f)
    verticalHalfFov = 1e-3f;
  float aspect = Camera::screenSize.y > 0.0f
                     ? Camera::screenSize.x / Camera::screenSize.y
                     : 1.0f;
  float horizontalHalfFov = std::atan(std::tan(verticalHalfFov) * aspect);

  constexpr float kDegToRad = static_cast<float>(M_PI) / 180.0f;
  float enterMargin =
      std::max(_residencyConfig.lodEnterViewDegrees, 0.0f) * kDegToRad;
  float exitMargin =
      std::max(_residencyConfig.lodExitViewDegrees, 0.0f) * kDegToRad;
  exitMargin = std::max(exitMargin, enterMargin);
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    const BoundingSphere &sphere =
        (objectIndex < _objectBounds.size())
            ? _objectBounds[objectIndex]
            : BoundingSphere{simd::make_float3(0.0f, 0.0f, 0.0f), 0.0f};
    simd::float3 toCenter = sphere.center - Camera::position;
    float distanceToCenter = simd::length(toCenter);
    float dist = distanceToCenter - sphere.radius;
    float forwardDepth = forwardValid ? simd::dot(toCenter, forward) : 0.0f;
    bool behind = forwardValid && forwardDepth + sphere.radius <= 0.0f;
    objectBehind[objectIndex] = behind;
    objectDistances[objectIndex] =
        behind ? std::numeric_limits<float>::max() : std::max(dist, 0.0f);
    sortedIndices[objectIndex] = objectIndex;

    if (!forwardValid) {
      objectViewEnter[objectIndex] = true;
      objectViewExit[objectIndex] = true;
      continue;
    }

    if (distanceToCenter <= 1e-5f || behind) {
      objectViewEnter[objectIndex] = !behind;
      objectViewExit[objectIndex] = !behind;
      continue;
    }

    float depth = std::max(forwardDepth, 1e-5f);
    float horiz = simd::dot(toCenter, right);
    float vert = simd::dot(toCenter, up);
    float horizontalAngle = std::atan2(std::fabs(horiz), depth);
    float verticalAngle = std::atan2(std::fabs(vert), depth);
    float radiusAngle = asinf(std::min(sphere.radius / distanceToCenter, 1.0f));

    float enterHorizontalLimit = horizontalHalfFov + radiusAngle + enterMargin;
    float exitHorizontalLimit = horizontalHalfFov + radiusAngle + exitMargin;
    float enterVerticalLimit = verticalHalfFov + radiusAngle + enterMargin;
    float exitVerticalLimit = verticalHalfFov + radiusAngle + exitMargin;

    objectViewEnter[objectIndex] =
        horizontalAngle <= enterHorizontalLimit &&
        verticalAngle <= enterVerticalLimit;
    objectViewExit[objectIndex] =
        horizontalAngle <= exitHorizontalLimit &&
        verticalAngle <= exitVerticalLimit;
  }

  std::sort(sortedIndices.begin(), sortedIndices.end(),
            [&objectDistances](size_t a, size_t b) {
              return objectDistances[a] < objectDistances[b];
            });

  for (size_t orderIndex = 0; orderIndex < objectCount; ++orderIndex) {
    size_t objectIndex = sortedIndices[orderIndex];
    float dist = (objectIndex < objectDistances.size())
                     ? objectDistances[objectIndex]
                     : 0.0f;

    bool currentlyActive =
        objectIndex < _objectActive.size() && _objectActive[objectIndex];
    bool behind = objectIndex < objectBehind.size() && objectBehind[objectIndex];
    bool viewAllowed = currentlyActive
                           ? (objectIndex < objectViewExit.size() &&
                              objectViewExit[objectIndex])
                           : (objectIndex < objectViewEnter.size() &&
                              objectViewEnter[objectIndex]);
    bool shouldBeActive = viewAllowed && !behind &&
                          (currentlyActive
                               ? (dist <= _residencyConfig.lodExitDistance)
                               : (dist < _residencyConfig.lodEnterDistance));
    bool canToggle =
        forceAllToggles || objectIndex >= _objectCooldown.size() ||
        _objectCooldown[objectIndex] == 0;
    if (!canToggle || shouldBeActive == currentlyActive)
      continue;

    const SceneObject &obj = _allSceneObjects[objectIndex];
    size_t togglesNeeded = 0;
    size_t first = obj.firstPrimitive;
    size_t last = first + obj.primitiveCount;
    for (size_t prim = first; prim < last && prim < _activePrimitive.size();
         ++prim) {
      if (_activePrimitive[prim] != shouldBeActive)
        ++togglesNeeded;
    }

    if (togglesNeeded == 0)
      continue;

    size_t toggleCost = forceAllToggles ? togglesNeeded : size_t(1);
    if (!forceAllToggles &&
        toggles + toggleCost > _residencyConfig.lodMaxTogglesPerFrame)
      continue;

    size_t toggled = setObjectActive(objectIndex, shouldBeActive);
    if (toggled > 0) {
      toggles += toggleCost;
      changed = true;
    }
  }

  size_t activeCount = 0;
  for (bool active : _activePrimitive)
    if (active)
      ++activeCount;

  if (activeCount == 0 && !_activePrimitive.empty()) {
    // Ensure at least one primitive remains visible to avoid a blank scene
    if (setPrimitiveActive(0, true))
      changed = true;
  }

  return changed;
}

size_t Renderer::setObjectActive(size_t objectIndex, bool active) {
  if (objectIndex >= _allSceneObjects.size())
    return 0;

  if (objectIndex >= _objectActive.size())
    _objectActive.resize(objectIndex + 1, false);
  if (objectIndex >= _objectCooldown.size())
    _objectCooldown.resize(objectIndex + 1, 0);
  if (objectIndex >= _objectLastToggleFrame.size())
    _objectLastToggleFrame.resize(objectIndex + 1, 0);
  if (objectIndex >= _objectActivePrimitiveCounts.size())
    _objectActivePrimitiveCounts.resize(objectIndex + 1, 0);
  if (objectIndex >= _desiredObjectState.size())
    _desiredObjectState.resize(objectIndex + 1, 0);
  if (objectIndex >= _pendingDesiredObjects.size())
    _pendingDesiredObjects.resize(objectIndex + 1, 0);
  if (objectIndex >= _desiredObjectPromotionFrame.size())
    _desiredObjectPromotionFrame.resize(objectIndex + 1, 0);
  if (objectIndex >= _desiredObjectDemotionFrame.size())
    _desiredObjectDemotionFrame.resize(objectIndex + 1, 0);

  bool prevState = _objectActive[objectIndex];

  const SceneObject &obj = _allSceneObjects[objectIndex];
  size_t toggled = 0;
  size_t first = obj.firstPrimitive;
  size_t last = first + obj.primitiveCount;
  for (size_t prim = first; prim < last && prim < _activePrimitive.size(); ++prim)
    if (setPrimitiveActive(prim, active))
      ++toggled;

  size_t activeCount =
      objectIndex < _objectActivePrimitiveCounts.size()
          ? _objectActivePrimitiveCounts[objectIndex]
          : size_t(0);
  bool newState = activeCount > 0;
  bool fullyInactive = activeCount == 0;

  _objectActive[objectIndex] = newState;
  if (objectIndex < _desiredObjectState.size()) {
    _desiredObjectState[objectIndex] = newState ? 1 : 0;
    if (newState && objectIndex < _desiredObjectPromotionFrame.size())
      _desiredObjectPromotionFrame[objectIndex] = _renderedFrameCount;
    if (!newState && objectIndex < _desiredObjectDemotionFrame.size())
      _desiredObjectDemotionFrame[objectIndex] = _renderedFrameCount;
  }
  if ((!newState || toggled > 0) && objectIndex < _pendingDesiredObjects.size())
    _pendingDesiredObjects[objectIndex] = 0;

  if (prevState != newState) {
    _objectLastToggleFrame[objectIndex] = _renderedFrameCount;
    if (newState)
      ++_frameObjectActivations;
    else
      ++_frameObjectDeactivations;
  }

  if (toggled > 0 || prevState != newState || fullyInactive)
    _objectCooldown[objectIndex] = _residencyConfig.stateCooldownFrames;

  if (toggled > 0 || prevState != newState || fullyInactive)
    _dirtyResidentObjects.push_back(objectIndex);

  return toggled;
}

std::vector<float> Renderer::computeUnifiedImportance(float &outTotalScore) {
  const size_t primCount = _activePrimitive.size();
  std::vector<float> unifiedScores(primCount, 0.0f);
  outTotalScore = 0.0f;
  if (primCount == 0)
    return unifiedScores;

  if (_primitiveScreenCoverage.size() != primCount)
    _primitiveScreenCoverage.assign(primCount, 0.0f);
  if (_primitiveDistanceFalloffCache.size() != primCount)
    _primitiveDistanceFalloffCache.assign(primCount, 0.0f);
  if (_primitiveCoverageDirty.size() != primCount)
    _primitiveCoverageDirty.assign(primCount, 1);
  if (_primitiveCoverageBoundsVersion.size() != primCount)
    _primitiveCoverageBoundsVersion.assign(primCount, 0);
  if (_primitiveCoverageVisibilityKey.size() != primCount)
    _primitiveCoverageVisibilityKey.assign(primCount, 0xFF);

  float screenArea = Camera::screenSize.x * Camera::screenSize.y;
  if (screenArea <= 0.0f)
    screenArea = 1.0f;
  float halfFov = Camera::verticalFov * static_cast<float>(M_PI) / 180.0f * 0.5f;
  float tanHalfFov = std::tan(halfFov);
  if (tanHalfFov <= 0.0f)
    tanHalfFov = 1e-3f;

  simd::float3 forward = simd::normalize(Camera::forward);
  simd::float3 up = simd::normalize(Camera::up);
  simd::float3 right = simd::cross(forward, up);
  float rightLenSq = simd::length_squared(right);
  if (rightLenSq < 1e-6f)
    right = {1.0f, 0.0f, 0.0f};
  else
    right /= std::sqrt(rightLenSq);

  float aspect = Camera::screenSize.y > 0.0f
                     ? Camera::screenSize.x / Camera::screenSize.y
                     : 1.0f;
  float horizontalHalfFov = std::atan(tanHalfFov * aspect);

  bool cameraDirty = _coverageCameraVersion != _cameraVersion;
  if (cameraDirty) {
    _coverageCameraVersion = _cameraVersion;
    std::fill(_primitiveCoverageDirty.begin(), _primitiveCoverageDirty.end(), 1);

    // When the camera changes, decay cached hit information so the unified
    // score does not overly favor primitives that were visible from a previous
    // viewpoint.
    const float cameraHitDecay = 0.25f;
    for (float &hit : _primitiveHitScores)
      hit *= cameraHitDecay;
  }

  auto boundsVersionForPrimitive = [&](size_t primIndex) -> uint64_t {
    if (primIndex < _primitiveBoundsVersion.size())
      return _primitiveBoundsVersion[primIndex];
    size_t objectIndex = primIndex < _primitiveToObject.size()
                             ? _primitiveToObject[primIndex]
                             : std::numeric_limits<size_t>::max();
    if (objectIndex < _objectBoundsVersion.size())
      return _objectBoundsVersion[objectIndex];
    return 0;
  };

  auto coverageForSphere = [&](const BoundingSphere &b, bool &visible) -> float {
    visible = isInView(b);
    if (!visible)
      return 0.0f;
    simd::float3 toCenter = b.center - Camera::position;
    float depth = simd::dot(toCenter, forward);
    if (depth <= 1e-3f)
      return 0.0f;
    float dist = simd::length(toCenter);
    float cosAngle = depth / std::max(dist, 1e-3f);
    float horiz = simd::dot(toCenter, right);
    float horizAngle = std::atan2(std::fabs(horiz), depth);
    if (horizAngle > horizontalHalfFov + 0.1f)
      return 0.0f;
    float radiusPixels = (b.radius / depth) / tanHalfFov *
                         (Camera::screenSize.y * 0.5f);
    radiusPixels = std::max(radiusPixels, 0.0f);
    float area = static_cast<float>(M_PI) * radiusPixels * radiusPixels;
    float angleFactor = std::max(cosAngle, 0.0f);
    return std::min(area * angleFactor, screenArea);
  };

  auto distanceFalloff = [&](const BoundingSphere &sphere,
                             bool sphereVisible) -> float {
    if (!sphereVisible && !isInView(sphere))
      return 0.0f;

    simd::float3 toCenter = sphere.center - Camera::position;
    float depth = simd::dot(toCenter, forward);
    if (depth <= 1e-3f)
      return 0.0f;

    float dist = simd::length(toCenter);
    float cosAngle = depth / std::max(dist, 1e-3f);
    float horiz = simd::dot(toCenter, right);
    float horizAngle = std::atan2(std::fabs(horiz), depth);
    if (horizAngle > horizontalHalfFov + 0.1f)
      return 0.0f;

    float angleFactor = std::max(cosAngle, 0.0f);
    return angleFactor / (1.0f + depth);
  };

  parallelChunkedAsync(0, primCount,
                       [&, forward, right, horizontalHalfFov, tanHalfFov](
                           size_t chunkBegin, size_t chunkEnd) {
                         for (size_t i = chunkBegin; i < chunkEnd; ++i) {
                           uint64_t boundsVersion = boundsVersionForPrimitive(i);
                           if (i < _primitiveCoverageBoundsVersion.size() &&
                               _primitiveCoverageBoundsVersion[i] != boundsVersion)
                             _primitiveCoverageDirty[i] = 1;

                           uint8_t activeKey =
                               (i < _activePrimitive.size() && _activePrimitive[i])
                                   ? 0x2
                                   : 0x0;
                           bool dirty = (i < _primitiveCoverageDirty.size())
                                            ? (_primitiveCoverageDirty[i] != 0)
                                            : true;
                           if (i >= _primitiveCoverageVisibilityKey.size() ||
                               ((_primitiveCoverageVisibilityKey[i] & 0x2) !=
                                activeKey))
                             dirty = true;

                           if (!dirty)
                             continue;

                           BoundingSphere chosenSphere{};
                           bool haveSphere = false;
                           if (i < _primitiveBounds.size()) {
                             chosenSphere = _primitiveBounds[i];
                             haveSphere = true;
                           } else {
                             size_t objectIndex =
                                 (i < _primitiveToObject.size())
                                     ? _primitiveToObject[i]
                                     : std::numeric_limits<size_t>::max();
                             if (objectIndex < _objectBounds.size()) {
                               chosenSphere = _objectBounds[objectIndex];
                               haveSphere = true;
                             }
                           }

                           bool visible = false;
                           float coverage = 0.0f;
                           if (i < _primitiveBounds.size())
                             coverage = coverageForSphere(_primitiveBounds[i], visible);

                           float distanceScore =
                               haveSphere ? distanceFalloff(chosenSphere, visible)
                                          : 0.0f;

                           _primitiveScreenCoverage[i] = coverage;
                           _primitiveDistanceFalloffCache[i] = distanceScore;
                           if (i < _primitiveCoverageBoundsVersion.size())
                             _primitiveCoverageBoundsVersion[i] = boundsVersion;
                           if (i < _primitiveCoverageVisibilityKey.size())
                             _primitiveCoverageVisibilityKey[i] =
                                 (visible ? 1 : 0) | activeKey;
                           if (i < _primitiveCoverageDirty.size())
                             _primitiveCoverageDirty[i] = 0;
                         }
                       });

  if (_primitiveHitScoresSnapshot.size() < primCount)
    _primitiveHitScoresSnapshot.resize(primCount, 0.0f);
  size_t copyCount = std::min(_primitiveHitScores.size(), primCount);
  std::copy_n(_primitiveHitScores.begin(), copyCount,
              _primitiveHitScoresSnapshot.begin());
  if (copyCount < primCount) {
    std::fill(_primitiveHitScoresSnapshot.begin() + copyCount,
              _primitiveHitScoresSnapshot.begin() + primCount, 0.0f);
  }

  const float alpha = _residencyConfig.unifiedEnergyWeight;
  const float beta = _residencyConfig.unifiedHitWeight;
  const float gamma = _residencyConfig.unifiedCoverageWeight;
  const float delta = _residencyConfig.unifiedDistanceWeight;
  const float offscreenDecay = 0.1f;

  std::vector<size_t> candidateIndices;
  candidateIndices.reserve(primCount);

  for (size_t i = 0; i < primCount; ++i) {
    float hit = (i < _primitiveHitScoresSnapshot.size())
                    ? _primitiveHitScoresSnapshot[i]
                    : 0.0f;
    float coverage = (i < _primitiveScreenCoverage.size())
                         ? _primitiveScreenCoverage[i]
                         : 0.0f;
    float distanceScore = (i < _primitiveDistanceFalloffCache.size())
                              ? _primitiveDistanceFalloffCache[i]
                              : 0.0f;
    bool visible =
        (i < _primitiveCoverageVisibilityKey.size())
            ? ((_primitiveCoverageVisibilityKey[i] & 0x1) != 0)
            : false;

    bool offscreenNoFalloff = !visible && coverage == 0.0f && distanceScore == 0.0f;
    if (offscreenNoFalloff)
      hit *= offscreenDecay;

    if (hit == 0.0f && !visible && coverage == 0.0f && distanceScore == 0.0f)
      continue;

    candidateIndices.push_back(i);
  }

  for (size_t i : candidateIndices) {
    float energy = (i < _primitiveImportance.size()) ? _primitiveImportance[i]
                                                     : 0.0f;
    float hit = (i < _primitiveHitScoresSnapshot.size())
                    ? _primitiveHitScoresSnapshot[i]
                    : 0.0f;
    float coverage = (i < _primitiveScreenCoverage.size())
                         ? _primitiveScreenCoverage[i]
                         : 0.0f;
    float distanceScore = (i < _primitiveDistanceFalloffCache.size())
                              ? _primitiveDistanceFalloffCache[i]
                              : 0.0f;
    bool visible =
        (i < _primitiveCoverageVisibilityKey.size())
            ? ((_primitiveCoverageVisibilityKey[i] & 0x1) != 0)
            : false;

    if (hit == 0.0f && coverage == 0.0f && distanceScore == 0.0f &&
        (i >= _primitiveCoverageVisibilityKey.size() ||
         (_primitiveCoverageVisibilityKey[i] & 0x1) == 0))
      continue;

    bool offscreenNoFalloff = !visible && coverage == 0.0f && distanceScore == 0.0f;
    if (offscreenNoFalloff) {
      hit *= offscreenDecay;
      energy *= offscreenDecay;
    }

    float score = alpha * energy + beta * hit + gamma * coverage +
                  delta * distanceScore;
    unifiedScores[i] = score;
    outTotalScore += std::max(score, 0.0f);
  }

  return unifiedScores;
}

bool Renderer::updateEnergyImportance(bool forceAllToggles) {
  if (_activePrimitive.empty())
    return false;

  float recomputedTotalImportance = 0.0f;
  for (float importance : _primitiveImportance)
    recomputedTotalImportance += std::max(importance, 0.0f);
  _totalPrimitiveImportance = recomputedTotalImportance;

  const size_t objectCount = _allSceneObjects.size();
  if (objectCount == 0) {
    // Fall back to primitive-level logic if no objects are available.
    const size_t primCount = _activePrimitive.size();
    std::vector<int8_t> desiredState(primCount, -1);
    std::vector<size_t> sortedIndices(primCount);
    std::iota(sortedIndices.begin(), sortedIndices.end(), size_t(0));
    size_t minActive =
        std::min(primCount, _residencyConfig.energyMinActivePrimitives);
    size_t targetActive = static_cast<size_t>(std::ceil(
        static_cast<float>(primCount) * _residencyConfig.energyTargetFraction));
    size_t sortCount =
        std::min(primCount, std::max(minActive, targetActive));
    if (sortCount > 0) {
      auto comparator = [this](size_t a, size_t b) {
        float scoreA = sanitizeSortValue(_primitiveImportance[a]);
        float scoreB = sanitizeSortValue(_primitiveImportance[b]);
        if (scoreA == scoreB)
          return a < b;
        return scoreA > scoreB;
      };
      std::partial_sort(sortedIndices.begin(),
                        sortedIndices.begin() + sortCount,
                        sortedIndices.end(), comparator);
    }

    if (_totalPrimitiveImportance <= 0.0f) {
      size_t enabledPrimitives = 0;
      for (size_t i = 0; i < sortCount; ++i) {
        size_t index = sortedIndices[i];
        desiredState[index] = 1;
        ++enabledPrimitives;
        if (enabledPrimitives >= minActive)
          break;
      }
    } else {
      float cumulative = 0.0f;
      float targetImportance =
          _totalPrimitiveImportance * _residencyConfig.energyTargetFraction;
      size_t enabled = 0;
      for (size_t i = 0; i < sortCount; ++i) {
        size_t index = sortedIndices[i];
        desiredState[index] = 1;
        cumulative += std::max(_primitiveImportance[index], 0.0f);
        ++enabled;
        if (enabled >= minActive && cumulative >= targetImportance)
          break;
      }

      for (size_t i = 0; i < minActive && i < sortCount; ++i)
        desiredState[sortedIndices[i]] = 1;
    }

    bool changed = false;
    size_t toggles = 0;
    for (size_t i = 0; i < sortCount; ++i) {
      size_t primIndex = sortedIndices[i];
      int8_t desired = desiredState[primIndex];
      if (desired < 0)
        continue;
      bool shouldBeActive = desired != 0;
      if (shouldBeActive == _activePrimitive[primIndex])
        continue;
      if (!forceAllToggles) {
        if (primIndex < _primitiveCooldown.size() &&
            _primitiveCooldown[primIndex] > 0)
          continue;
        if (toggles >= _residencyConfig.energyMaxTogglesPerFrame)
          break;
      }
      if (setPrimitiveActive(primIndex, shouldBeActive)) {
        changed = true;
        ++toggles;
      }
    }

    size_t activeCount = 0;
    for (bool active : _activePrimitive)
      if (active)
        ++activeCount;

    if (activeCount == 0 && !_activePrimitive.empty()) {
      size_t fallback = !sortedIndices.empty() ? sortedIndices.front() : size_t(0);
      if (setPrimitiveActive(fallback, true))
        changed = true;
    }

    return changed;
  }

  if (_objectImportance.size() != objectCount)
    _objectImportance.assign(objectCount, 0.0f);
  else
    std::fill(_objectImportance.begin(), _objectImportance.end(), 0.0f);

  bool resetImportanceHistory = false;
  if (_objectImportanceHistory.size() != objectCount) {
    _objectImportanceHistory.assign(objectCount, 0.0f);
    resetImportanceHistory = true;
  }

  if (_energySortedIndices.size() != objectCount)
    _energySortedIndices.resize(objectCount);
  std::iota(_energySortedIndices.begin(), _energySortedIndices.end(),
            size_t(0));

  float screenArea = Camera::screenSize.x * Camera::screenSize.y;
  if (screenArea <= 0.0f)
    screenArea = 1.0f;

  float visibilityBoost = std::max(_residencyConfig.energyVisibilityBoost, 1.0f);
  const bool applyVisibilityBoost = visibilityBoost > 1.0001f;

  simd::float3 forward = simd::normalize(Camera::forward);
  simd::float3 up = simd::normalize(Camera::up);
  simd::float3 right = simd::cross(forward, up);
  float rightLenSq = simd::length_squared(right);
  if (rightLenSq < 1e-6f)
    right = {1.0f, 0.0f, 0.0f};
  else
    right /= std::sqrt(rightLenSq);

  float halfFov = Camera::verticalFov * static_cast<float>(M_PI) / 180.0f * 0.5f;
  float tanHalfFov = std::tan(halfFov);
  if (tanHalfFov <= 0.0f)
    tanHalfFov = 1e-3f;

  float aspect = Camera::screenSize.y > 0.0f
                     ? Camera::screenSize.x / Camera::screenSize.y
                     : 1.0f;
  float horizontalHalfFov = std::atan(tanHalfFov * aspect);

  auto coverageForSphere = [&](const BoundingSphere &b) -> float {
    if (!applyVisibilityBoost)
      return 0.0f;
    if (!isInView(b))
      return 0.0f;
    simd::float3 toCenter = b.center - Camera::position;
    float depth = simd::dot(toCenter, forward);
    if (depth <= 1e-3f)
      return 0.0f;
    float dist = simd::length(toCenter);
    float cosAngle = depth / std::max(dist, 1e-3f);
    float horiz = simd::dot(toCenter, right);
    float horizAngle = std::atan2(std::fabs(horiz), depth);
    if (horizAngle > horizontalHalfFov + 0.1f)
      return 0.0f;
    float radiusPixels = (b.radius / depth) / tanHalfFov *
                         (Camera::screenSize.y * 0.5f);
    radiusPixels = std::max(radiusPixels, 0.0f);
    float area = static_cast<float>(M_PI) * radiusPixels * radiusPixels;
    float angleFactor = std::max(cosAngle, 0.0f);
    return std::min(area * angleFactor, screenArea);
  };

  const auto &objectPrimitiveCounts = _objectPrimitiveCounts;
  bool anyMeshGroups = _anyMeshGroups;
  float boostedTotalImportance = 0.0f;
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    const SceneObject &obj = _allSceneObjects[objectIndex];
    size_t first = obj.firstPrimitive;
    size_t last = first + obj.primitiveCount;
    float totalImportance = 0.0f;
    float coverage = 0.0f;
    for (size_t prim = first; prim < last && prim < _primitiveImportance.size();
         ++prim) {
      totalImportance += std::max(_primitiveImportance[prim], 0.0f);
      if (applyVisibilityBoost && prim < _primitiveScreenCoverage.size())
        coverage += std::max(_primitiveScreenCoverage[prim], 0.0f);
    }
    if (applyVisibilityBoost) {
      float sphereCoverage = 0.0f;
      if (objectIndex < _objectBounds.size())
        sphereCoverage = coverageForSphere(_objectBounds[objectIndex]);
      float combinedCoverage = std::max(coverage, sphereCoverage);
      float coverageFraction =
          std::clamp(combinedCoverage / screenArea, 0.0f, 1.0f);
      float multiplier =
          1.0f + (visibilityBoost - 1.0f) * coverageFraction;
      _objectImportance[objectIndex] = totalImportance * multiplier;
    } else {
      _objectImportance[objectIndex] = totalImportance;
    }

    if (applyVisibilityBoost)
      boostedTotalImportance += std::max(_objectImportance[objectIndex], 0.0f);
  }

  std::vector<float> energyImportance(objectCount, 0.0f);
  float historyWeight = std::clamp(_residencyConfig.energyImportanceSmoothing,
                                   0.0f, 0.999f);
  float currentWeight = 1.0f - historyWeight;
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    float current = _objectImportance[objectIndex];
    float smoothed = current;
    if (historyWeight > 0.0f) {
      float previous = resetImportanceHistory
                           ? current
                           : _objectImportanceHistory[objectIndex];
      smoothed = previous * historyWeight + current * currentWeight;
    }
    _objectImportanceHistory[objectIndex] = smoothed;
    energyImportance[objectIndex] = smoothed;
  }

  if (applyVisibilityBoost) {
    boostedTotalImportance = 0.0f;
    for (float importance : energyImportance)
      boostedTotalImportance += std::max(importance, 0.0f);
  }

  float targetImportanceBase = applyVisibilityBoost ? boostedTotalImportance
                                                    : _totalPrimitiveImportance;

  std::vector<float> energyImportancePerPrimitive(objectCount, 0.0f);
  float totalEnergyImportancePerPrimitive = 0.0f;
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    size_t count = objectIndex < objectPrimitiveCounts.size()
                       ? objectPrimitiveCounts[objectIndex]
                       : 0;
    if (count == 0)
      continue;
    float avgImportance = energyImportance[objectIndex] /
                          static_cast<float>(count);
    energyImportancePerPrimitive[objectIndex] = avgImportance;
    totalEnergyImportancePerPrimitive += std::max(avgImportance, 0.0f);
  }

  std::sort(_energySortedIndices.begin(), _energySortedIndices.end(),
            [&](size_t a, size_t b) {
              auto scoreFor = [&](size_t idx) {
                float score = 0.0f;
                if (idx < _objectImportanceHistory.size())
                  score = sanitizeSortValue(_objectImportanceHistory[idx]);
                if (!anyMeshGroups) {
                  size_t count = idx < objectPrimitiveCounts.size()
                                     ? objectPrimitiveCounts[idx]
                                     : 0;
                  if (count == 0)
                    return 0.0f;
                  score /= static_cast<float>(count);
                }
                return score;
              };
              float scoreA = scoreFor(a);
              float scoreB = scoreFor(b);
              if (scoreA == scoreB)
                return a < b;
              return scoreA > scoreB;
            });

  const size_t primCount = _activePrimitive.size();
  const size_t minActivePrimitives =
      std::min(primCount, _residencyConfig.energyMinActivePrimitives);

  if (_objectLastToggleFrame.size() < objectCount)
    _objectLastToggleFrame.resize(objectCount, 0);

  auto withinObjectStateCooldown = [&](size_t objectIndex) {
    if (forceAllToggles)
      return false;
    uint64_t minDuration =
        static_cast<uint64_t>(_residencyConfig.stateCooldownFrames);
    if (minDuration == 0)
      return false;
    if (objectIndex >= _objectLastToggleFrame.size())
      return false;
    uint64_t lastFrame = _objectLastToggleFrame[objectIndex];
    if (lastFrame == 0)
      return false;
    return (_renderedFrameCount >= lastFrame) &&
           (_renderedFrameCount - lastFrame < minDuration);
  };

  if (!anyMeshGroups) {
    std::vector<bool> desiredObjectState(objectCount, false);
    size_t primitivesEnabled = 0;
    float targetImportanceBaseSelection =
        totalEnergyImportancePerPrimitive > 0.0f
            ? totalEnergyImportancePerPrimitive
            : targetImportanceBase;

    if (_totalPrimitiveImportance <= 0.0f) {
      for (size_t idx : _energySortedIndices) {
        if (idx >= objectPrimitiveCounts.size())
          continue;
        size_t count = objectPrimitiveCounts[idx];
        if (count == 0)
          continue;
        desiredObjectState[idx] = true;
        primitivesEnabled += count;
        if (primitivesEnabled >= minActivePrimitives)
          break;
      }
    } else {
      float cumulativeImportance = 0.0f;
      float targetImportance =
          targetImportanceBaseSelection *
          _residencyConfig.energyTargetFraction;
      for (size_t idx : _energySortedIndices) {
        if (idx >= objectPrimitiveCounts.size())
          continue;
        size_t count = objectPrimitiveCounts[idx];
        float importance = (idx < energyImportancePerPrimitive.size())
                               ? energyImportancePerPrimitive[idx]
                               : 0.0f;
        if (count == 0 && importance <= 0.0f)
          continue;
        desiredObjectState[idx] = true;
        primitivesEnabled += count;
        cumulativeImportance += std::max(importance, 0.0f);
        if (primitivesEnabled >= minActivePrimitives &&
            cumulativeImportance >= targetImportance)
          break;
      }
    }

    if (primitivesEnabled < minActivePrimitives) {
      for (size_t idx : _energySortedIndices) {
        if (idx >= objectPrimitiveCounts.size())
          continue;
        if (desiredObjectState[idx])
          continue;
        size_t count = objectPrimitiveCounts[idx];
        if (count == 0)
          continue;
        desiredObjectState[idx] = true;
        primitivesEnabled += count;
        if (primitivesEnabled >= minActivePrimitives)
          break;
      }
    }

    bool changed = false;
    size_t toggledPrimitiveCount = 0;
    auto attemptToggleObject = [&](size_t objectIndex,
                                   bool shouldBeActive) -> size_t {
      if (objectIndex >= _allSceneObjects.size())
        return 0;
      bool currentlyActive =
          objectIndex < _objectActive.size() && _objectActive[objectIndex];
      if (shouldBeActive == currentlyActive)
        return 0;

      if (!forceAllToggles) {
        if (objectIndex < _objectCooldown.size() &&
            _objectCooldown[objectIndex] > 0)
          return 0;
        if (withinObjectStateCooldown(objectIndex))
          return 0;
      }

      const SceneObject &obj = _allSceneObjects[objectIndex];
      size_t first = obj.firstPrimitive;
      size_t last = first + obj.primitiveCount;
      size_t togglesNeeded = 0;
      bool canToggle = true;
      for (size_t prim = first; prim < last && prim < _activePrimitive.size();
           ++prim) {
        if (_activePrimitive[prim] == shouldBeActive)
          continue;
        if (!forceAllToggles && prim < _primitiveCooldown.size() &&
            _primitiveCooldown[prim] > 0) {
          canToggle = false;
          break;
        }
        ++togglesNeeded;
      }

      if (!canToggle || togglesNeeded == 0)
        return 0;

      if (!forceAllToggles &&
          toggledPrimitiveCount >= _residencyConfig.energyMaxTogglesPerFrame)
        return 0;

      size_t toggled = setObjectActive(objectIndex, shouldBeActive);
      if (toggled > 0 && !forceAllToggles) {
        toggledPrimitiveCount =
            std::min(toggledPrimitiveCount + toggled,
                     _residencyConfig.energyMaxTogglesPerFrame);
      }
      return toggled;
    };

    for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
      bool shouldBeActive = desiredObjectState[objectIndex];
      size_t toggled = attemptToggleObject(objectIndex, shouldBeActive);
      if (toggled > 0) {
        changed = true;
      }
    }

    bool anyActiveObject = false;
    for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
      if (objectIndex >= _objectActive.size() ||
          objectIndex >= objectPrimitiveCounts.size())
        continue;
      if (objectPrimitiveCounts[objectIndex] == 0)
        continue;
      if (_objectActive[objectIndex]) {
        anyActiveObject = true;
        break;
      }
    }

    if (!anyActiveObject) {
      for (size_t idx : _energySortedIndices) {
        if (idx >= objectPrimitiveCounts.size())
          continue;
        if (objectPrimitiveCounts[idx] == 0)
          continue;
        if (attemptToggleObject(idx, true) > 0) {
          changed = true;
          anyActiveObject = true;
        }
        if (anyActiveObject)
          break;
      }
    }

    if (!anyActiveObject && !_activePrimitive.empty()) {
      size_t fallbackObject = std::numeric_limits<size_t>::max();
      for (size_t idx : _energySortedIndices) {
        if (idx >= objectPrimitiveCounts.size())
          continue;
        if (objectPrimitiveCounts[idx] == 0)
          continue;
        fallbackObject = idx;
        break;
      }

      if (fallbackObject < _allSceneObjects.size()) {
        if (attemptToggleObject(fallbackObject, true) > 0)
          changed = true;
      } else {
        if (setPrimitiveActive(0, true))
          changed = true;
      }
    }

    return changed;
  }

  struct MeshGroupAggregate {
    const MeshGroupInfo *info = nullptr;
    float importance = 0.0f;
    size_t primitiveCount = 0;
  };

  std::vector<MeshGroupAggregate> meshGroups;
  meshGroups.reserve(_meshGroups.size());
  for (const auto &info : _meshGroups) {
    MeshGroupAggregate aggregate;
    aggregate.info = &info;
    size_t fallbackPrimitiveCount = 0;
    for (size_t objectIndex : info.objectIndices) {
      if (objectIndex < energyImportance.size())
        aggregate.importance += energyImportance[objectIndex];
      if (objectIndex < objectPrimitiveCounts.size())
        fallbackPrimitiveCount += objectPrimitiveCounts[objectIndex];
    }
    if (info.primitiveCount > 0)
      aggregate.primitiveCount = info.primitiveCount;
    else
      aggregate.primitiveCount = fallbackPrimitiveCount;
    meshGroups.push_back(std::move(aggregate));
  }

  std::vector<float> meshGroupAverageImportance(meshGroups.size(), 0.0f);
  for (size_t groupIndex = 0; groupIndex < meshGroups.size(); ++groupIndex) {
    const MeshGroupAggregate &group = meshGroups[groupIndex];
    if (group.primitiveCount > 0) {
      meshGroupAverageImportance[groupIndex] =
          group.importance /
          static_cast<float>(group.primitiveCount);
    }
  }

  std::vector<size_t> meshSortedIndices(meshGroups.size());
  std::iota(meshSortedIndices.begin(), meshSortedIndices.end(), size_t(0));
  std::sort(meshSortedIndices.begin(), meshSortedIndices.end(),
            [&meshGroups](size_t a, size_t b) {
              float scoreA = (a < meshGroups.size())
                                 ? sanitizeSortValue(meshGroups[a].importance)
                                 : 0.0f;
              float scoreB = (b < meshGroups.size())
                                 ? sanitizeSortValue(meshGroups[b].importance)
                                 : 0.0f;
              if (scoreA == scoreB)
                return a < b;
              return scoreA > scoreB;
            });

  std::vector<bool> desiredGroupState(meshGroups.size(), false);
  size_t primitivesEnabled = 0;

  bool meshTargetSatisfied = false;
  float meshLastPrimaryAverage = 0.0f;
  float meshPrevPrimaryAverage = std::numeric_limits<float>::quiet_NaN();
  bool meshHasPrimaryAverage = false;
  size_t meshLastPrimarySortedPos = std::numeric_limits<size_t>::max();

  if (_totalPrimitiveImportance <= 0.0f) {
    for (size_t idx : meshSortedIndices) {
      if (idx >= meshGroups.size())
        continue;
      const MeshGroupAggregate &group = meshGroups[idx];
      if (group.primitiveCount == 0)
        continue;
      desiredGroupState[idx] = true;
      primitivesEnabled += group.primitiveCount;
      if (primitivesEnabled >= minActivePrimitives)
        break;
    }
  } else {
    float cumulativeImportance = 0.0f;
    float targetImportance =
        targetImportanceBase * _residencyConfig.energyTargetFraction;
    for (size_t sortedPos = 0; sortedPos < meshSortedIndices.size();
         ++sortedPos) {
      size_t idx = meshSortedIndices[sortedPos];
      if (idx >= meshGroups.size())
        continue;
      const MeshGroupAggregate &group = meshGroups[idx];
      if (group.primitiveCount == 0 && group.importance <= 0.0f)
        continue;
      desiredGroupState[idx] = true;
      primitivesEnabled += group.primitiveCount;
      cumulativeImportance += std::max(group.importance, 0.0f);
      if (meshHasPrimaryAverage)
        meshPrevPrimaryAverage = meshLastPrimaryAverage;
      meshLastPrimaryAverage =
          (idx < meshGroupAverageImportance.size())
              ? meshGroupAverageImportance[idx]
              : 0.0f;
      meshHasPrimaryAverage = true;
      meshLastPrimarySortedPos = sortedPos;
      if (primitivesEnabled >= minActivePrimitives &&
          cumulativeImportance >= targetImportance) {
        meshTargetSatisfied = true;
        break;
      }
    }
  }

  if (primitivesEnabled < minActivePrimitives) {
    for (size_t idx : meshSortedIndices) {
      if (idx >= meshGroups.size())
        continue;
      const MeshGroupAggregate &group = meshGroups[idx];
      if (group.primitiveCount == 0)
        continue;
      if (desiredGroupState[idx])
        continue;
      desiredGroupState[idx] = true;
      primitivesEnabled += group.primitiveCount;
      if (primitivesEnabled >= minActivePrimitives)
        break;
    }
  }

  if (meshTargetSatisfied && meshHasPrimaryAverage &&
      meshLastPrimaryAverage > 0.0f) {
    float prevDiff = std::numeric_limits<float>::infinity();
    if (!std::isnan(meshPrevPrimaryAverage))
      prevDiff =
          std::fabs(meshLastPrimaryAverage - meshPrevPrimaryAverage);

    float nextDiff = std::numeric_limits<float>::infinity();
    size_t nextSortedPos = meshLastPrimarySortedPos + 1;
    if (nextSortedPos < meshSortedIndices.size()) {
      size_t nextIdx = meshSortedIndices[nextSortedPos];
      if (nextIdx < meshGroupAverageImportance.size()) {
        float nextAverage = meshGroupAverageImportance[nextIdx];
        if (std::isfinite(nextAverage))
          nextDiff = std::fabs(meshLastPrimaryAverage - nextAverage);
      }
    }

    float minNeighborDiff = std::min(prevDiff, nextDiff);
    float allowedRatio = 0.0f;
    if (std::isfinite(minNeighborDiff) && minNeighborDiff > 0.0f)
      allowedRatio = minNeighborDiff /
                     std::max(meshLastPrimaryAverage, std::numeric_limits<float>::min());

    const float kRelativeFallback = 0.01f;
    allowedRatio = std::max(allowedRatio, kRelativeFallback);

    float allowedDelta = meshLastPrimaryAverage * allowedRatio;
    const float kMaxAllowedFraction = 0.25f;
    if (std::isfinite(allowedDelta) && meshLastPrimaryAverage > 0.0f) {
      float maxAllowedDelta = meshLastPrimaryAverage * kMaxAllowedFraction;
      if (std::isfinite(maxAllowedDelta))
        allowedDelta = std::min(allowedDelta, maxAllowedDelta);
    }

    if (allowedDelta > 0.0f) {
      float epsilon = std::max(1e-5f, meshLastPrimaryAverage * 1e-3f);
      for (size_t groupIndex = 0; groupIndex < meshGroups.size();
           ++groupIndex) {
        if (desiredGroupState[groupIndex])
          continue;
        const MeshGroupAggregate &group = meshGroups[groupIndex];
        if (group.primitiveCount == 0)
          continue;
        float average = meshGroupAverageImportance[groupIndex];
        if (average <= 0.0f)
          continue;
        float difference = meshLastPrimaryAverage - average;
        if (difference <= allowedDelta + epsilon)
          desiredGroupState[groupIndex] = true;
      }
    }
  }

  std::vector<bool> desiredObjectState(objectCount, false);
  for (size_t groupIndex = 0; groupIndex < meshGroups.size(); ++groupIndex) {
    if (!desiredGroupState[groupIndex])
      continue;
    const MeshGroupAggregate &group = meshGroups[groupIndex];
    const auto *objectIndices =
        group.info ? &group.info->objectIndices : nullptr;
    if (!objectIndices)
      continue;
    for (size_t objectIndex : *objectIndices) {
      if (objectIndex < desiredObjectState.size())
        desiredObjectState[objectIndex] = true;
    }
  }

  bool changed = false;
  size_t toggledPrimitiveCount = 0;
  auto attemptToggleObject = [&](size_t objectIndex, bool shouldBeActive) {
    if (objectIndex >= _allSceneObjects.size())
      return size_t(0);
    bool currentlyActive =
        (objectIndex < _objectActive.size()) ? _objectActive[objectIndex] : false;
    if (currentlyActive == shouldBeActive)
      return size_t(0);

    if (!forceAllToggles) {
      if (objectIndex < _objectCooldown.size() &&
          _objectCooldown[objectIndex] > 0)
        return size_t(0);
      if (withinObjectStateCooldown(objectIndex))
        return size_t(0);
    }

    const SceneObject &obj = _allSceneObjects[objectIndex];
    size_t first = obj.firstPrimitive;
    size_t last = first + obj.primitiveCount;
    size_t togglesNeeded = 0;
    bool canToggle = true;
    for (size_t prim = first; prim < last && prim < _activePrimitive.size();
         ++prim) {
      if (_activePrimitive[prim] == shouldBeActive)
        continue;
      if (!forceAllToggles && prim < _primitiveCooldown.size() &&
          _primitiveCooldown[prim] > 0) {
        canToggle = false;
        break;
      }
      ++togglesNeeded;
    }

    if (!canToggle || togglesNeeded == 0)
      return size_t(0);
    if (!forceAllToggles &&
        toggledPrimitiveCount >= _residencyConfig.energyMaxTogglesPerFrame)
      return size_t(0);

    size_t toggled = setObjectActive(objectIndex, shouldBeActive);
    if (toggled > 0 && !forceAllToggles) {
      toggledPrimitiveCount =
          std::min(toggledPrimitiveCount + toggled,
                   _residencyConfig.energyMaxTogglesPerFrame);
    }
    return toggled;
  };

  for (size_t groupIndex = 0; groupIndex < meshGroups.size(); ++groupIndex) {
    const MeshGroupAggregate &group = meshGroups[groupIndex];
    bool groupShouldBeActive = desiredGroupState[groupIndex];
    size_t groupToggleCount = 0;
    bool groupNeedsToggle = false;
    bool groupCanToggle = true;
    const auto *objectIndices = group.info ? &group.info->objectIndices : nullptr;

    if (!objectIndices)
      continue;

    for (size_t objectIndex : *objectIndices) {
      bool shouldBeActive =
          (objectIndex < desiredObjectState.size())
              ? desiredObjectState[objectIndex]
              : groupShouldBeActive;
      bool currentlyActive =
          (objectIndex < _objectActive.size()) ? _objectActive[objectIndex]
                                               : false;
      if (shouldBeActive == currentlyActive)
        continue;
      if (!forceAllToggles) {
        if (objectIndex < _objectCooldown.size() &&
            _objectCooldown[objectIndex] > 0) {
          groupCanToggle = false;
          break;
        }
        if (withinObjectStateCooldown(objectIndex)) {
          groupCanToggle = false;
          break;
        }
      }
      groupNeedsToggle = true;

      const SceneObject &obj = _allSceneObjects[objectIndex];
      size_t first = obj.firstPrimitive;
      size_t last = first + obj.primitiveCount;
      for (size_t prim = first; prim < last && prim < _activePrimitive.size();
           ++prim) {
        if (_activePrimitive[prim] == shouldBeActive)
          continue;
        if (!forceAllToggles && prim < _primitiveCooldown.size() &&
            _primitiveCooldown[prim] > 0) {
          groupCanToggle = false;
          break;
        }
        ++groupToggleCount;
      }

      if (!groupCanToggle)
        break;
    }

    if (!groupNeedsToggle || !groupCanToggle)
      continue;

    if (!forceAllToggles &&
        toggledPrimitiveCount >= _residencyConfig.energyMaxTogglesPerFrame)
      continue;

    size_t toggledThisGroup = 0;
    for (size_t objectIndex : *objectIndices) {
      bool shouldBeActive =
          (objectIndex < desiredObjectState.size())
              ? desiredObjectState[objectIndex]
              : groupShouldBeActive;
      toggledThisGroup += attemptToggleObject(objectIndex, shouldBeActive);
    }

    if (toggledThisGroup > 0) {
      changed = true;
    }
  }

  bool anyActiveObject = false;
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    if (objectIndex >= _objectActive.size() ||
        objectIndex >= objectPrimitiveCounts.size())
      continue;
    if (objectPrimitiveCounts[objectIndex] == 0)
      continue;
    if (_objectActive[objectIndex]) {
      anyActiveObject = true;
      break;
    }
  }

  if (!anyActiveObject) {
    for (size_t idx : meshSortedIndices) {
      if (idx >= meshGroups.size())
        continue;
      const MeshGroupAggregate &group = meshGroups[idx];
      if (group.primitiveCount == 0)
        continue;
      const auto *objectIndices = group.info ? &group.info->objectIndices : nullptr;
      if (!objectIndices)
        continue;
      size_t toggled = 0;
      for (size_t objectIndex : *objectIndices)
        toggled += attemptToggleObject(objectIndex, true);
      if (toggled > 0) {
        changed = true;
        anyActiveObject = true;
      }
      if (anyActiveObject)
        break;
    }
  }

  if (!anyActiveObject && !_activePrimitive.empty()) {
    size_t fallbackGroup = std::numeric_limits<size_t>::max();
    for (size_t idx : meshSortedIndices) {
      if (idx >= meshGroups.size())
        continue;
      if (meshGroups[idx].primitiveCount == 0)
        continue;
      fallbackGroup = idx;
      break;
    }

    if (fallbackGroup < meshGroups.size()) {
      size_t toggled = 0;
      const auto *objectIndices =
          meshGroups[fallbackGroup].info
              ? &meshGroups[fallbackGroup].info->objectIndices
              : nullptr;
      if (objectIndices) {
        for (size_t objectIndex : *objectIndices)
          toggled += attemptToggleObject(objectIndex, true);
      }
      if (toggled > 0)
        changed = true;
    } else {
      if (setPrimitiveActive(0, true))
        changed = true;
    }
  }

  return changed;
}

bool Renderer::updateUnifiedResidency(bool forceAllToggles) {
  float totalUnifiedScore = 0.0f;
  std::vector<float> unifiedScores = computeUnifiedImportance(totalUnifiedScore);
  if (unifiedScores.empty())
    return false;

  std::vector<float> originalImportance = _primitiveImportance;
  float originalTotalImportance = 0.0f;
  for (float importance : originalImportance)
    originalTotalImportance += std::max(importance, 0.0f);
  float originalVisibilityBoost = _residencyConfig.energyVisibilityBoost;

  _primitiveImportance = std::move(unifiedScores);
  _totalPrimitiveImportance = totalUnifiedScore;
  _residencyConfig.energyVisibilityBoost = 1.0f;

  bool changed = updateEnergyImportance(forceAllToggles);

  _primitiveImportance = std::move(originalImportance);
  _totalPrimitiveImportance = originalTotalImportance;
  _residencyConfig.energyVisibilityBoost = originalVisibilityBoost;
  return changed;
}

bool Renderer::updateRayHitBudget(bool forceAllToggles) {
  if (_activePrimitive.empty())
    return false;

  if (_rayHitSortedIndices.size() != _activePrimitive.size()) {
    _rayHitSortedIndices.resize(_activePrimitive.size());
    std::iota(_rayHitSortedIndices.begin(), _rayHitSortedIndices.end(), size_t(0));
  }
  if (_primitiveHitScores.size() < _activePrimitive.size())
    _primitiveHitScores.resize(_activePrimitive.size(), 0.0f);

  if (_rayHitRebuildCooldown > 0) {
    if (forceAllToggles)
      _rayHitRebuildCooldown = 0;
    else {
      --_rayHitRebuildCooldown;
      return false;
    }
  }

  const size_t primCount = _activePrimitive.size();
  if (_primitiveHitScoresSnapshot.size() < primCount)
    _primitiveHitScoresSnapshot.resize(primCount, 0.0f);

  size_t copyCount = std::min(_primitiveHitScores.size(), primCount);
  std::copy_n(_primitiveHitScores.begin(), copyCount,
              _primitiveHitScoresSnapshot.begin());
  if (copyCount < primCount) {
    std::fill(_primitiveHitScoresSnapshot.begin() + copyCount,
              _primitiveHitScoresSnapshot.begin() + primCount, 0.0f);
  }

  if (_primitiveVisible.size() < primCount)
    _primitiveVisible.resize(primCount, 0);
  if (_primitiveHitLastFrame.size() < primCount)
    _primitiveHitLastFrame.resize(primCount, 0);

  auto &hitScores = _primitiveHitScoresSnapshot;
  parallelChunkedAsync(0, primCount, [&](size_t chunkBegin, size_t chunkEnd) {
    for (size_t i = chunkBegin; i < chunkEnd; ++i) {
      bool visible = false;
      if (i < _primitiveBounds.size())
        visible = isInView(_primitiveBounds[i]);
      _primitiveVisible[i] = visible ? 1 : 0;

      float score = (i < hitScores.size()) ? hitScores[i] : 0.0f;
      uint32_t hitsLast =
          (i < _primitiveHitLastFrame.size()) ? _primitiveHitLastFrame[i] : 0;

      if (!visible) {
        if (hitsLast == 0)
          score = 0.0f;
        else
          score *= 0.5f;
      } else if (score <= 0.0f) {
        score = 1.0f;
      }

      if (i < hitScores.size())
        hitScores[i] = score;
    }
  });

  const auto &adjustedScores = hitScores;

  const size_t minActive =
      std::min(primCount, _residencyConfig.rayHitMinActivePrimitives);
  size_t targetActive = static_cast<size_t>(
      std::ceil(primCount * _residencyConfig.rayHitTargetFraction));
  targetActive = std::max(targetActive, minActive);
  targetActive = std::min(targetActive, primCount);

  size_t sortCount = std::max(minActive, targetActive);
  sortCount = std::max<size_t>(sortCount, 1);
  sortCount = std::min(sortCount, primCount);

  if (sortCount > 0) {
    std::partial_sort(_rayHitSortedIndices.begin(),
                      _rayHitSortedIndices.begin() + sortCount,
                      _rayHitSortedIndices.end(),
                      [&adjustedScores](size_t a, size_t b) {
                        float rawA =
                            (a < adjustedScores.size()) ? adjustedScores[a] : 0.0f;
                        float rawB =
                            (b < adjustedScores.size()) ? adjustedScores[b] : 0.0f;
                        float scoreA = sanitizeSortValue(rawA);
                        float scoreB = sanitizeSortValue(rawB);
                        if (scoreA == scoreB)
                          return a < b;
                        return scoreA > scoreB;
                      });
  }

  std::vector<bool> desired(primCount, false);
  size_t enabled = 0;
  for (size_t idx : _rayHitSortedIndices) {
    if (enabled >= targetActive)
      break;
    desired[idx] = true;
    ++enabled;
  }

  for (size_t i = 0; i < minActive && i < _rayHitSortedIndices.size(); ++i)
    desired[_rayHitSortedIndices[i]] = true;

  size_t toggles = 0;
  bool changed = false;
  for (size_t i = 0; i < primCount; ++i) {
    bool shouldBeActive = desired[i];
    if (shouldBeActive == _activePrimitive[i])
      continue;
    if (!forceAllToggles) {
      if (i < _primitiveCooldown.size() && _primitiveCooldown[i] > 0)
        continue;
      if (toggles >= _residencyConfig.rayHitMaxTogglesPerFrame)
        break;
    }
    if (setPrimitiveActive(i, shouldBeActive)) {
      ++toggles;
      changed = true;
    }
  }

  size_t activeCount = 0;
  for (bool active : _activePrimitive)
    if (active)
      ++activeCount;

  if (activeCount == 0 && !_activePrimitive.empty()) {
    size_t fallback = !_rayHitSortedIndices.empty() ? _rayHitSortedIndices.front()
                                                    : size_t(0);
    if (setPrimitiveActive(fallback, true))
      changed = true;
  }

  if (changed)
    _rayHitRebuildCooldown = _residencyConfig.rayHitRebuildCooldownFrames;

  return changed;
}

bool Renderer::updateProbabilisticResidency(bool forceAllToggles) {
  if (_activePrimitive.empty())
    return false;

  _frameProbabilityTargetPrimitives = 0;
  _frameProbabilityInitialDesiredPrimitives = 0;
  _frameProbabilityFinalDesiredPrimitives = 0;
  _frameProbabilityTrimmedPrimitives = 0;
  _frameProbabilityBudgetHit = false;

  const size_t primCount = _activePrimitive.size();
  const size_t objectCount = _allSceneObjects.size();

  constexpr float kPosteriorFloor = 1.0e-3f;
  constexpr float kMinimalEvidenceThreshold = 1.0e-3f;
  constexpr float kVisibilityEvidenceDecay = 0.9f;
  const float configuredWindow = _residencyConfig.probabilityEvidenceWindow;
  const bool finiteEvidenceWindow =
      configuredWindow > 0.0f && std::isfinite(configuredWindow);
  float probabilityVisibleFloor =
      std::clamp(_residencyConfig.probabilityVisibleFloor, 0.0f, 1.0f);
  float scoringWindow = finiteEvidenceWindow
                            ? std::max(configuredWindow, kPosteriorFloor)
                            : std::max(64.0f, kPosteriorFloor);
  auto computeEvidenceFactor = [&](float mass, bool visible = true) {
    if (!(mass > 0.0f) || !std::isfinite(mass))
      return 0.0f;
    if (!visible)
      return 0.0f;
    if (finiteEvidenceWindow) {
      float normalized = mass / scoringWindow;
      return std::clamp(normalized, 0.0f, 1.0f);
    }
    float normalized = mass / (mass + scoringWindow);
    return std::clamp(normalized, 0.0f, 1.0f);
  };
  auto sanitizePosteriorProbability = [](float probability) {
    if (!std::isfinite(probability))
      return 0.5f;
    return std::clamp(probability, 0.0f, 1.0f);
  };
  auto sanitizePosteriorVariance = [](float variance) {
    if (!std::isfinite(variance) || variance < 0.0f)
      return 0.0f;
    return variance;
  };
  auto computeRegressedProbabilityFromEvidence =
      [&](float probability, float evidence) {
        float clampedEvidence = std::clamp(evidence, 0.0f, 1.0f);
        float sanitized = sanitizePosteriorProbability(probability);
        return sanitized * clampedEvidence + 0.5f * (1.0f - clampedEvidence);
      };
  auto computeRegressedProbability = [&](float probability, float mass,
                                         bool visible = true) {
    float evidence = computeEvidenceFactor(mass, visible);
    return computeRegressedProbabilityFromEvidence(probability, evidence);
  };
  auto computePosteriorScoreFromEvidence =
      [&](float probability, float variance, float evidence) {
        float regressed =
            computeRegressedProbabilityFromEvidence(probability, evidence);
        float sqrtVariance =
            std::sqrt(std::max(sanitizePosteriorVariance(variance), 0.0f));
        return regressed +
               _residencyConfig.probabilityUncertaintyBoost * sqrtVariance;
      };
  auto computePosteriorScore = [&](float probability, float variance, float mass,
                                   bool visible = true) {
    float evidence = computeEvidenceFactor(mass, visible);
    return computePosteriorScoreFromEvidence(probability, variance, evidence);
  };

  if (objectCount == 0) {
    _desiredObjectState.clear();
    _pendingDesiredObjects.clear();
    _desiredObjectPromotionFrame.clear();
    _desiredObjectDemotionFrame.clear();
    _objectVisibilityEvidence.clear();
    if (_primitiveHitProbability.size() < primCount)
      _primitiveHitProbability.resize(primCount, 0.5f);
    if (_probabilitySortedIndices.size() != primCount) {
      _probabilitySortedIndices.resize(primCount);
      std::iota(_probabilitySortedIndices.begin(), _probabilitySortedIndices.end(),
                size_t(0));
    }

    if (_primitiveVisible.size() < primCount)
      _primitiveVisible.resize(primCount, 0);
    if (_primitiveExplorationScore.size() < primCount)
      _primitiveExplorationScore.resize(primCount, 0.0f);

    std::vector<uint8_t> primitiveBecameVisible(primCount, 0);
    std::vector<bool> desired(primCount, false);
    float threshold = _residencyConfig.probabilityThreshold;
    for (size_t i = 0; i < primCount; ++i) {
      float probability = (i < _primitiveHitProbability.size())
                              ? _primitiveHitProbability[i]
                              : 0.5f;
      float mass = (i < _primitivePosteriorMass.size())
                       ? _primitivePosteriorMass[i]
                       : 0.0f;
      float effectiveProbability = computeRegressedProbability(probability, mass);
      if (effectiveProbability >= threshold)
        desired[i] = true;
    }

    size_t minActive =
        std::min(_residencyConfig.probabilityMinActivePrimitives, primCount);
    size_t partialCount = std::max<size_t>(
        1, std::min(primCount, std::max(minActive, size_t(1))));
    auto comparator = [this, &computePosteriorScore](size_t a, size_t b) {
      float probA = (a < _primitiveHitProbability.size())
                        ? _primitiveHitProbability[a]
                        : 0.5f;
      float varA = (a < _primitiveHitVariance.size())
                       ? _primitiveHitVariance[a]
                       : 0.0f;
      float massA = (a < _primitivePosteriorMass.size())
                        ? _primitivePosteriorMass[a]
                        : 0.0f;
      float scoreA = sanitizeSortValue(computePosteriorScore(probA, varA, massA));
      float probB = (b < _primitiveHitProbability.size())
                        ? _primitiveHitProbability[b]
                        : 0.5f;
      float varB = (b < _primitiveHitVariance.size())
                       ? _primitiveHitVariance[b]
                       : 0.0f;
      float massB = (b < _primitivePosteriorMass.size())
                        ? _primitivePosteriorMass[b]
                        : 0.0f;
      float scoreB = sanitizeSortValue(computePosteriorScore(probB, varB, massB));
      if (scoreA == scoreB)
        return a < b;
      return scoreA > scoreB;
    };
    std::partial_sort(_probabilitySortedIndices.begin(),
                      _probabilitySortedIndices.begin() + partialCount,
                      _probabilitySortedIndices.end(), comparator);

    auto computeVisibility = [this, &primitiveBecameVisible](size_t idx) {
      bool previousVisible =
          (idx < _primitiveVisible.size()) ? (_primitiveVisible[idx] != 0) : false;
      bool visible = false;
      if (idx < _primitiveBounds.size()) {
        visible = isInView(_primitiveBounds[idx]);
        if (idx < _primitiveVisible.size())
          _primitiveVisible[idx] = visible ? 1 : 0;
      } else if (idx < _primitiveVisible.size()) {
        visible = _primitiveVisible[idx] != 0;
      }
      if (visible && !previousVisible && idx < primitiveBecameVisible.size())
        primitiveBecameVisible[idx] = 1;
      return visible;
    };

    size_t desiredCount = 0;
    for (bool flag : desired)
      if (flag)
        ++desiredCount;

    std::vector<size_t> visibleExplore;
    std::vector<size_t> hiddenExplore;
    visibleExplore.reserve(primCount);
    hiddenExplore.reserve(primCount);

    for (size_t idx : _probabilitySortedIndices) {
      if (idx >= primCount)
        continue;
      if (idx < _activePrimitive.size() && _activePrimitive[idx])
        continue;
      if (desired[idx])
        continue;
      float probability = (idx < _primitiveHitProbability.size())
                              ? _primitiveHitProbability[idx]
                              : 0.5f;
      float mass = (idx < _primitivePosteriorMass.size())
                       ? _primitivePosteriorMass[idx]
                       : 0.0f;
      bool visible = computeVisibility(idx);
      float evidence = computeEvidenceFactor(mass, visible);
      bool lowEvidence = evidence <= kMinimalEvidenceThreshold;
      float effectiveProbability = computeRegressedProbabilityFromEvidence(
          probability, evidence);
      bool visibilityBootstrap = (idx < primitiveBecameVisible.size()) &&
                                 primitiveBecameVisible[idx] != 0 &&
                                 mass <= kMinimalEvidenceThreshold;
      float visibilityAdjustedProbability =
          (visible && lowEvidence)
              ? std::min(effectiveProbability, threshold)
              : effectiveProbability;
      float gatedProbability =
          visibilityBootstrap
              ? std::min(visibilityAdjustedProbability, threshold)
              : visibilityAdjustedProbability;
      if (gatedProbability > threshold)
        continue;
      float exploreScore =
          (idx < _primitiveExplorationScore.size())
              ? _primitiveExplorationScore[idx]
              : 0.0f;
      uint32_t raysTested =
          (idx < _primitiveRaysTestedLastFrame.size())
              ? _primitiveRaysTestedLastFrame[idx]
              : 0u;
      float effectiveExplore = exploreScore;
      if (visibilityBootstrap)
        effectiveExplore = std::max(effectiveExplore, kIdleVisibleExploreSeed);
      if (visible && (raysTested == 0 || lowEvidence))
        effectiveExplore = std::max(effectiveExplore, kIdleVisibleExploreSeed);
      if (visible && raysTested > 0 && !lowEvidence)
        effectiveExplore = std::max(effectiveExplore, kPosteriorFloor);
      if (idx < _primitiveExplorationScore.size())
        _primitiveExplorationScore[idx] = effectiveExplore;
      if (effectiveExplore <= 0.0f)
        continue;
      if (visible)
        visibleExplore.push_back(idx);
      else
        hiddenExplore.push_back(idx);
    }

    auto explorationComparator = [this](size_t a, size_t b) {
      float exploreA = (a < _primitiveExplorationScore.size())
                           ? sanitizeSortValue(_primitiveExplorationScore[a])
                           : -std::numeric_limits<float>::max();
      float exploreB = (b < _primitiveExplorationScore.size())
                           ? sanitizeSortValue(_primitiveExplorationScore[b])
                           : -std::numeric_limits<float>::max();
      if (exploreA == exploreB) {
        uint32_t raysA =
            (a < _primitiveRaysTestedLastFrame.size())
                ? _primitiveRaysTestedLastFrame[a]
                : 0u;
        uint32_t raysB =
            (b < _primitiveRaysTestedLastFrame.size())
                ? _primitiveRaysTestedLastFrame[b]
                : 0u;
        if (raysA == raysB)
          return a < b;
        return raysA > raysB;
      }
      return exploreA > exploreB;
    };

    std::sort(visibleExplore.begin(), visibleExplore.end(),
              explorationComparator);
    std::sort(hiddenExplore.begin(), hiddenExplore.end(),
              explorationComparator);

    auto promote = [&](const std::vector<size_t> &candidates, size_t &slots) {
      if (slots == 0)
        return;
      for (size_t idx : candidates) {
        if (slots == 0)
          break;
        if (idx >= primCount)
          continue;
        if (desired[idx])
          continue;
        desired[idx] = true;
        ++desiredCount;
        if (slots > 0)
          --slots;
        bool wasVisible =
            (idx < _primitiveVisible.size()) ? (_primitiveVisible[idx] != 0)
                                             : false;
        bool idle =
            (idx < _primitiveRaysTestedLastFrame.size())
                ? (_primitiveRaysTestedLastFrame[idx] == 0)
                : true;
        if (wasVisible && idle && idx < _primitiveExplorationScore.size() &&
            _primitiveExplorationScore[idx] < kIdleVisibleExploreSeed)
          _primitiveExplorationScore[idx] = kIdleVisibleExploreSeed;
      }
    };

    auto countRemaining = [&](const std::vector<size_t> &candidates) {
      size_t remaining = 0;
      for (size_t idx : candidates) {
        if (idx < primCount && !desired[idx])
          ++remaining;
      }
      return remaining;
    };

    if (desiredCount < minActive) {
      size_t slots =
          std::min(minActive - desiredCount, primCount - desiredCount);
      promote(visibleExplore, slots);
      promote(hiddenExplore, slots);
    }

    if (desiredCount < minActive) {
      for (size_t idx : _probabilitySortedIndices) {
        if (desiredCount >= minActive)
          break;
        if (idx >= primCount)
          continue;
        if (desired[idx])
          continue;
        desired[idx] = true;
        ++desiredCount;
      }
    }

    size_t remainingExplore = countRemaining(visibleExplore) +
                              countRemaining(hiddenExplore);
    if (remainingExplore > 0 && desiredCount < primCount) {
      size_t slots =
          std::min({std::max<size_t>(size_t(1), minActive / 2), remainingExplore,
                    primCount - desiredCount});
      promote(visibleExplore, slots);
      promote(hiddenExplore, slots);
    }

    size_t fallbackCandidate = primCount;
    if (!visibleExplore.empty())
      fallbackCandidate = visibleExplore.front();
    else if (!hiddenExplore.empty())
      fallbackCandidate = hiddenExplore.front();
    else if (!_probabilitySortedIndices.empty())
      fallbackCandidate = _probabilitySortedIndices.front();

    size_t toggles = 0;
    bool changed = false;
    size_t maxToggles = _residencyConfig.probabilityMaxTogglesPerFrame;
#ifndef NDEBUG
    assert(_probabilitySortedIndices.size() == primCount);
#endif
    auto tryToggle = [&](size_t idx) {
      if (idx >= primCount)
        return false;
      bool shouldBeActive = desired[idx];
      if (shouldBeActive == _activePrimitive[idx])
        return false;
      if (!forceAllToggles) {
        if (idx < _primitiveCooldown.size() && _primitiveCooldown[idx] > 0)
          return false;
        if (toggles >= maxToggles)
          return true;
      }
      if (setPrimitiveActive(idx, shouldBeActive)) {
        ++toggles;
        ++_frameProbabilisticToggles;
        changed = true;
        if (!shouldBeActive && idx < _primitiveExplorationScore.size())
          _primitiveExplorationScore[idx] = 0.0f;
      }
      return false;
    };

    auto walkList = [&](const std::vector<size_t> &candidates) {
      for (size_t idx : candidates) {
        if (!forceAllToggles && toggles >= maxToggles)
          return true;
        if (tryToggle(idx))
          return true;
      }
      return false;
    };

    // Walk probability-sorted primitives first so that we spend the toggle budget
    // on the most-likely contributors and then fall back to the exploration order
    // as needed.
    bool budgetReached = walkList(_probabilitySortedIndices);
    if (!budgetReached)
      budgetReached = walkList(visibleExplore);
    if (!budgetReached)
      budgetReached = walkList(hiddenExplore);

#ifndef NDEBUG
    if (!forceAllToggles)
      assert(toggles <= maxToggles);
#endif

    size_t activeCount = 0;
    for (bool active : _activePrimitive)
      if (active)
        ++activeCount;

    if (activeCount == 0 && primCount > 0) {
      size_t fallback = fallbackCandidate < primCount ? fallbackCandidate
                                                      : size_t(0);
      if (fallback >= primCount)
        fallback = primCount - 1;
      if (setPrimitiveActive(fallback, true)) {
        ++_frameProbabilisticToggles;
        changed = true;
      }
    }

    return changed;
  }

  if (_objectHitProbability.size() < objectCount)
    _objectHitProbability.resize(objectCount, 0.5f);
  if (_objectHitVariance.size() < objectCount)
    _objectHitVariance.resize(objectCount, 1.0f / 12.0f);
  if (_objectPosteriorMass.size() < objectCount)
    _objectPosteriorMass.resize(objectCount, 2.0f);
  if (_objectProbabilitySortedIndices.size() != objectCount) {
    _objectProbabilitySortedIndices.resize(objectCount);
    std::iota(_objectProbabilitySortedIndices.begin(),
              _objectProbabilitySortedIndices.end(), size_t(0));
  }
  if (_objectVisible.size() < objectCount)
    _objectVisible.resize(objectCount, 0);
  if (_objectVisibilityEvidence.size() < objectCount)
    _objectVisibilityEvidence.resize(objectCount, 0.0f);
  if (_objectExplorationScore.size() < objectCount)
    _objectExplorationScore.resize(objectCount, 0.0f);
  if (_objectRaysTestedLastFrame.size() < objectCount)
    _objectRaysTestedLastFrame.resize(objectCount, 0u);

  if (_desiredObjectState.size() < objectCount)
    _desiredObjectState.resize(objectCount, 0);
  else if (_desiredObjectState.size() > objectCount)
    _desiredObjectState.resize(objectCount);
  if (_pendingDesiredObjects.size() < objectCount)
    _pendingDesiredObjects.resize(objectCount, 0);
  else if (_pendingDesiredObjects.size() > objectCount)
    _pendingDesiredObjects.resize(objectCount);

  if (_desiredObjectPromotionFrame.size() < objectCount)
    _desiredObjectPromotionFrame.resize(objectCount, 0);
  else if (_desiredObjectPromotionFrame.size() > objectCount)
    _desiredObjectPromotionFrame.resize(objectCount);
  if (_desiredObjectDemotionFrame.size() < objectCount)
    _desiredObjectDemotionFrame.resize(objectCount, 0);
  else if (_desiredObjectDemotionFrame.size() > objectCount)
    _desiredObjectDemotionFrame.resize(objectCount);
  if (_objectDemotionDwell.size() < objectCount)
    _objectDemotionDwell.resize(objectCount, 0);
  else if (_objectDemotionDwell.size() > objectCount)
    _objectDemotionDwell.resize(objectCount);

  std::vector<uint8_t> desiredObjects = _desiredObjectState;
  auto &pendingDesiredObjects = _pendingDesiredObjects;
  std::vector<uint8_t> objectBecameVisible(objectCount, 0);
  auto computeObjectVisibility = [this, &objectBecameVisible](size_t idx) {
    bool previousVisible =
        (idx < _objectVisible.size()) ? (_objectVisible[idx] != 0) : false;
    bool visible = false;
    if (idx < _objectBounds.size()) {
      visible = isInView(_objectBounds[idx]);
      if (idx < _objectVisible.size())
        _objectVisible[idx] = visible ? 1 : 0;
    } else if (idx < _objectVisible.size()) {
      visible = _objectVisible[idx] != 0;
    }
    if (visible && !previousVisible && idx < objectBecameVisible.size())
      objectBecameVisible[idx] = 1;
    return visible;
  };

  std::vector<uint8_t> objectVisibility(objectCount, 0);
  for (size_t i = 0; i < objectCount; ++i) {
    bool visible = computeObjectVisibility(i);
    objectVisibility[i] = visible ? 1 : 0;
  }

  auto visibilityForIndex = [&](size_t idx) {
    if (idx < objectVisibility.size())
      return objectVisibility[idx] != 0;
    return computeObjectVisibility(idx);
  };
  auto bufferedEvidenceForIndex = [&](size_t idx, float rawEvidence,
                                      bool visible) {
    float previous =
        (idx < _objectVisibilityEvidence.size()) ? _objectVisibilityEvidence[idx]
                                                 : 0.0f;
    float decayed = previous * kVisibilityEvidenceDecay;
    float buffered = std::max(std::clamp(rawEvidence, 0.0f, 1.0f), decayed);
    if (idx < _objectVisibilityEvidence.size())
      _objectVisibilityEvidence[idx] = buffered;
    return buffered;
  };
  float threshold = _residencyConfig.probabilityThreshold;
  float hysteresis =
      std::clamp(_residencyConfig.probabilityDesiredHysteresis, 0.0f, 0.5f);
  float enterThreshold = std::clamp(threshold + hysteresis, 0.0f, 1.0f);
  float exitThreshold = std::clamp(threshold - hysteresis, 0.0f, 1.0f);
  uint32_t demotionDwellFrames =
      _residencyConfig.probabilityVisibleDemotionDwellFrames;

  size_t desiredPrimitiveCount = 0;
  for (size_t i = 0; i < objectCount; ++i) {
    float probability = (i < _objectHitProbability.size())
                            ? _objectHitProbability[i]
                            : 0.5f;
    float variance = (i < _objectHitVariance.size()) ? _objectHitVariance[i]
                                                     : 0.0f;
    float mass = (i < _objectPosteriorMass.size())
                     ? _objectPosteriorMass[i]
                     : 0.0f;
    bool visible = visibilityForIndex(i);
    float rawEvidence = computeEvidenceFactor(mass, visible);
    float bufferedEvidence = bufferedEvidenceForIndex(i, rawEvidence, visible);
    float sanitizedProbability = sanitizePosteriorProbability(probability);
    float effectiveProbability = computeRegressedProbabilityFromEvidence(
        sanitizedProbability, bufferedEvidence);
    float boostedProbability = computePosteriorScoreFromEvidence(
        probability, variance, bufferedEvidence);
    float enterScore = std::max(effectiveProbability, boostedProbability);
    float exitScore = effectiveProbability;
    bool previousDesired = desiredObjects[i] != 0;
    bool desired = previousDesired;
    bool cooldownExpired =
        (i >= _objectCooldown.size()) || _objectCooldown[i] == 0;
    float lowEvidenceMetric =
        std::min(bufferedEvidence, std::clamp(rawEvidence, 0.0f, 1.0f));
    bool lowEvidence = lowEvidenceMetric <= kMinimalEvidenceThreshold;
    float promotionProbability = enterScore;
    float demotionProbability = exitScore;
    float evaluationProbability = lowEvidence ? effectiveProbability
                                              : enterScore;

    bool dwellApplies = visible && demotionDwellFrames > 0;
    if (!dwellApplies || demotionProbability > exitThreshold)
      _objectDemotionDwell[i] = 0;

    if (visible) {
      demotionProbability =
          std::max(demotionProbability, probabilityVisibleFloor);
      evaluationProbability =
          std::max(evaluationProbability, probabilityVisibleFloor);
    }

    if (promotionProbability >= enterThreshold) {
      desired = true;
      _objectDemotionDwell[i] = 0;
    } else if (demotionProbability <= exitThreshold) {
      if (dwellApplies && previousDesired) {
        uint32_t &dwellCount = _objectDemotionDwell[i];
        if (dwellCount < demotionDwellFrames)
          ++dwellCount;
        if (dwellCount >= demotionDwellFrames)
          desired = false;
        else
          desired = true;
      } else {
        desired = false;
        _objectDemotionDwell[i] = 0;
      }
    } else if (cooldownExpired && !previousDesired) {
      desired = evaluationProbability >= threshold;
      if (desired)
        _objectDemotionDwell[i] = 0;
    }

    if (desired && !previousDesired && i < _desiredObjectPromotionFrame.size())
      _desiredObjectPromotionFrame[i] = _renderedFrameCount;
    else if (!desired && previousDesired &&
             i < _desiredObjectDemotionFrame.size())
      _desiredObjectDemotionFrame[i] = _renderedFrameCount;
    desiredObjects[i] = desired ? 1 : 0;
    if (!desired && i < pendingDesiredObjects.size())
      pendingDesiredObjects[i] = 0;
    if (desired) {
      bool pending =
          (i < pendingDesiredObjects.size()) ? pendingDesiredObjects[i] != 0
                                             : false;
      size_t contribution =
          (i < _objectPrimitiveCounts.size()) ? _objectPrimitiveCounts[i] : 0;
      if (contribution == 0)
        contribution = 1;
      if (!pending)
        desiredPrimitiveCount += contribution;
    }
  }

  size_t minActivePrimitives =
      std::min(_residencyConfig.probabilityMinActivePrimitives, primCount);

  auto primitiveContribution = [this](size_t idx) {
    size_t count =
        (idx < _objectPrimitiveCounts.size()) ? _objectPrimitiveCounts[idx] : 0;
    return std::max<size_t>(count, 1);
  };

  std::sort(_objectProbabilitySortedIndices.begin(),
            _objectProbabilitySortedIndices.end(),
            [this, &computePosteriorScore, &visibilityForIndex](size_t a,
                                                               size_t b) {
              float probA = (a < _objectHitProbability.size())
                                ? _objectHitProbability[a]
                                : 0.5f;
              float varA = (a < _objectHitVariance.size())
                               ? _objectHitVariance[a]
                               : 0.0f;
              float massA = (a < _objectPosteriorMass.size())
                                ? _objectPosteriorMass[a]
                                : 0.0f;
              float scoreA =
                  sanitizeSortValue(computePosteriorScore(
                      probA, varA, massA, visibilityForIndex(a)));
              float probB = (b < _objectHitProbability.size())
                                ? _objectHitProbability[b]
                                : 0.5f;
              float varB = (b < _objectHitVariance.size())
                               ? _objectHitVariance[b]
                               : 0.0f;
              float massB = (b < _objectPosteriorMass.size())
                                ? _objectPosteriorMass[b]
                                : 0.0f;
              float scoreB =
                  sanitizeSortValue(computePosteriorScore(
                      probB, varB, massB, visibilityForIndex(b)));
              if (scoreA == scoreB)
                return a < b;
              return scoreA > scoreB;
            });

  float targetFraction =
      std::clamp(_residencyConfig.probabilityTargetFraction, 0.0f, 1.0f);
  size_t targetPrimitiveBudget = static_cast<size_t>(std::ceil(
      static_cast<float>(primCount) * targetFraction));
  targetPrimitiveBudget = std::max(targetPrimitiveBudget, minActivePrimitives);
  if (targetPrimitiveBudget == 0 && primCount > 0)
    targetPrimitiveBudget = 1;
  targetPrimitiveBudget = std::min(targetPrimitiveBudget, primCount);

  _frameProbabilityTargetPrimitives = targetPrimitiveBudget;
  _frameProbabilityInitialDesiredPrimitives = desiredPrimitiveCount;

  size_t trimmedPrimitives = 0;
  if (targetPrimitiveBudget > 0 &&
      desiredPrimitiveCount > targetPrimitiveBudget) {
    for (size_t position = _objectProbabilitySortedIndices.size();
         position-- > 0;) {
      if (desiredPrimitiveCount <= targetPrimitiveBudget)
        break;
      size_t idx = _objectProbabilitySortedIndices[position];
      if (idx >= objectCount)
        continue;
      if (desiredObjects[idx] == 0)
        continue;
      bool pending =
          (idx < pendingDesiredObjects.size()) ? pendingDesiredObjects[idx] != 0
                                               : false;
      if (pending)
        continue;
      size_t contribution = primitiveContribution(idx);
      if (contribution == 0)
        continue;
      desiredObjects[idx] = 0;
      if (idx < pendingDesiredObjects.size())
        pendingDesiredObjects[idx] = 0;
      trimmedPrimitives += contribution;
      if (idx < _desiredObjectDemotionFrame.size())
        _desiredObjectDemotionFrame[idx] = _renderedFrameCount;
      desiredPrimitiveCount =
          (desiredPrimitiveCount > contribution)
              ? desiredPrimitiveCount - contribution
              : size_t(0);
    }
  }

  _frameProbabilityTrimmedPrimitives = trimmedPrimitives;
  _frameProbabilityBudgetHit = trimmedPrimitives > 0;

  std::vector<size_t> visibleExplore;
  std::vector<size_t> hiddenExplore;
  visibleExplore.reserve(objectCount);
  hiddenExplore.reserve(objectCount);

  for (size_t idx : _objectProbabilitySortedIndices) {
    if (idx >= objectCount)
      continue;
    bool currentlyActive =
        idx < _objectActive.size() ? _objectActive[idx] : false;
    if (currentlyActive)
      continue;
    if (desiredObjects[idx] != 0)
      continue;
    float probability = (idx < _objectHitProbability.size())
                            ? _objectHitProbability[idx]
                            : 0.5f;
    float mass = (idx < _objectPosteriorMass.size())
                     ? _objectPosteriorMass[idx]
                     : 0.0f;
    bool visible = visibilityForIndex(idx);
    float rawEvidence = computeEvidenceFactor(mass, visible);
    float bufferedEvidence = (idx < _objectVisibilityEvidence.size())
                                 ? _objectVisibilityEvidence[idx]
                                 : rawEvidence;
    float effectiveProbability = computeRegressedProbabilityFromEvidence(
        probability, bufferedEvidence);
    bool visibilityBootstrap = (idx < objectBecameVisible.size()) &&
                               objectBecameVisible[idx] != 0 &&
                               mass <= kMinimalEvidenceThreshold;
    float lowEvidenceMetric =
        std::min(bufferedEvidence, std::clamp(rawEvidence, 0.0f, 1.0f));
    bool lowEvidence = lowEvidenceMetric <= kMinimalEvidenceThreshold;
    float visibilityAdjustedProbability =
        (visible && lowEvidence) ? std::min(effectiveProbability, threshold)
                                 : effectiveProbability;
    float gatedProbability = visibilityBootstrap
                                 ? std::min(visibilityAdjustedProbability, threshold)
                                 : visibilityAdjustedProbability;
    if (gatedProbability > threshold)
      continue;
    float exploreScore = (idx < _objectExplorationScore.size())
                             ? _objectExplorationScore[idx]
                             : 0.0f;
    uint32_t raysTested = (idx < _objectRaysTestedLastFrame.size())
                              ? _objectRaysTestedLastFrame[idx]
                              : 0u;
    float effectiveExplore = exploreScore;
    if (visibilityBootstrap)
      effectiveExplore = std::max(effectiveExplore, kIdleVisibleExploreSeed);
    if (visible && (raysTested == 0 || lowEvidence))
      effectiveExplore = std::max(effectiveExplore, kIdleVisibleExploreSeed);
    if (visible && raysTested > 0 && !lowEvidence)
      effectiveExplore = std::max(effectiveExplore, kPosteriorFloor);
    if (idx < _objectExplorationScore.size())
      _objectExplorationScore[idx] = effectiveExplore;
    if (effectiveExplore <= 0.0f)
      continue;
    if (visible)
      visibleExplore.push_back(idx);
    else
      hiddenExplore.push_back(idx);
  }

  auto objectExploreComparator = [this](size_t a, size_t b) {
    float exploreA = (a < _objectExplorationScore.size())
                         ? sanitizeSortValue(_objectExplorationScore[a])
                         : -std::numeric_limits<float>::max();
    float exploreB = (b < _objectExplorationScore.size())
                         ? sanitizeSortValue(_objectExplorationScore[b])
                         : -std::numeric_limits<float>::max();
    if (exploreA == exploreB) {
      uint32_t raysA = (a < _objectRaysTestedLastFrame.size())
                           ? _objectRaysTestedLastFrame[a]
                           : 0u;
      uint32_t raysB = (b < _objectRaysTestedLastFrame.size())
                           ? _objectRaysTestedLastFrame[b]
                           : 0u;
      if (raysA == raysB)
        return a < b;
      return raysA > raysB;
    }
    return exploreA > exploreB;
  };

  std::sort(visibleExplore.begin(), visibleExplore.end(),
            objectExploreComparator);
  std::sort(hiddenExplore.begin(), hiddenExplore.end(),
            objectExploreComparator);

  auto isRecentlyPromoted = [&, window =
                                   _residencyConfig
                                       .probabilityRecentPromotionFrames](
                                  size_t idx) {
    if (window == 0)
      return false;
    uint64_t lastPromotion =
        idx < _desiredObjectPromotionFrame.size()
            ? _desiredObjectPromotionFrame[idx]
            : 0;
    uint64_t lastDemotion =
        idx < _desiredObjectDemotionFrame.size()
            ? _desiredObjectDemotionFrame[idx]
            : 0;
    uint64_t lastFrame = std::max(lastPromotion, lastDemotion);
    if (lastFrame == 0)
      return false;
    uint64_t currentFrame = _renderedFrameCount;
    if (currentFrame <= lastFrame)
      return true;
    return currentFrame - lastFrame < window;
  };

  auto markDesired = [&](size_t idx) {
    desiredObjects[idx] = 1;
    if (idx < _desiredObjectPromotionFrame.size())
      _desiredObjectPromotionFrame[idx] = _renderedFrameCount;
    size_t contribution = primitiveContribution(idx);
    bool pending =
        (idx < pendingDesiredObjects.size()) ? pendingDesiredObjects[idx] != 0
                                             : false;
    if (!pending)
      desiredPrimitiveCount += contribution;
    bool wasVisible =
        (idx < _objectVisible.size()) ? (_objectVisible[idx] != 0) : false;
    bool idle =
        (idx < _objectRaysTestedLastFrame.size())
            ? (_objectRaysTestedLastFrame[idx] == 0)
            : true;
    if (wasVisible && idle && idx < _objectExplorationScore.size() &&
        _objectExplorationScore[idx] < kIdleVisibleExploreSeed)
      _objectExplorationScore[idx] = kIdleVisibleExploreSeed;
    return contribution;
  };

  auto promoteObjects = [&](const std::vector<size_t> &candidates,
                            size_t &slots, bool allowSuppressed) {
    if (slots == 0)
      return;
    for (size_t idx : candidates) {
      if (slots == 0)
        break;
      if (idx >= objectCount)
        continue;
      if (desiredObjects[idx] != 0)
        continue;
      if (!allowSuppressed && isRecentlyPromoted(idx))
        continue;
      size_t contribution = markDesired(idx);
      if (slots <= contribution)
        slots = 0;
      else
        slots -= contribution;
    }
  };

  auto countRemainingPrimitives = [&](const std::vector<size_t> &candidates) {
    size_t remaining = 0;
    for (size_t idx : candidates) {
      if (idx >= objectCount)
        continue;
      if (desiredObjects[idx] != 0)
        continue;
      bool pending =
          (idx < pendingDesiredObjects.size()) ? pendingDesiredObjects[idx] != 0
                                               : false;
      if (pending)
        continue;
      remaining += primitiveContribution(idx);
    }
    return remaining;
  };

  if (desiredPrimitiveCount < minActivePrimitives) {
    size_t slots =
        std::min(minActivePrimitives - desiredPrimitiveCount,
                 primCount > desiredPrimitiveCount
                     ? primCount - desiredPrimitiveCount
                     : size_t(0));
    promoteObjects(visibleExplore, slots, false);
    promoteObjects(hiddenExplore, slots, false);
    if (slots > 0) {
      promoteObjects(visibleExplore, slots, true);
      promoteObjects(hiddenExplore, slots, true);
    }
  }

  if (desiredPrimitiveCount < minActivePrimitives) {
    auto promoteByProbability = [&](bool allowSuppressed) {
      for (size_t idx : _objectProbabilitySortedIndices) {
        if (desiredPrimitiveCount >= minActivePrimitives)
          break;
        if (idx >= objectCount)
          continue;
        if (desiredObjects[idx] != 0)
          continue;
        if (!allowSuppressed && isRecentlyPromoted(idx))
          continue;
        markDesired(idx);
      }
    };
    promoteByProbability(false);
    if (desiredPrimitiveCount < minActivePrimitives)
      promoteByProbability(true);
  }

  size_t remainingExplorePrimitives =
      countRemainingPrimitives(visibleExplore) +
      countRemainingPrimitives(hiddenExplore);
    if (remainingExplorePrimitives > 0 && desiredPrimitiveCount < primCount) {
      size_t slots = std::min({std::max<size_t>(size_t(1),
                                                minActivePrimitives / 2),
                               remainingExplorePrimitives,
                               primCount - desiredPrimitiveCount});
      promoteObjects(visibleExplore, slots, false);
      promoteObjects(hiddenExplore, slots, false);
      if (slots > 0) {
        promoteObjects(visibleExplore, slots, true);
        promoteObjects(hiddenExplore, slots, true);
      }
    }

  desiredPrimitiveCount = 0;
  for (size_t idx = 0; idx < objectCount; ++idx) {
    if (desiredObjects[idx] == 0)
      continue;
    bool pending =
        (idx < pendingDesiredObjects.size()) ? pendingDesiredObjects[idx] != 0
                                             : false;
    if (pending)
      continue;
    desiredPrimitiveCount += primitiveContribution(idx);
  }

  if (targetPrimitiveBudget > 0 && desiredPrimitiveCount > targetPrimitiveBudget) {
    for (size_t position = _objectProbabilitySortedIndices.size(); position-- > 0;) {
      if (desiredPrimitiveCount <= targetPrimitiveBudget)
        break;
      size_t idx = _objectProbabilitySortedIndices[position];
      if (idx >= objectCount)
        continue;
      if (desiredObjects[idx] == 0)
        continue;
      bool pending =
          (idx < pendingDesiredObjects.size()) ? pendingDesiredObjects[idx] != 0
                                               : false;
      if (pending)
        continue;
      size_t contribution = primitiveContribution(idx);
      if (contribution == 0)
        continue;
      desiredObjects[idx] = 0;
      if (idx < pendingDesiredObjects.size())
        pendingDesiredObjects[idx] = 0;
      trimmedPrimitives += contribution;
      if (idx < _desiredObjectDemotionFrame.size())
        _desiredObjectDemotionFrame[idx] = _renderedFrameCount;
      desiredPrimitiveCount =
          (desiredPrimitiveCount > contribution) ? desiredPrimitiveCount - contribution
                                                 : size_t(0);
    }
  }

  _frameProbabilityTrimmedPrimitives = trimmedPrimitives;
  _frameProbabilityBudgetHit = trimmedPrimitives > 0;

  size_t fallbackObject = objectCount;
  if (!visibleExplore.empty())
    fallbackObject = visibleExplore.front();
  else if (!hiddenExplore.empty())
    fallbackObject = hiddenExplore.front();
  else if (!_objectProbabilitySortedIndices.empty())
    fallbackObject = _objectProbabilitySortedIndices.front();

  bool changed = false;
  size_t toggledPrimitiveCount = 0;
  size_t maxPrimitiveToggles = _residencyConfig.probabilityMaxTogglesPerFrame;

  std::vector<size_t> toggleOrder;
  toggleOrder.reserve(objectCount);
  std::vector<uint8_t> added(objectCount, 0);
  std::vector<uint8_t> appliedDesired(objectCount, 0);
  size_t appliedDesiredPrimitiveCount = 0;

  auto appendList = [&](const std::vector<size_t> &candidates) {
    for (size_t idx : candidates) {
      if (idx >= objectCount)
        continue;
      if (added[idx])
        continue;
      toggleOrder.push_back(idx);
      added[idx] = 1;
    }
  };

  std::vector<size_t> desiredByPriority;
  desiredByPriority.reserve(objectCount);
  for (size_t idx : _objectProbabilitySortedIndices) {
    if (idx >= objectCount)
      continue;
    if (desiredObjects[idx] == 0)
      continue;
    desiredByPriority.push_back(idx);
  }

  std::vector<size_t> pendingCandidates;
  pendingCandidates.reserve(objectCount);
  for (size_t idx = 0; idx < objectCount; ++idx) {
    bool pending =
        (idx < pendingDesiredObjects.size()) ? pendingDesiredObjects[idx] != 0
                                             : false;
    if (pending)
      pendingCandidates.push_back(idx);
  }

  appendList(pendingCandidates);
  appendList(desiredByPriority);
  appendList(visibleExplore);
  appendList(hiddenExplore);
  appendList(_objectProbabilitySortedIndices);

  auto markPendingPromotion = [&](size_t idx) {
    if (idx < pendingDesiredObjects.size())
      pendingDesiredObjects[idx] = 1;
  };

  for (size_t objectIndex : toggleOrder) {
    if (!forceAllToggles) {
      if (maxPrimitiveToggles == 0 || toggledPrimitiveCount >= maxPrimitiveToggles)
        break;
    }

    bool shouldBeActive = desiredObjects[objectIndex] != 0;
    bool currentlyActive =
        objectIndex < _objectActive.size() && _objectActive[objectIndex];
    if (shouldBeActive == currentlyActive)
      continue;

    if (!forceAllToggles) {
      if (objectIndex < _objectCooldown.size() &&
          _objectCooldown[objectIndex] > 0) {
        if (shouldBeActive)
          markPendingPromotion(objectIndex);
        continue;
      }

      const SceneObject &obj = _allSceneObjects[objectIndex];
      size_t first = obj.firstPrimitive;
      size_t last = first + obj.primitiveCount;
      size_t pendingToggles = 0;
      bool canToggle = true;
      for (size_t prim = first; prim < last && prim < _activePrimitive.size();
           ++prim) {
        if (_activePrimitive[prim] == shouldBeActive)
          continue;
        ++pendingToggles;
        if (prim < _primitiveCooldown.size() && _primitiveCooldown[prim] > 0) {
          canToggle = false;
          break;
        }
      }

      if (!canToggle) {
        if (shouldBeActive)
          markPendingPromotion(objectIndex);
        continue;
      }
      if (pendingToggles == 0)
        continue;
      size_t remainingBudget =
          (maxPrimitiveToggles > toggledPrimitiveCount)
              ? (maxPrimitiveToggles - toggledPrimitiveCount)
              : size_t(0);
      if (pendingToggles > remainingBudget) {
        if (shouldBeActive)
          markPendingPromotion(objectIndex);
        continue;
      }
    }

    size_t toggled = setObjectActive(objectIndex, shouldBeActive);
    if (toggled > 0) {
      size_t applied = toggled;
      if (!forceAllToggles) {
        size_t remainingBudget =
            (maxPrimitiveToggles > toggledPrimitiveCount)
                ? (maxPrimitiveToggles - toggledPrimitiveCount)
                : size_t(0);
        applied = std::min(toggled, remainingBudget);
        toggledPrimitiveCount += applied;
      }
      if (shouldBeActive && !appliedDesired[objectIndex]) {
        appliedDesired[objectIndex] = 1;
        appliedDesiredPrimitiveCount += primitiveContribution(objectIndex);
        if (objectIndex < pendingDesiredObjects.size())
          pendingDesiredObjects[objectIndex] = 0;
        if (objectIndex < _desiredObjectState.size())
          _desiredObjectState[objectIndex] = 1;
      }
      _frameProbabilisticToggles += applied;
      changed = true;
      if (!shouldBeActive && objectIndex < _objectExplorationScore.size())
        _objectExplorationScore[objectIndex] = 0.0f;
      if (!shouldBeActive && objectIndex < pendingDesiredObjects.size())
        pendingDesiredObjects[objectIndex] = 0;
      if (!shouldBeActive && objectIndex < _desiredObjectState.size())
        _desiredObjectState[objectIndex] = 0;
    }
  }

  size_t activePrimitiveCount = 0;
  for (bool active : _activePrimitive)
    if (active)
      ++activePrimitiveCount;

  if (activePrimitiveCount == 0 && objectCount > 0) {
    size_t fallback = fallbackObject < objectCount ? fallbackObject : size_t(0);
    if (fallback >= objectCount)
      fallback = objectCount - 1;
    size_t remainingBudget =
        (maxPrimitiveToggles > toggledPrimitiveCount)
            ? (maxPrimitiveToggles - toggledPrimitiveCount)
            : size_t(0);
    bool canToggleFallback = forceAllToggles || remainingBudget > 0;
    size_t toggled = canToggleFallback ? setObjectActive(fallback, true) : 0;
    if (toggled > 0) {
      size_t applied = toggled;
      if (!forceAllToggles) {
        applied = std::min(toggled, remainingBudget);
        toggledPrimitiveCount += applied;
      }
      _frameProbabilisticToggles += applied;
      changed = true;
    }
    if (fallback < _desiredObjectState.size()) {
      _desiredObjectState[fallback] = 1;
      if (fallback < _desiredObjectPromotionFrame.size())
        _desiredObjectPromotionFrame[fallback] = _renderedFrameCount;
    }
  }

  _frameProbabilityFinalDesiredPrimitives = appliedDesiredPrimitiveCount;

  return changed;
}
bool Renderer::updateScreenSpaceFootprint(bool forceAllToggles) {
  if (_activePrimitive.empty())
    return false;

  const size_t primCount = _activePrimitive.size();
  if (_primitiveScreenCoverage.size() != primCount)
    _primitiveScreenCoverage.assign(primCount, 0.0f);
  if (_screenCoverageSortedIndices.size() != primCount) {
    _screenCoverageSortedIndices.resize(primCount);
    std::iota(_screenCoverageSortedIndices.begin(),
              _screenCoverageSortedIndices.end(), size_t(0));
  }

  float screenArea = Camera::screenSize.x * Camera::screenSize.y;
  if (screenArea <= 0.0f)
    screenArea = 1.0f;
  float halfFov = Camera::verticalFov * static_cast<float>(M_PI) / 180.0f * 0.5f;
  float tanHalfFov = std::tan(halfFov);
  if (tanHalfFov <= 0.0f)
    tanHalfFov = 1e-3f;

  simd::float3 forward = simd::normalize(Camera::forward);
  simd::float3 up = simd::normalize(Camera::up);
  simd::float3 right = simd::cross(forward, up);
  float rightLenSq = simd::length_squared(right);
  if (rightLenSq < 1e-6f)
    right = {1.0f, 0.0f, 0.0f};
  else
    right /= std::sqrt(rightLenSq);

  float aspect = Camera::screenSize.y > 0.0f
                     ? Camera::screenSize.x / Camera::screenSize.y
                     : 1.0f;
  float horizontalHalfFov = std::atan(tanHalfFov * aspect);

  parallelChunkedAsync(0, primCount, [this, screenArea, forward, right,
                                      horizontalHalfFov,
                                      tanHalfFov](size_t chunkBegin,
                                                 size_t chunkEnd) {
    for (size_t i = chunkBegin; i < chunkEnd; ++i) {
      float coverage = 0.0f;
      if (i < _primitiveBounds.size() && isInView(_primitiveBounds[i])) {
        const BoundingSphere &b = _primitiveBounds[i];
        simd::float3 toCenter = b.center - Camera::position;
        float depth = simd::dot(toCenter, forward);
        if (depth > 1e-3f) {
          float dist = simd::length(toCenter);
          float cosAngle = depth / std::max(dist, 1e-3f);
          float horiz = simd::dot(toCenter, right);
          float horizAngle = std::atan2(std::fabs(horiz), depth);
          if (horizAngle <= horizontalHalfFov + 0.1f) {
            float radiusPixels = (b.radius / depth) / tanHalfFov *
                                 (Camera::screenSize.y * 0.5f);
            radiusPixels = std::max(radiusPixels, 0.0f);
            float area = static_cast<float>(M_PI) * radiusPixels * radiusPixels;
            float angleFactor = std::max(cosAngle, 0.0f);
            coverage = std::min(area * angleFactor, screenArea);
          }
        }
      }
      _primitiveScreenCoverage[i] = coverage;
    }
  });

  const size_t objectCount = _allSceneObjects.size();
  if (objectCount == 0)
    return false;

  std::vector<float> objectCoverage(objectCount, 0.0f);
  std::vector<size_t> objectPrimitiveContribution(objectCount, 0);
  for (size_t primIndex = 0; primIndex < primCount; ++primIndex) {
    size_t objectIndex =
        primIndex < _primitiveToObject.size() ? _primitiveToObject[primIndex]
                                              : std::numeric_limits<size_t>::max();
    if (objectIndex >= objectCount)
      continue;
    objectCoverage[objectIndex] += _primitiveScreenCoverage[primIndex];
    ++objectPrimitiveContribution[objectIndex];
  }

  std::vector<size_t> objectPrimitiveTotals(objectCount, 0);
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    size_t declaredPrimitiveCount =
        objectIndex < _allSceneObjects.size()
            ? _allSceneObjects[objectIndex].primitiveCount
            : size_t(0);
    if (declaredPrimitiveCount == 0)
      declaredPrimitiveCount = objectPrimitiveContribution[objectIndex];
    objectPrimitiveTotals[objectIndex] = declaredPrimitiveCount;
  }

  const auto &meshGroups = _meshGroups;
  std::vector<float> meshGroupCoverage(meshGroups.size(), 0.0f);
  std::vector<size_t> meshGroupPrimitiveCount(meshGroups.size(), 0);
  for (size_t groupIndex = 0; groupIndex < meshGroups.size(); ++groupIndex) {
    const MeshGroupInfo &info = meshGroups[groupIndex];
    float coverageSum = 0.0f;
    size_t fallbackCount = 0;
    for (size_t objectIndex : info.objectIndices) {
      if (objectIndex < objectCoverage.size())
        coverageSum += objectCoverage[objectIndex];
      if (objectIndex < objectPrimitiveTotals.size())
        fallbackCount += objectPrimitiveTotals[objectIndex];
    }
    meshGroupCoverage[groupIndex] = coverageSum;
    size_t declaredCount = info.primitiveCount;
    if (declaredCount == 0)
      declaredCount = fallbackCount;
    meshGroupPrimitiveCount[groupIndex] = declaredCount;
  }

  std::vector<size_t> sortedGroups(meshGroups.size());
  std::iota(sortedGroups.begin(), sortedGroups.end(), size_t(0));
  std::sort(sortedGroups.begin(), sortedGroups.end(),
            [&meshGroupCoverage](size_t a, size_t b) {
              float ca = (a < meshGroupCoverage.size())
                             ? sanitizeSortValue(meshGroupCoverage[a])
                             : 0.0f;
              float cb = (b < meshGroupCoverage.size())
                             ? sanitizeSortValue(meshGroupCoverage[b])
                             : 0.0f;
              if (ca == cb)
                return a < b;
              return ca > cb;
            });

  std::vector<bool> desiredGroupState(meshGroups.size(), false);
  const size_t minActivePrimitives = std::min(
      primCount, _residencyConfig.screenFootprintMinActivePrimitives);
  const float targetCoverage =
      screenArea * _residencyConfig.screenFootprintTargetFraction;
  size_t primitivesEnabled = 0;
  float accumulatedCoverage = 0.0f;

  for (size_t groupIndex : sortedGroups) {
    if (groupIndex >= meshGroups.size())
      continue;
    size_t declaredPrimitiveCount =
        (groupIndex < meshGroupPrimitiveCount.size())
            ? meshGroupPrimitiveCount[groupIndex]
            : size_t(0);
    if (declaredPrimitiveCount == 0)
      continue;

    float coverage =
        (groupIndex < meshGroupCoverage.size()) ? meshGroupCoverage[groupIndex]
                                                : 0.0f;
    bool minPrimitivesSatisfied = primitivesEnabled >= minActivePrimitives;
    bool coverageSatisfied = accumulatedCoverage >= targetCoverage;
    if (minPrimitivesSatisfied && coverageSatisfied)
      break;
    if (minPrimitivesSatisfied &&
        coverage < _residencyConfig.screenFootprintMinPixelCoverage)
      break;
    if (minPrimitivesSatisfied && coverage <= 0.0f)
      break;

    desiredGroupState[groupIndex] = true;
    primitivesEnabled += declaredPrimitiveCount;
    accumulatedCoverage += coverage;
  }

  for (size_t groupIndex : sortedGroups) {
    if (primitivesEnabled >= minActivePrimitives)
      break;
    if (groupIndex >= meshGroups.size())
      continue;
    if (desiredGroupState[groupIndex])
      continue;
    size_t declaredPrimitiveCount =
        (groupIndex < meshGroupPrimitiveCount.size())
            ? meshGroupPrimitiveCount[groupIndex]
            : size_t(0);
    if (declaredPrimitiveCount == 0)
      continue;
    desiredGroupState[groupIndex] = true;
    primitivesEnabled += declaredPrimitiveCount;
    accumulatedCoverage += (groupIndex < meshGroupCoverage.size())
                               ? meshGroupCoverage[groupIndex]
                               : 0.0f;
  }

  std::vector<bool> desiredObjectState(objectCount, false);
  for (size_t groupIndex = 0; groupIndex < meshGroups.size(); ++groupIndex) {
    bool groupDesired = desiredGroupState[groupIndex];
    for (size_t objectIndex : meshGroups[groupIndex].objectIndices) {
      if (objectIndex < desiredObjectState.size())
        desiredObjectState[objectIndex] = groupDesired;
    }
  }

  bool changed = false;
  size_t toggledPrimitiveCount = 0;
  for (size_t groupIndex : sortedGroups) {
    if (groupIndex >= meshGroups.size())
      continue;
    if (!forceAllToggles &&
        toggledPrimitiveCount >=
            _residencyConfig.screenFootprintMaxTogglesPerFrame)
      break;

    bool groupDesired = desiredGroupState[groupIndex];
    const auto &objectIndices = meshGroups[groupIndex].objectIndices;
    std::vector<size_t> objectsToToggle;
    objectsToToggle.reserve(objectIndices.size());
    bool canToggleGroup = true;

    for (size_t objectIndex : objectIndices) {
      if (objectIndex >= desiredObjectState.size())
        continue;
      bool shouldBeActive = desiredObjectState[objectIndex];
      bool currentlyActive =
          objectIndex < _objectActive.size() && _objectActive[objectIndex];
      if (shouldBeActive == currentlyActive)
        continue;

      if (!forceAllToggles && objectIndex < _objectCooldown.size() &&
          _objectCooldown[objectIndex] > 0) {
        canToggleGroup = false;
        break;
      }

      const SceneObject &obj = _allSceneObjects[objectIndex];
      size_t first = obj.firstPrimitive;
      size_t last = first + obj.primitiveCount;
      for (size_t prim = first; prim < last && prim < _activePrimitive.size();
           ++prim) {
        if (_activePrimitive[prim] == shouldBeActive)
          continue;
        if (!forceAllToggles && prim < _primitiveCooldown.size() &&
            _primitiveCooldown[prim] > 0) {
          canToggleGroup = false;
          break;
        }
      }

      if (!canToggleGroup)
        break;

      objectsToToggle.push_back(objectIndex);
    }

    if (!canToggleGroup || objectsToToggle.empty())
      continue;

    for (size_t objectIndex : objectsToToggle) {
      size_t toggled = setObjectActive(objectIndex, groupDesired);
      if (toggled > 0) {
        toggledPrimitiveCount = std::min(
            toggledPrimitiveCount + toggled,
            _residencyConfig.screenFootprintMaxTogglesPerFrame);
        changed = true;
      }
    }
  }

  bool anyActivePrimitive = false;
  for (bool active : _activePrimitive) {
    if (active) {
      anyActivePrimitive = true;
      break;
    }
  }

  if (!anyActivePrimitive) {
    size_t fallbackPrimitives = 0;
    for (size_t groupIndex : sortedGroups) {
      if (groupIndex >= meshGroups.size())
        continue;
      size_t declaredPrimitiveCount =
          (groupIndex < meshGroupPrimitiveCount.size())
              ? meshGroupPrimitiveCount[groupIndex]
              : size_t(0);
      if (declaredPrimitiveCount == 0)
        continue;

      bool groupActivated = false;
      for (size_t objectIndex : meshGroups[groupIndex].objectIndices) {
        if (objectIndex >= _allSceneObjects.size())
          continue;
        if (setObjectActive(objectIndex, true) > 0) {
          changed = true;
          groupActivated = true;
        }
      }

      if (groupActivated) {
        fallbackPrimitives += declaredPrimitiveCount;
        anyActivePrimitive = true;
      }

      if (anyActivePrimitive && fallbackPrimitives >= minActivePrimitives)
        break;
    }
  }

  return changed;
}

bool Renderer::updateEnvironmentHitResidency(bool forceAllToggles) {
  size_t objectCount = _allSceneObjects.size();
  size_t primitiveCount = _activePrimitive.size();
  if (objectCount == 0 || primitiveCount == 0)
    return false;

  if (_objectProbabilitySortedIndices.size() != objectCount) {
    _objectProbabilitySortedIndices.resize(objectCount);
    std::iota(_objectProbabilitySortedIndices.begin(),
              _objectProbabilitySortedIndices.end(), size_t(0));
  }

  if (_objectImportance.size() != objectCount)
    _objectImportance.assign(objectCount, 0.0f);
  else
    std::fill(_objectImportance.begin(), _objectImportance.end(), 0.0f);

  std::vector<uint8_t> desiredObjectState(objectCount, 0);
  std::vector<float> weightedImportance(objectCount, 0.0f);

  auto computeDepthWeight = [&](size_t objectIndex) -> float {
    const auto &weights = _residencyConfig.environmentDepthWeights;
    if (weights.empty())
      return 1.0f;
    const auto &radii = _residencyConfig.environmentDepthRadii;
    size_t pairCount = std::min(weights.size(), radii.size());
    if (pairCount == 0)
      return weights.back();
    float fallbackWeight = weights[pairCount - 1];
    if (objectIndex >= _objectBounds.size())
      return fallbackWeight;

    simd::float3 toCenter = _objectBounds[objectIndex].center - Camera::position;
    float distance = simd::length(toCenter);
    if (!(distance > 0.0f))
      return weights.front();

    auto end = radii.begin() + pairCount;
    auto lowerBound = std::lower_bound(radii.begin(), end, distance);
    if (lowerBound != end)
      return weights[std::distance(radii.begin(), lowerBound)];
    return fallbackWeight;
  };

  constexpr float kExplorationPriorFloor = 1.0e-4f;
  constexpr float kExplorationPriorScale = 0.5f;

  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    float hitProbability =
        (objectIndex < _objectHitProbability.size())
            ? std::clamp(_objectHitProbability[objectIndex], 0.0f, 1.0f)
            : 0.0f;
    bool hasEvidence =
        (objectIndex < _objectRaysTestedLastFrame.size())
            ? (_objectRaysTestedLastFrame[objectIndex] > 0)
            : false;
    bool currentlyActive =
        (objectIndex < _objectActive.size()) ? _objectActive[objectIndex] : false;

    if (!hasEvidence && !currentlyActive) {
      float exploration =
          (objectIndex < _objectExplorationScore.size())
              ? std::max(_objectExplorationScore[objectIndex], 0.0f)
              : 0.0f;
      float visibilityHint =
          (objectIndex < _objectVisible.size()) ? _objectVisible[objectIndex] : 0u;
      float explorationPrior = 1.0f -
                               std::exp(-exploration * kExplorationPriorScale);
      if (visibilityHint) {
        float hintedPrior = 1.0f -
                            std::exp(-kIdleVisibleExploreSeed *
                                     kExplorationPriorScale);
        explorationPrior = std::max(explorationPrior, hintedPrior);
      }
      explorationPrior =
          std::clamp(explorationPrior, kExplorationPriorFloor, 1.0f);
      hitProbability = std::max(hitProbability, explorationPrior);
    }

    _objectImportance[objectIndex] = hitProbability;

    float depthWeight = computeDepthWeight(objectIndex);
    float weighted = hitProbability * std::max(depthWeight, 0.0f);
    weightedImportance[objectIndex] = std::isfinite(weighted) ? weighted : 0.0f;
  }

  auto &sortedIndices = _objectProbabilitySortedIndices;
  std::sort(sortedIndices.begin(), sortedIndices.end(),
            [&](size_t a, size_t b) {
              float scoreA =
                  (a < weightedImportance.size()) ? weightedImportance[a] : 0.0f;
              float scoreB =
                  (b < weightedImportance.size()) ? weightedImportance[b] : 0.0f;
              if (scoreA == scoreB) {
                float importanceA =
                    (a < _objectImportance.size()) ? _objectImportance[a] : 0.0f;
                float importanceB =
                    (b < _objectImportance.size()) ? _objectImportance[b] : 0.0f;
                if (importanceA == importanceB)
                  return a < b;
                return importanceA > importanceB;
              }
              return scoreA > scoreB;
            });

  float escapeThreshold =
      std::clamp(_residencyConfig.environmentEscapeThreshold, 0.0f, 1.0f);
  size_t minActivePrimitives =
      std::min(primitiveCount, _residencyConfig.environmentMinActivePrimitives);
  if (minActivePrimitives == 0 && primitiveCount > 0)
    minActivePrimitives = 1;

  float targetFraction =
      std::clamp(_residencyConfig.environmentTargetActiveFraction, 0.0f, 1.0f);
  size_t targetPrimitiveCount = 0;
  if (targetFraction > 0.0f && primitiveCount > 0) {
    targetPrimitiveCount = static_cast<size_t>(std::ceil(
        targetFraction * static_cast<float>(primitiveCount)));
    targetPrimitiveCount = std::min(targetPrimitiveCount, primitiveCount);
  }
  size_t activationFloor = std::max(minActivePrimitives, targetPrimitiveCount);
  if (activationFloor == 0 && primitiveCount > 0)
    activationFloor = 1;

  size_t desiredPrimitiveCount = 0;
  float weightedEscape = 0.0f;
  auto averageEscape = [](size_t primCount, float weighted) {
    if (primCount == 0)
      return 1.0f;
    return weighted / static_cast<float>(primCount);
  };

  for (size_t objectIndex : sortedIndices) {
    if (objectIndex >= _allSceneObjects.size())
      continue;
    const SceneObject &object = _allSceneObjects[objectIndex];
    if (object.primitiveCount == 0)
      continue;

    bool needMorePrimitives = desiredPrimitiveCount < activationFloor;
    bool needsEscapeReduction =
        averageEscape(desiredPrimitiveCount, weightedEscape) > escapeThreshold;
    if (!needMorePrimitives && !needsEscapeReduction)
      break;

    desiredObjectState[objectIndex] = 1;
    desiredPrimitiveCount += object.primitiveCount;
    float escapeProbability =
        1.0f - ((objectIndex < _objectImportance.size())
                    ? _objectImportance[objectIndex]
                    : 0.0f);
    weightedEscape += escapeProbability * static_cast<float>(object.primitiveCount);
  }

  if (desiredPrimitiveCount == 0 && primitiveCount > 0) {
    for (size_t objectIndex : sortedIndices) {
      if (objectIndex >= _allSceneObjects.size())
        continue;
      const SceneObject &object = _allSceneObjects[objectIndex];
      if (object.primitiveCount == 0)
        continue;
      desiredObjectState[objectIndex] = 1;
      desiredPrimitiveCount = object.primitiveCount;
      float escapeProbability =
          1.0f - ((objectIndex < _objectImportance.size())
                      ? _objectImportance[objectIndex]
                      : 0.0f);
      weightedEscape =
          escapeProbability * static_cast<float>(object.primitiveCount);
      break;
    }
  }

  bool changed = false;
  size_t toggledPrimitiveCount = 0;
  size_t maxToggleBudget = _residencyConfig.environmentMaxTogglesPerFrame;

  for (size_t objectIndex : sortedIndices) {
    bool shouldBeActive =
        (objectIndex < desiredObjectState.size()) ? desiredObjectState[objectIndex]
                                                  : 0;
    bool currentlyActive =
        (objectIndex < _objectActive.size()) ? _objectActive[objectIndex] : false;
    if (shouldBeActive == currentlyActive)
      continue;

    if (!forceAllToggles && objectIndex < _objectCooldown.size() &&
        _objectCooldown[objectIndex] > 0)
      continue;

    const SceneObject &object = _allSceneObjects[objectIndex];
    size_t first = object.firstPrimitive;
    size_t last = std::min(first + object.primitiveCount, _activePrimitive.size());
    if (last <= first)
      continue;

    bool canToggle = true;
    size_t pendingToggles = 0;
    for (size_t prim = first; prim < last; ++prim) {
      if (_activePrimitive[prim] == shouldBeActive)
        continue;
      if (!forceAllToggles && prim < _primitiveCooldown.size() &&
          _primitiveCooldown[prim] > 0) {
        canToggle = false;
        break;
      }
      ++pendingToggles;
    }

    if (!canToggle || pendingToggles == 0)
      continue;

    if (!forceAllToggles && toggledPrimitiveCount >= maxToggleBudget)
      continue;

    size_t toggled = setObjectActive(objectIndex, shouldBeActive);
    if (toggled > 0) {
      changed = true;
      if (!forceAllToggles) {
        toggledPrimitiveCount =
            std::min(toggledPrimitiveCount + toggled, maxToggleBudget);
      }
    }
  }

  bool anyActiveObject = false;
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    if (objectIndex >= _objectActive.size())
      continue;
    if (objectIndex >= _objectPrimitiveCounts.size())
      continue;
    if (_objectPrimitiveCounts[objectIndex] == 0)
      continue;
    if (_objectActive[objectIndex]) {
      anyActiveObject = true;
      break;
    }
  }

  if (!anyActiveObject) {
    for (size_t objectIndex : sortedIndices) {
      if (objectIndex >= _objectPrimitiveCounts.size())
        continue;
      if (_objectPrimitiveCounts[objectIndex] == 0)
        continue;
      if (setObjectActive(objectIndex, true) > 0) {
        changed = true;
        anyActiveObject = true;
      }
      if (anyActiveObject)
        break;
    }
  }

  if (!anyActiveObject && !_activePrimitive.empty()) {
    if (setPrimitiveActive(0, true))
      changed = true;
  }

  return changed;
}

// Propagate any pending primitive/object toggles to GPU memory.  The method
// skips expensive repacks if nothing changed, otherwise it forwards to
// rebuildResidentResources to patch only the dirty ranges (or rebuild
// everything when forced).
void Renderer::flushResidencyChanges(bool forceFullRebuild) {
  if (!forceFullRebuild && _recentlyActivated.empty() &&
      _recentlyDeactivated.empty() && _dirtyResidentObjects.empty()) {
    for (size_t objectIndex = 0;
         objectIndex < _residentObjectGpuResources.size(); ++objectIndex) {
      auto &resident = _residentObjectGpuResources[objectIndex];
      auto &record = _instanceRecords[objectIndex];
      resident.clearPendingCommand();
      bool pending = resident.hasPendingCommands();

      if (_frameStrategy != ResidencyStrategy::AlwaysResident &&
          !resident.isResident() && !pending) {
        transitionResidentToCold(objectIndex, resident, record);
      }
    }
    return;
  }
  rebuildResidentResources(forceFullRebuild);
}

void Renderer::drawableSizeWillChange(MTK::View *pView, CGSize size) {
  Camera::screenSize = {(float)size.width, (float)size.height};

  buildTextures();
  recalculateViewport();
}

bool Renderer::hasKeyframes() const { return !_pScene->cameraPath.empty(); }

bool Renderer::setPrimitiveActive(size_t index, bool active) {
  if (index >= _activePrimitive.size())
    return false;
  if (_activePrimitive[index] == active)
    return false;
  _activePrimitive[index] = active;
  if (active)
    ++_framePrimitiveActivations;
  else
    ++_framePrimitiveDeactivations;
  if (!active)
    _alwaysResidentCache.markDirty();
  auto &cancelList = active ? _recentlyDeactivated : _recentlyActivated;
  cancelList.erase(
      std::remove(cancelList.begin(), cancelList.end(), index),
      cancelList.end());
  if (active)
    _recentlyActivated.push_back(index);
  else
    _recentlyDeactivated.push_back(index);
  if (index < _primitiveCooldown.size())
    _primitiveCooldown[index] = _residencyConfig.stateCooldownFrames;

  if (index < _primitiveToObject.size()) {
    size_t objectIndex = _primitiveToObject[index];
    if (objectIndex < _allSceneObjects.size()) {
      if (objectIndex >= _objectActive.size())
        _objectActive.resize(objectIndex + 1, false);
      if (objectIndex >= _objectCooldown.size())
        _objectCooldown.resize(objectIndex + 1, 0);
      if (objectIndex >= _objectActivePrimitiveCounts.size())
        _objectActivePrimitiveCounts.resize(objectIndex + 1, 0);

      size_t &activeCount = _objectActivePrimitiveCounts[objectIndex];
      if (active)
        ++activeCount;
      else if (activeCount > 0)
        --activeCount;

      bool newState = activeCount > 0;
      bool fullyInactive = activeCount == 0;
      bool prevState = _objectActive[objectIndex];
      _objectActive[objectIndex] = newState;
      if (prevState != newState || fullyInactive)
        _objectCooldown[objectIndex] = _residencyConfig.stateCooldownFrames;

      _dirtyResidentObjects.push_back(objectIndex);
    }
  }
  return true;
}

void Renderer::dumpAccelerationStructure(const std::string &path) {
  std::filesystem::create_directories(
      std::filesystem::path(path).parent_path());
  std::ofstream out(path);
  if (!out.is_open())
    return;

  out << "{\n";

  size_t tlasCount = 0;
  simd::float4 *tlasData = _pScene->createTLASBuffer(tlasCount);
  out << "  \"tlas\": [\n";
  for (size_t i = 0; i < tlasCount; ++i) {
    simd::float4 bmin = tlasData[2 * i];
    simd::float4 bmax = tlasData[2 * i + 1];
    int first = reinterpret_cast<const int *>(&bmin)[3];
    int second = reinterpret_cast<const int *>(&bmax)[3];
    bool isLeaf = first < 0;
    out << "    {\"index\":" << i << ",\"leaf\":"
        << (isLeaf ? "true" : "false") << ",\"min\":[" << bmin.x << ","
        << bmin.y << "," << bmin.z << "],\"max\":[" << bmax.x << ","
        << bmax.y << "," << bmax.z << "]";
    if (isLeaf) {
      int objectIndex = -(first + 1);
      int blasRoot = -1;
      if (objectIndex >= 0 &&
          static_cast<size_t>(objectIndex) < _instanceRecords.size())
        blasRoot = _instanceRecords[objectIndex].blasRootIndex;
      out << ",\"object\":" << objectIndex << ",\"instance\":"
          << second << ",\"blasRoot\":" << blasRoot;
    } else {
      out << ",\"left\":" << first << ",\"right\":" << second;
    }
    out << "}";
    if (i + 1 < tlasCount)
      out << ",\n";
    else
      out << "\n";
  }
  out << "  ],\n";
  delete[] tlasData;

  const auto &nodes = _pScene->getBVHNodes();
  out << "  \"blas\": [\n";
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto &n = nodes[i];
    out << "    {\"index\":" << i << ",\"min\":[" << n.boundsMin.x << ","
        << n.boundsMin.y << "," << n.boundsMin.z << "],\"max\":["
        << n.boundsMax.x << "," << n.boundsMax.y << "," << n.boundsMax.z
        << "],\"leftFirst\":" << n.leftFirst << ",\"count\":" << n.count << "}";
    if (i + 1 < nodes.size())
      out << ",\n";
    else
      out << "\n";
  }
  out << "  ],\n";

  out << "  \"primitives\": [\n";
  size_t primCount = std::min(_allPrimitives.size(), _activePrimitive.size());
  for (size_t i = 0; i < primCount; ++i) {
    out << "    {\"index\":" << i
        << ",\"active\":" << (_activePrimitive[i] ? "true" : "false")
        << "}";
    if (i + 1 < primCount)
      out << ",\n";
    else
      out << "\n";
  }
  out << "  ]\n";

  out << "}\n";
}

double Renderer::currentGPUMemoryMB() const {
  return static_cast<double>(_pDevice->currentAllocatedSize()) /
         (1024.0 * 1024.0);
}

bool Renderer::rayHitCopyReady() const {
  if (!_lastRayHitCommandBuffer)
    return true;

  auto status = _lastRayHitCommandBuffer->status();
  return status == MTL::CommandBufferStatus::CommandBufferStatusCompleted ||
         status == MTL::CommandBufferStatus::CommandBufferStatusError;
}

bool Renderer::flushRayHitCopy() {
  if (!_lastRayHitCommandBuffer)
    return true;

  auto status = _lastRayHitCommandBuffer->status();

  switch (status) {
  case MTL::CommandBufferStatus::CommandBufferStatusCompleted:
    _rayHitCopyError = false;
    break;
  case MTL::CommandBufferStatus::CommandBufferStatusError:
    _rayHitCopyError = true;
    break;
  case MTL::CommandBufferStatus::CommandBufferStatusCommitted:
  case MTL::CommandBufferStatus::CommandBufferStatusScheduled:
  case MTL::CommandBufferStatus::CommandBufferStatusEnqueued:
  case MTL::CommandBufferStatus::CommandBufferStatusNotEnqueued:
  default:
    return false;
  }

  _lastRayHitCommandBuffer->release();
  _lastRayHitCommandBuffer = nullptr;
  return true;
}

void Renderer::processRayHitCounters() {
  if (!flushRayHitCopy())
    return;

  if (!_pPrimitiveHitReadback) {
    _primitiveHitScores.clear();
    _primitiveHitLastFrame.clear();
    _primitiveRayContributions.clear();
    _primitiveRaysTestedLastFrame.clear();
    _primitiveHitAlpha.clear();
    _primitiveHitBeta.clear();
    _primitiveHitProbability.clear();
    _primitiveHitVariance.clear();
    _primitivePosteriorMass.clear();
    _primitiveExplorationScore.clear();
    _probabilitySortedIndices.clear();
    _objectHitAlpha.clear();
    _objectHitBeta.clear();
    _objectHitProbability.clear();
    _objectHitVariance.clear();
    _objectPosteriorMass.clear();
    _objectExplorationScore.clear();
    _objectHitLastFrame.clear();
    _objectRaysTestedLastFrame.clear();
    _objectVisible.clear();
    _objectVisibilityEvidence.clear();
    _objectProbabilitySortedIndices.clear();
    return;
  }

  size_t bufferLength = _pPrimitiveHitReadback->length();
  uint32_t *hitPtr =
      static_cast<uint32_t *>(_pPrimitiveHitReadback->contents());
  if (!hitPtr) {
    _primitiveHitScores.clear();
    _primitiveHitLastFrame.clear();
    _primitiveRayContributions.clear();
    _primitiveRaysTestedLastFrame.clear();
    _primitiveHitAlpha.clear();
    _primitiveHitBeta.clear();
    _primitiveHitProbability.clear();
    _primitiveHitVariance.clear();
    _primitivePosteriorMass.clear();
    _primitiveExplorationScore.clear();
    _probabilitySortedIndices.clear();
    _objectHitAlpha.clear();
    _objectHitBeta.clear();
    _objectHitProbability.clear();
    _objectHitVariance.clear();
    _objectPosteriorMass.clear();
    _objectExplorationScore.clear();
    _objectHitLastFrame.clear();
    _objectRaysTestedLastFrame.clear();
    _objectVisible.clear();
    _objectVisibilityEvidence.clear();
    _objectProbabilitySortedIndices.clear();
    return;
  }

  if (_rayHitCopyError) {
    std::memset(hitPtr, 0, bufferLength);
    _primitiveHitScores.clear();
    _primitiveHitLastFrame.clear();
    _primitiveRayContributions.clear();
    _primitiveRaysTestedLastFrame.clear();
    _primitiveHitAlpha.clear();
    _primitiveHitBeta.clear();
    _primitiveHitProbability.clear();
    _primitiveHitVariance.clear();
    _primitivePosteriorMass.clear();
    _primitiveExplorationScore.clear();
    _probabilitySortedIndices.clear();
    _rayHitCopyError = false;
    _objectHitAlpha.clear();
    _objectHitBeta.clear();
    _objectHitProbability.clear();
    _objectHitVariance.clear();
    _objectPosteriorMass.clear();
    _objectExplorationScore.clear();
    _objectHitLastFrame.clear();
    _objectRaysTestedLastFrame.clear();
    _objectVisible.clear();
    _objectVisibilityEvidence.clear();
    _objectProbabilitySortedIndices.clear();
    return;
  }

  ResidencyStrategy strategy =
      _pScene ? _pScene->getResidencyStrategy()
              : ResidencyStrategy::DistanceLOD;
  bool strategyUsesHits =
      strategy == ResidencyStrategy::RayHitBudget ||
      strategy == ResidencyStrategy::Probabilistic ||
      strategy == ResidencyStrategy::EnvironmentHit ||
      strategy == ResidencyStrategy::UnifiedScore;
  if (!strategyUsesHits) {
    std::memset(hitPtr, 0, bufferLength);
    _primitiveHitScores.clear();
    _primitiveHitLastFrame.clear();
    _primitiveRayContributions.clear();
    _primitiveRaysTestedLastFrame.clear();
    _primitiveHitAlpha.clear();
    _primitiveHitBeta.clear();
    _primitiveHitProbability.clear();
    _primitiveHitVariance.clear();
    _primitivePosteriorMass.clear();
    _primitiveExplorationScore.clear();
    _probabilitySortedIndices.clear();
    _objectHitAlpha.clear();
    _objectHitBeta.clear();
    _objectHitProbability.clear();
    _objectHitVariance.clear();
    _objectPosteriorMass.clear();
    _objectExplorationScore.clear();
    _objectHitLastFrame.clear();
    _objectRaysTestedLastFrame.clear();
    _objectVisible.clear();
    _objectVisibilityEvidence.clear();
    _objectProbabilitySortedIndices.clear();
    return;
  }

  size_t totalPrimitiveCount = _allPrimitives.size();
  if (totalPrimitiveCount == 0)
    return;

  constexpr size_t kStatsPerPrimitive = 2;
  size_t bufferCount = bufferLength / sizeof(uint32_t);
  size_t count = std::min(totalPrimitiveCount, bufferCount / kStatsPerPrimitive);
  if (count == 0)
    return;

  if (_primitiveHitScores.size() < totalPrimitiveCount)
    _primitiveHitScores.resize(totalPrimitiveCount, 0.0f);
  if (_primitiveHitLastFrame.size() < totalPrimitiveCount)
    _primitiveHitLastFrame.resize(totalPrimitiveCount, 0);
  if (_primitiveRayContributions.size() < totalPrimitiveCount)
    _primitiveRayContributions.resize(totalPrimitiveCount, 0.0f);
  if (_primitiveRaysTestedLastFrame.size() < totalPrimitiveCount)
    _primitiveRaysTestedLastFrame.resize(totalPrimitiveCount, 0);
  if (_primitiveHitAlpha.size() < totalPrimitiveCount)
    _primitiveHitAlpha.resize(totalPrimitiveCount, 1.0f);
  if (_primitiveHitBeta.size() < totalPrimitiveCount)
    _primitiveHitBeta.resize(totalPrimitiveCount, 1.0f);
  if (_primitiveHitProbability.size() < totalPrimitiveCount)
    _primitiveHitProbability.resize(totalPrimitiveCount, 0.5f);
  if (_primitiveHitVariance.size() < totalPrimitiveCount)
    _primitiveHitVariance.resize(totalPrimitiveCount, 1.0f / 12.0f);
  if (_primitivePosteriorMass.size() < totalPrimitiveCount)
    _primitivePosteriorMass.resize(totalPrimitiveCount, 2.0f);
  if (_primitiveExplorationScore.size() < totalPrimitiveCount)
    _primitiveExplorationScore.resize(totalPrimitiveCount, 0.0f);
  if (_primitiveIdleFrames.size() < totalPrimitiveCount)
    _primitiveIdleFrames.resize(totalPrimitiveCount, 0u);
  if (_probabilitySortedIndices.size() != totalPrimitiveCount) {
    _probabilitySortedIndices.resize(totalPrimitiveCount);
    std::iota(_probabilitySortedIndices.begin(), _probabilitySortedIndices.end(),
              size_t(0));
  }

  float rayHitDecay = _residencyConfig.rayHitDecay;
  float probabilityDecay = _residencyConfig.probabilityDecay;
  float probabilityThreshold = _residencyConfig.probabilityThreshold;
  constexpr float kMinPosteriorMass = 1.0e-3f;
  const float configuredWindow = _residencyConfig.probabilityEvidenceWindow;
  const float maxPosteriorMass =
      (configuredWindow > 0.0f && std::isfinite(configuredWindow))
          ? std::max(configuredWindow, kMinPosteriorMass)
          : std::numeric_limits<float>::max();
  const bool clampPosteriorMass =
      maxPosteriorMass < std::numeric_limits<float>::max();
  const uint32_t idleCooldownFrames =
      _residencyConfig.probabilityIdleCooldownFrames;
  const float idleGraceDecay =
      std::clamp(_residencyConfig.probabilityIdleDecay, 0.0f, 1.0f);
  auto renormalizePosterior = [&](float &alpha, float &beta) {
    float sum = alpha + beta;
    if (!(sum > 0.0f)) {
      alpha = beta = kMinPosteriorMass * 0.5f;
      sum = kMinPosteriorMass;
    } else if (sum < kMinPosteriorMass) {
      float scale = kMinPosteriorMass / sum;
      alpha *= scale;
      beta *= scale;
      sum = kMinPosteriorMass;
    }
    if (clampPosteriorMass && sum > maxPosteriorMass) {
      float scale = maxPosteriorMass / sum;
      alpha *= scale;
      beta *= scale;
      sum = maxPosteriorMass;
    }
    return sum;
  };
  parallelChunkedAsync(0, count, [&](size_t chunkStart, size_t chunkEnd) {
    for (size_t i = chunkStart; i < chunkEnd; ++i) {
      size_t base = i * kStatsPerPrimitive;
      uint32_t hits = hitPtr[base + 0];
      uint32_t raysTested = hitPtr[base + 1];
      _primitiveHitLastFrame[i] = hits;
      _primitiveRaysTestedLastFrame[i] = raysTested;
      _primitiveHitScores[i] = _primitiveHitScores[i] * rayHitDecay +
                               static_cast<float>(hits);
      _primitiveRayContributions[i] =
          _primitiveRayContributions[i] * rayHitDecay +
          static_cast<float>(raysTested);

      float success = static_cast<float>(hits);
      float failure =
          std::max(static_cast<float>(raysTested) - success, 0.0f);
      float alpha = _primitiveHitAlpha[i] * probabilityDecay + success;
      float beta = _primitiveHitBeta[i] * probabilityDecay + failure;
      float sum = alpha + beta;
      float probability = (sum > 0.0f) ? (alpha / sum) : 0.5f;
      bool wasVisible =
          (i < _primitiveVisible.size()) ? (_primitiveVisible[i] != 0) : false;
      bool wasActive =
          (i < _activePrimitive.size()) ? (_activePrimitive[i] != 0) : false;
      uint32_t idleFrames =
          (i < _primitiveIdleFrames.size()) ? _primitiveIdleFrames[i] : 0u;
      bool idleGraceActive =
          (idleCooldownFrames > 0) && (wasVisible || wasActive) &&
          (idleFrames < idleCooldownFrames);
      if (raysTested == 0) {
        // When no rays were fired this frame we still decay the posterior so
        // idle primitives drift toward deactivation. Idle primitives that are
        // still visible or were recently active are given a grace period
        // before applying the full cooling decay. During the grace period the
        // decay is either skipped entirely or uses a milder factor to avoid
        // immediately dropping the exploration probability.
        float coolingFactor = probabilityDecay;
        bool applyCooling = true;
        if (idleGraceActive) {
          if (idleGraceDecay >= 0.999f)
            applyCooling = false;
          else
            coolingFactor = idleGraceDecay;
        }
        if (applyCooling) {
          float cooledProbability =
              std::clamp(probability * coolingFactor, 0.0f, 1.0f);
          float cooledMass = std::max(sum, kMinPosteriorMass);
          if (clampPosteriorMass && cooledMass > maxPosteriorMass)
            cooledMass = maxPosteriorMass;
          alpha = cooledProbability * cooledMass;
          beta = std::max(cooledMass - alpha, 0.0f);
        }
      }
      sum = renormalizePosterior(alpha, beta);
      probability = (sum > 0.0f) ? (alpha / sum) : 0.5f;
      float clampedProbability = std::clamp(probability, 0.0f, 1.0f);
      float variance = 0.0f;
      if (sum > 1.0f)
        variance = (alpha * beta) / ((sum * sum) * (sum + 1.0f));
      _primitiveHitAlpha[i] = alpha;
      _primitiveHitBeta[i] = beta;
      _primitiveHitProbability[i] = clampedProbability;
      _primitiveHitVariance[i] = std::max(variance, 0.0f);
      _primitivePosteriorMass[i] = sum;

      float exploration = _primitiveExplorationScore[i] * probabilityDecay;
      if (raysTested > 0)
        _primitiveIdleFrames[i] = 0u;
      else if (i < _primitiveIdleFrames.size()) {
        uint32_t next = idleFrames;
        if (next < std::numeric_limits<uint32_t>::max())
          ++next;
        _primitiveIdleFrames[i] = next;
      }
      if (raysTested > 0) {
        if (clampedProbability < probabilityThreshold)
          exploration += static_cast<float>(raysTested);
        else
          exploration *= rayHitDecay;
      } else {
        exploration *= rayHitDecay;
        if (wasVisible || wasActive)
          exploration = std::max(exploration, kIdleVisibleExploreSeed);
      }
      _primitiveExplorationScore[i] = exploration;
      hitPtr[base + 0] = 0;
      hitPtr[base + 1] = 0;
    }
  });

  if (count < totalPrimitiveCount) {
    parallelChunkedAsync(count, totalPrimitiveCount,
                         [&](size_t chunkStart, size_t chunkEnd) {
                           for (size_t i = chunkStart; i < chunkEnd; ++i) {
                             _primitiveHitScores[i] *= rayHitDecay;
                             _primitiveRayContributions[i] *= rayHitDecay;
                             _primitiveHitLastFrame[i] = 0;
                             _primitiveRaysTestedLastFrame[i] = 0;

                             float alpha = _primitiveHitAlpha[i] * probabilityDecay;
                             float beta = _primitiveHitBeta[i] * probabilityDecay;
                             float sum = alpha + beta;
                             float probability =
                                 (sum > 0.0f) ? (alpha / sum) : 0.5f;
                             float cooledProbability =
                                 std::clamp(probability * probabilityDecay, 0.0f, 1.0f);
                             float cooledMass = std::max(sum, kMinPosteriorMass);
                             if (clampPosteriorMass && cooledMass > maxPosteriorMass)
                               cooledMass = maxPosteriorMass;
                             alpha = cooledProbability * cooledMass;
                             beta = std::max(cooledMass - alpha, 0.0f);
                             float updatedSum = renormalizePosterior(alpha, beta);
                             float updatedProbability =
                                 (updatedSum > 0.0f) ? (alpha / updatedSum) : 0.5f;
                             float variance = 0.0f;
                             if (updatedSum > 1.0f)
                               variance =
                                   (alpha * beta) /
                                   ((updatedSum * updatedSum) *
                                    (updatedSum + 1.0f));
                             _primitiveHitAlpha[i] = alpha;
                             _primitiveHitBeta[i] = beta;
                             _primitiveHitProbability[i] =
                                 std::clamp(updatedProbability, 0.0f, 1.0f);
                             _primitiveHitVariance[i] = std::max(variance, 0.0f);
                             _primitivePosteriorMass[i] = updatedSum;

                             float exploration =
                                 _primitiveExplorationScore[i] * probabilityDecay;
                             exploration *= rayHitDecay;
                             bool wasVisible =
                                 (i < _primitiveVisible.size())
                                     ? (_primitiveVisible[i] != 0)
                                     : false;
                             if (wasVisible)
                               exploration =
                                   std::max(exploration, kIdleVisibleExploreSeed);
                             _primitiveExplorationScore[i] = exploration;
                           }
                         });
  }

  size_t objectCount = _allSceneObjects.size();
  if (objectCount == 0) {
    _objectHitAlpha.clear();
    _objectHitBeta.clear();
    _objectHitProbability.clear();
    _objectHitVariance.clear();
    _objectPosteriorMass.clear();
    _objectExplorationScore.clear();
    _objectHitLastFrame.clear();
    _objectRaysTestedLastFrame.clear();
    _objectVisible.clear();
    _objectProbabilitySortedIndices.clear();
    return;
  }

  if (_objectHitAlpha.size() < objectCount)
    _objectHitAlpha.resize(objectCount, 1.0f);
  if (_objectHitBeta.size() < objectCount)
    _objectHitBeta.resize(objectCount, 1.0f);
  if (_objectHitProbability.size() < objectCount)
    _objectHitProbability.resize(objectCount, 0.5f);
  if (_objectHitVariance.size() < objectCount)
    _objectHitVariance.resize(objectCount, 1.0f / 12.0f);
  if (_objectPosteriorMass.size() < objectCount)
    _objectPosteriorMass.resize(objectCount, 2.0f);
  if (_objectExplorationScore.size() < objectCount)
    _objectExplorationScore.resize(objectCount, 0.0f);
  if (_objectHitLastFrame.size() < objectCount)
    _objectHitLastFrame.resize(objectCount, 0u);
  if (_objectRaysTestedLastFrame.size() < objectCount)
    _objectRaysTestedLastFrame.resize(objectCount, 0u);
  if (_objectIdleFrames.size() < objectCount)
    _objectIdleFrames.resize(objectCount, 0u);
  if (_objectVisible.size() < objectCount)
    _objectVisible.resize(objectCount, 0u);
  if (_objectProbabilitySortedIndices.size() != objectCount) {
    _objectProbabilitySortedIndices.resize(objectCount);
    std::iota(_objectProbabilitySortedIndices.begin(),
              _objectProbabilitySortedIndices.end(), size_t(0));
  }

  std::vector<uint64_t> objectHits(objectCount, 0);
  std::vector<uint64_t> objectRays(objectCount, 0);
  std::vector<uint8_t> objectActiveFlags(objectCount, 0);
  size_t processedPrimitiveCount = std::min(count, totalPrimitiveCount);
  for (size_t i = 0; i < processedPrimitiveCount; ++i) {
    size_t objectIndex =
        (i < _primitiveToObject.size()) ? _primitiveToObject[i] : SIZE_MAX;
    if (objectIndex >= objectCount)
      continue;
    objectHits[objectIndex] += _primitiveHitLastFrame[i];
    objectRays[objectIndex] += _primitiveRaysTestedLastFrame[i];
  }
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    if (objectIndex >= _allSceneObjects.size())
      break;
    const SceneObject &object = _allSceneObjects[objectIndex];
    size_t start = object.firstPrimitive;
    size_t end = std::min(start + object.primitiveCount, _activePrimitive.size());
    bool active = false;
    for (size_t prim = start; prim < end; ++prim) {
      if (_activePrimitive[prim]) {
        active = true;
        break;
      }
    }
    objectActiveFlags[objectIndex] = active ? 1u : 0u;
  }

  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    float success = static_cast<float>(objectHits[objectIndex]);
    float raysTested = static_cast<float>(objectRays[objectIndex]);
    _objectHitLastFrame[objectIndex] = static_cast<uint32_t>(success);
    _objectRaysTestedLastFrame[objectIndex] =
        static_cast<uint32_t>(raysTested);

    float failure = std::max(raysTested - success, 0.0f);
    float alpha = _objectHitAlpha[objectIndex] * probabilityDecay + success;
    float beta = _objectHitBeta[objectIndex] * probabilityDecay + failure;
    float sum = alpha + beta;
    float probability = (sum > 0.0f) ? (alpha / sum) : 0.5f;
    uint32_t idleFrames =
        (objectIndex < _objectIdleFrames.size()) ? _objectIdleFrames[objectIndex]
                                                 : 0u;
    bool hasActivePrimitive =
        (objectIndex < objectActiveFlags.size()) &&
        (objectActiveFlags[objectIndex] != 0);
    bool visible =
        (objectIndex < _objectBounds.size()) ? isInView(_objectBounds[objectIndex])
                                             : false;
    if (objectIndex < _objectVisible.size())
      _objectVisible[objectIndex] = visible ? 1 : 0;
    if (raysTested == 0.0f) {
      bool idleGraceActive =
          (idleCooldownFrames > 0) && (visible || hasActivePrimitive) &&
          (idleFrames < idleCooldownFrames);
      float coolingFactor = probabilityDecay;
      bool applyCooling = true;
      if (idleGraceActive) {
        if (idleGraceDecay >= 0.999f)
          applyCooling = false;
        else
          coolingFactor = idleGraceDecay;
      }
      if (applyCooling) {
        float cooledProbability =
            std::clamp(probability * coolingFactor, 0.0f, 1.0f);
        float cooledMass = std::max(sum, kMinPosteriorMass);
        if (clampPosteriorMass && cooledMass > maxPosteriorMass)
          cooledMass = maxPosteriorMass;
        alpha = cooledProbability * cooledMass;
        beta = std::max(cooledMass - alpha, 0.0f);
      }
    }

    sum = renormalizePosterior(alpha, beta);
    probability = (sum > 0.0f) ? (alpha / sum) : 0.5f;
    float clampedProbability = std::clamp(probability, 0.0f, 1.0f);
    float variance = 0.0f;
    if (sum > 1.0f)
      variance = (alpha * beta) / ((sum * sum) * (sum + 1.0f));
    _objectHitAlpha[objectIndex] = alpha;
    _objectHitBeta[objectIndex] = beta;
    _objectHitProbability[objectIndex] = clampedProbability;
    _objectHitVariance[objectIndex] = std::max(variance, 0.0f);
    _objectPosteriorMass[objectIndex] = sum;

    float exploration = _objectExplorationScore[objectIndex] * probabilityDecay;
    if (raysTested > 0.0f) {
      _objectIdleFrames[objectIndex] = 0u;
      if (clampedProbability < probabilityThreshold)
        exploration += raysTested;
      else
        exploration *= rayHitDecay;
    } else {
      uint32_t next = idleFrames;
      if (next < std::numeric_limits<uint32_t>::max())
        ++next;
      _objectIdleFrames[objectIndex] = next;
      exploration *= rayHitDecay;
      if (visible || hasActivePrimitive)
        exploration = std::max(exploration, kIdleVisibleExploreSeed);
    }
    _objectExplorationScore[objectIndex] = exploration;
  }
}

void Renderer::trackFrameCommandBuffer(MTL::CommandBuffer *commandBuffer) {
  if (!commandBuffer)
    return;

  commandBuffer->retain();

  FrameCommandBufferRecord record;
  record.buffer = commandBuffer;
  record.trackedSince = std::chrono::steady_clock::now();

  {
    std::lock_guard<std::mutex> lock(_frameCommandBufferMutex);
    _frameCommandBuffers.push_back(record);
  }

  commandBuffer->addCompletedHandler([this](MTL::CommandBuffer *completed) {
    bool release = false;
    {
      std::lock_guard<std::mutex> lock(_frameCommandBufferMutex);
      auto it = std::find_if(
          _frameCommandBuffers.begin(), _frameCommandBuffers.end(),
          [completed](const FrameCommandBufferRecord &record) {
            return record.buffer == completed;
          });
      if (it != _frameCommandBuffers.end()) {
        _frameCommandBuffers.erase(it);
        release = true;
      }
    }
    if (release)
      completed->release();
  });
}

bool Renderer::waitForPendingFrameCommands(
    std::chrono::milliseconds timeout,
    std::chrono::steady_clock::time_point *waitSnapshot) {
  std::vector<FrameCommandBufferRecord> pending;
  std::vector<MTL::CommandBuffer *> snapshotBuffers;
  std::chrono::steady_clock::time_point waitStart;
  {
    std::lock_guard<std::mutex> lock(_frameCommandBufferMutex);
    pending.reserve(_frameCommandBuffers.size());
    snapshotBuffers.reserve(_frameCommandBuffers.size());
    for (const auto &record : _frameCommandBuffers) {
      if (record.buffer)
        record.buffer->retain();
      pending.push_back(record);
      if (record.buffer)
        snapshotBuffers.push_back(record.buffer);
    }
    waitStart = std::chrono::steady_clock::now();
    if (waitSnapshot)
      *waitSnapshot = waitStart;
  }

  const bool infiniteTimeout = timeout == std::chrono::milliseconds::max();
  bool allComplete = true;

  for (auto &record : pending) {
    auto *buffer = record.buffer;
    if (!buffer)
      continue;

    const auto startTime = record.trackedSince;
    bool completed = false;

    while (true) {
      auto status = buffer->status();
      if (status == MTL::CommandBufferStatusCompleted ||
          status == MTL::CommandBufferStatusError) {
        completed = true;
        break;
      }

      if (!infiniteTimeout) {
        auto now = std::chrono::steady_clock::now();
        if (now - startTime >= timeout)
          break;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (!completed)
      allComplete = false;
  }

  std::unordered_set<MTL::CommandBuffer *> snapshotSet(snapshotBuffers.begin(),
                                                       snapshotBuffers.end());
  bool newCommandTracked = false;
  {
    std::lock_guard<std::mutex> lock(_frameCommandBufferMutex);
    for (const auto &record : _frameCommandBuffers) {
      if (!record.buffer)
        continue;
      bool presentInSnapshot = snapshotSet.find(record.buffer) != snapshotSet.end();
      if (!presentInSnapshot || record.trackedSince >= waitStart) {
        newCommandTracked = true;
        break;
      }
    }
  }

  for (auto &record : pending) {
    if (record.buffer)
      record.buffer->release();
  }

  if (newCommandTracked)
    return false;

  return allComplete;
}

void Renderer::beginFrameMetrics() {
  _cpuStart = std::chrono::high_resolution_clock::now();
  _lastRayCount = static_cast<size_t>(Camera::screenSize.x * Camera::screenSize.y);

  if (_benchmarkEnabled) {
    ensureBenchmarkStream();
    BenchmarkSample sample;
    sample.frameIndex = _benchmarkFrameCounter++;
    sample.rayCount = _lastRayCount;
    sample.primitiveActivations = _framePrimitiveActivations;
    sample.primitiveDeactivations = _framePrimitiveDeactivations;
    sample.objectActivations = _frameObjectActivations;
    sample.objectDeactivations = _frameObjectDeactivations;
    sample.activePrimitiveCount = _activePrimitiveCount;
    sample.residentPrimitiveCount = _residentPrimitiveCount;
    sample.totalPrimitiveCount = _allPrimitives.size();
    sample.activeTriangleCount = _activeTriangleCount;
    sample.residentTriangleCount = _residentTriangleCount;
    sample.totalTriangleCount = _pScene ? _pScene->getTriangleCount() : 0;
    sample.activeNodeCount = _activeNodeCount;
    sample.residentNodeCount = _residentNodeCount;
    sample.totalNodeCount = _totalNodeCount;
    size_t activeObjects = 0;
    for (bool active : _objectActive)
      if (active)
        ++activeObjects;
    sample.activeObjectCount = activeObjects;
    size_t residentObjects = 0;
    for (const auto &resident : _residentObjectGpuResources)
      if (resident.isResident())
        ++residentObjects;
    sample.residentObjectCount = residentObjects;
    sample.gpuMemoryMB = currentGPUMemoryMB();
    sample.scratchMemoryMB = scratchMemoryMB();
    sample.residencyMemoryMB =
        std::max(0.0, sample.gpuMemoryMB - sample.scratchMemoryMB);
    sample.textureMemoryCapMB = _textureResidencyMemoryCapMB;
    sample.avgHitProbability = 0.0;
    sample.p95HitProbability = 0.0;
    sample.probabilityThreshold = _residencyConfig.probabilityThreshold;
    sample.probabilityTargetFraction = _residencyConfig.probabilityTargetFraction;
    sample.probabilityVisibleFloor = _residencyConfig.probabilityVisibleFloor;
    sample.environmentTargetActiveFraction =
        _residencyConfig.environmentTargetActiveFraction;
    sample.environmentEscapeThreshold =
        _residencyConfig.environmentEscapeThreshold;
    sample.environmentDepthWeights =
        formatFloatList(_residencyConfig.environmentDepthWeights);
    sample.environmentDepthRadii =
        formatFloatList(_residencyConfig.environmentDepthRadii);
    if (!_primitiveHitProbability.empty()) {
      size_t primitiveCount =
          std::min(_primitiveHitProbability.size(), _allPrimitives.size());
      std::vector<float> validProbabilities;
      validProbabilities.reserve(primitiveCount);
      double probabilitySum = 0.0;
      std::ostringstream primitiveStream;
      bool firstPrimitive = true;

      std::vector<double> objectProbabilitySums(_allSceneObjects.size(), 0.0);
      std::vector<size_t> objectProbabilityCounts(_allSceneObjects.size(), 0);

      for (size_t index = 0; index < primitiveCount; ++index) {
        float probability = _primitiveHitProbability[index];
        bool probabilityFinite = std::isfinite(probability);
        float sanitized = probabilityFinite
                               ? std::clamp(probability, 0.0f, 1.0f)
                               : 0.0f;

        if (!firstPrimitive)
          primitiveStream << ';';
        primitiveStream << index << ':'
                        << formatFixed(static_cast<double>(sanitized), 6);
        firstPrimitive = false;

        if (probabilityFinite) {
          probabilitySum += sanitized;
          validProbabilities.push_back(sanitized);
        }

        if (index < _primitiveToObject.size()) {
          size_t objectIndex = _primitiveToObject[index];
          if (objectIndex < objectProbabilitySums.size()) {
            objectProbabilitySums[objectIndex] += static_cast<double>(sanitized);
            objectProbabilityCounts[objectIndex] += 1;
          }
        }
      }

      sample.primitiveProbabilities = primitiveStream.str();

      if (!_allSceneObjects.empty()) {
        std::ostringstream objectStream;
        bool firstObject = true;
        for (size_t objectIndex = 0; objectIndex < _allSceneObjects.size();
             ++objectIndex) {
          double avgProbability = 0.0;
          if (objectIndex < objectProbabilitySums.size() &&
              objectProbabilityCounts[objectIndex] > 0) {
            avgProbability = objectProbabilitySums[objectIndex] /
                              static_cast<double>(
                                  objectProbabilityCounts[objectIndex]);
          }
          if (!firstObject)
            objectStream << ';';
          objectStream << objectIndex << ':'
                       << formatFixed(avgProbability, 6);
          firstObject = false;
        }
        sample.objectProbabilities = objectStream.str();
      }

      if (!validProbabilities.empty()) {
        sample.avgHitProbability =
            probabilitySum / static_cast<double>(validProbabilities.size());
        size_t percentileIndex = static_cast<size_t>(std::floor(
            0.95 * static_cast<double>(validProbabilities.size() - 1)));
        percentileIndex = std::min(percentileIndex,
                                    validProbabilities.size() - 1);
        std::nth_element(validProbabilities.begin(),
                         validProbabilities.begin() + percentileIndex,
                         validProbabilities.end());
        sample.p95HitProbability = validProbabilities[percentileIndex];
      }
    } else if (!_allSceneObjects.empty()) {
      std::ostringstream objectStream;
      bool firstObject = true;
      for (size_t objectIndex = 0; objectIndex < _allSceneObjects.size();
           ++objectIndex) {
        if (!firstObject)
          objectStream << ';';
        objectStream << objectIndex << ':' << formatFixed(0.0, 6);
        firstObject = false;
      }
      sample.objectProbabilities = objectStream.str();
    }
    sample.probabilisticToggles = _frameProbabilisticToggles;
    sample.probabilityTargetPrimitives = _frameProbabilityTargetPrimitives;
    sample.probabilityInitialDesiredPrimitives =
        _frameProbabilityInitialDesiredPrimitives;
    sample.probabilityFinalDesiredPrimitives =
        _frameProbabilityFinalDesiredPrimitives;
    sample.probabilityTrimmedPrimitives = _frameProbabilityTrimmedPrimitives;
    sample.probabilityBudgetHit = _frameProbabilityBudgetHit;
    sample.deltaTimeSeconds = _deltaTimeSeconds;
    sample.wallSeconds = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - _benchmarkStartTime)
                            .count();
    sample.strategy = _frameStrategy;
    sample.strategyName = residencyStrategyName(_frameStrategy);
    sample.minSamplesPerPixel = _minSamplesPerPixel;
    sample.maxSamplesPerPixel = _maxSamplesPerPixel;
    sample.accumulationReset = _needsAccumulationReset;
    sample.residentCompacted = _residentCompacted;
    sample.overMemoryCap =
        sample.residencyMemoryMB > _textureResidencyMemoryCapMB;
    _pendingBenchmarkSamples.push_back(std::move(sample));
  }
}

void Renderer::completeFrameMetrics(MTL::CommandBuffer *pCmd) {
  auto cpuEnd = std::chrono::high_resolution_clock::now();
  _lastCPUTime = std::chrono::duration<double>(cpuEnd - _cpuStart).count();
  if (pCmd) {
    _lastGPUTime = pCmd->GPUEndTime() - pCmd->GPUStartTime();
  } else {
    _lastGPUTime = 0.0;
  }
  if (_lastGPUTime > 0.0) {
    _lastRaysPerSecond = static_cast<double>(_lastRayCount) / _lastGPUTime;
  } else {
    _lastRaysPerSecond = 0.0;
  }
  bool canRecalculateNodes =
      (_blasNodeCount > 0 && _cachedBVHNodes.size() >= _blasNodeCount * 2) ||
      (_tlasNodeCount > 0 && _cachedTLASNodes.size() >= _tlasNodeCount * 2) ||
      (_residentCompacted && (_blasNodeCount > 0 || _tlasNodeCount > 0));
  if (canRecalculateNodes) {
    if (_residentObjectGpuResources.empty()) {
      recalculateNodeCounters(_objectResidentState);
    } else {
      auto gpuResidentMask = buildResidentMaskFromGpuResources();
      recalculateNodeCounters(gpuResidentMask);
    }
  }

  size_t offloaded = _totalNodeCount > _residentNodeCount ?
                         _totalNodeCount - _residentNodeCount :
                         0;
  printf("Active nodes: %zu Resident nodes: %zu Offloaded nodes: %zu CPU: %.3f ms GPU: %.3f ms Rays/s: %.2f\n",
         _activeNodeCount, _residentNodeCount, offloaded,
         _lastCPUTime * 1000.0, _lastGPUTime * 1000.0, _lastRaysPerSecond);

  if (_benchmarkEnabled && !_pendingBenchmarkSamples.empty()) {
    BenchmarkSample sample = std::move(_pendingBenchmarkSamples.front());
    _pendingBenchmarkSamples.pop_front();
    sample.cpuTimeSeconds = _lastCPUTime;
    sample.gpuTimeSeconds = _lastGPUTime;
    sample.raysPerSecond = _lastRaysPerSecond;
    sample.rayCount = _lastRayCount;
    sample.wallSeconds = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - _benchmarkStartTime)
                            .count();
    writeBenchmarkRow(sample);
  }
}

double Renderer::lastCPUTime() const { return _lastCPUTime; }
double Renderer::lastGPUTime() const { return _lastGPUTime; }
double Renderer::lastRaysPerSecond() const { return _lastRaysPerSecond; }
size_t Renderer::activeNodeCount() const { return _activeNodeCount; }
size_t Renderer::residentNodeCount() const { return _residentNodeCount; }
size_t Renderer::totalNodeCount() const { return _totalNodeCount; }
bool Renderer::isAlwaysResidentStrategy() const {
  return _frameStrategy == ResidencyStrategy::AlwaysResident;
}
