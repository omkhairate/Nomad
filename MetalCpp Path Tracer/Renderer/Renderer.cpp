#include "Renderer.h"

#include "Camera.h"
#include "InputSystem.h"
#include "Scene.h"
#include "SceneLoader.h"
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
#include <iomanip>
#include <numeric>
#include <simd/simd.h>
#include <sstream>
#include <string>
#include <utility>
#include <unordered_map>

namespace MetalCppPathTracer {

void ResidentObjectGpuResources::clearPendingCommand() {
  if (pendingCommand) {
    pendingCommand->release();
    pendingCommand = nullptr;
  }
}

void ResidentObjectGpuResources::transitionToStreaming(
    MTL::CommandBuffer *pending) {
  clearPendingCommand();
  if (pending) {
    pendingCommand = pending;
    pendingCommand->retain();
  }
  state = ResidencyState::Streaming;
  lastStateChange = std::chrono::steady_clock::now();
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
  if (isResident() && !forceRebuild) {
    lastStateChange = std::chrono::steady_clock::now();
    return true;
  }

  transitionToStreaming();
  bool built = renderer.buildObjectBlas(objectIndex, object, *this);
  if (!built) {
    transitionToCold(instanceRecord);
    return false;
  }

  clearPendingCommand();
  state = ResidencyState::Resident;
  lastStateChange = std::chrono::steady_clock::now();
  return true;
}

} // namespace MetalCppPathTracer

using namespace MetalCppPathTracer;

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
};

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
  float reflective = luminance(m.albedo);
  return area * (emissive + reflective);
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

  float bestCost = std::numeric_limits<float>::max();
  int bestAxis = -1;
  size_t bestSplit = start + range / 2;

  std::vector<simd::float3> leftMin(range);
  std::vector<simd::float3> leftMax(range);
  std::vector<simd::float3> rightMin(range);
  std::vector<simd::float3> rightMax(range);

  for (int axis = 0; axis < 3; ++axis) {
    std::sort(primitiveIndices.begin() + start, primitiveIndices.begin() + end,
              [&](int a, int b) {
                return primitiveAxisValueLocal(primitives[a], axis) <
                       primitiveAxisValueLocal(primitives[b], axis);
              });

    simd::float3 currMin(std::numeric_limits<float>::max());
    simd::float3 currMax(-std::numeric_limits<float>::max());
    for (size_t i = start; i < end; ++i) {
      const Primitive &p = primitives[primitiveIndices[i]];
      simd::float3 pMin, pMax;
      primitiveBoundsLocal(p, pMin, pMax);
      currMin = simd::min(currMin, pMin);
      currMax = simd::max(currMax, pMax);
      leftMin[i - start] = currMin;
      leftMax[i - start] = currMax;
    }

    currMin = simd::float3(std::numeric_limits<float>::max());
    currMax = simd::float3(-std::numeric_limits<float>::max());
    for (size_t i = end; i-- > start;) {
      const Primitive &p = primitives[primitiveIndices[i]];
      simd::float3 pMin, pMax;
      primitiveBoundsLocal(p, pMin, pMax);
      currMin = simd::min(currMin, pMin);
      currMax = simd::max(currMax, pMax);
      rightMin[i - start] = currMin;
      rightMax[i - start] = currMax;
    }

    for (size_t i = 1; i < range; ++i) {
      float saLeft = boundingSurfaceArea(leftMin[i - 1], leftMax[i - 1]);
      float saRight = boundingSurfaceArea(rightMin[i], rightMax[i]);
      size_t leftCount = i;
      size_t rightCount = range - i;
      float cost = 0.125f + (saLeft / parentArea) * leftCount +
                   (saRight / parentArea) * rightCount;
      if (cost < bestCost) {
        bestCost = cost;
        bestAxis = axis;
        bestSplit = start + i;
      }
    }
  }

  if (bestAxis == -1)
    return nodeIndex;

  std::sort(primitiveIndices.begin() + start, primitiveIndices.begin() + end,
            [&](int a, int b) {
              return primitiveAxisValueLocal(primitives[a], bestAxis) <
                     primitiveAxisValueLocal(primitives[b], bestAxis);
            });

  int leftChild =
      buildBVHRecursive(primitives, primitiveIndices, nodes, start, bestSplit);
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
                                    MTL::ResourceOptions storageMode) {
  if (requiredBytes == 0)
    requiredBytes = 1;

  size_t desiredCapacity = requiredBytes;

  if (!allowShrink) {
    if (buffer && requiredBytes <= currentCapacity)
      return;
    desiredCapacity = std::max(requiredBytes, currentCapacity);
  } else if (buffer) {
    size_t shrinkThreshold = currentCapacity / 2;
    if (requiredBytes <= currentCapacity && requiredBytes > shrinkThreshold)
      return;
    if (requiredBytes > currentCapacity)
      desiredCapacity = std::max(requiredBytes, currentCapacity * 2);
  }

  if (buffer) {
    buffer->release();
    buffer = nullptr;
    currentCapacity = 0;
  }

  desiredCapacity = std::max(desiredCapacity, size_t(1));
  buffer = _pDevice->newBuffer(desiredCapacity, storageMode);
  currentCapacity = buffer ? buffer->length() : 0;
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
    : _pDevice(pDevice->retain()), _pScene(new Scene()) {
  _pCommandQueue = _pDevice->newCommandQueue();
  _tlasHeap.initialize(_pDevice);
  _dummyBlasResources.initialize(_pDevice);

  Camera::reset();
  _primaryCameraState = Camera::captureState();
  _observerCameraState = _primaryCameraState;

  updateVisibleScene();
  buildShaders();
  buildTextures();

  recalculateViewport();
  initializeBenchmarking();
}

Renderer::~Renderer() {
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
  if (_pFrustumVertexBuffer)
    _pFrustumVertexBuffer->release();

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
         "resident_nodes,total_nodes,active_objects,resident_objects,gpu_memory_mb,"
         "texture_memory_cap_mb,over_memory_cap,residency_compacted,"
         "accumulation_reset,ray_hit_decay,state_cooldown_frames,lod_toggle_budget,"
         "energy_toggle_budget,screen_toggle_budget,rayhit_toggle_budget,"
         "rayhit_target_fraction,rayhit_min_active,rayhit_rebuild_cooldown,"
         "lod_enter_distance,lod_exit_distance,energy_target_fraction,"
         "energy_min_active,energy_visibility_boost,screen_target_fraction,"
         "screen_min_pixels,screen_min_active";
  _benchmarkStream << '\n';
  _benchmarkHeaderWritten = true;
}

static std::string formatFixed(double value, int precision) {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(precision) << value;
  return ss.str();
}

void Renderer::writeBenchmarkRow(const BenchmarkSample &sample) {
  if (!_benchmarkStream.is_open())
    return;

  auto boolToInt = [](bool v) { return v ? 1 : 0; };

  std::ostringstream row;
  row << sample.frameIndex << ',' << formatFixed(sample.wallSeconds, 6) << ','
      << formatFixed(sample.cpuTimeSeconds * 1000.0, 3) << ','
      << formatFixed(sample.gpuTimeSeconds * 1000.0, 3) << ','
      << formatFixed(sample.raysPerSecond, 2) << ',' << sample.rayCount << ','
      << '"' << sample.strategyName << "\"," << static_cast<int>(sample.strategy)
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
      << formatFixed(sample.gpuMemoryMB, 3) << ','
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
      << formatFixed(_residencyConfig.energyTargetFraction, 3) << ','
      << _residencyConfig.energyMinActivePrimitives << ','
      << formatFixed(_residencyConfig.energyVisibilityBoost, 3) << ','
      << formatFixed(_residencyConfig.screenFootprintTargetFraction, 3) << ','
      << formatFixed(_residencyConfig.screenFootprintMinPixelCoverage, 3) << ','
      << _residencyConfig.screenFootprintMinActivePrimitives;

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
  case ResidencyStrategy::AlwaysResident:
    return "Always resident";
  case ResidencyStrategy::DistanceLOD:
  default:
    return "Distance-based LOD";
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

void Renderer::updateVisibleScene() {
  if (!SceneLoader::LoadSceneFromXML("scene.xml", _pScene)) {
    std::filesystem::path alt =
        std::filesystem::path(__FILE__).parent_path() / "../scene_rayhit_budget_crossfire.xml";
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
  case ResidencyStrategy::AlwaysResident:
    strategyName = "Always resident";
    break;
  case ResidencyStrategy::DistanceLOD:
  default:
    strategyName = "Distance-based LOD";
    break;
  }
  printf("Active primitive residency strategy: %s\n", strategyName);

  // Store full primitive list and initialize tracking
  _allPrimitives = _pScene->getPrimitives();
  size_t primCount = _allPrimitives.size();
  _activePrimitive.assign(primCount, false);
  _primitiveCooldown.assign(primCount, 0);
  _primitiveToResidentIndex.assign(primCount, -1);
  _primitiveToObject.assign(primCount, std::numeric_limits<size_t>::max());
  _primitiveBounds.resize(primCount);
  _primitiveImportance.assign(primCount, 0.0f);
  _objectImportance.clear();
  _energySortedIndices.clear();
  _primitiveHitScores.assign(primCount, 0.0f);
  _primitiveHitLastFrame.assign(primCount, 0);
  _primitiveVisible.assign(primCount, 0);
  _rayHitSortedIndices.resize(primCount);
  _primitiveScreenCoverage.assign(primCount, 0.0f);
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
  for (size_t i = 0; i < primCount; ++i) {
    const Primitive &p = _allPrimitives[i];
    if (p.type == PrimitiveType::Sphere) {
      _primitiveBounds[i] = {p.sphere.center, p.sphere.radius};
    } else if (p.type == PrimitiveType::Triangle) {
      simd::float3 c = (p.triangle.v0 + p.triangle.v1 + p.triangle.v2) / 3.0f;
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
    if (i < _screenCoverageSortedIndices.size())
      _screenCoverageSortedIndices[i] = i;
    _totalPrimitiveImportance += std::max(_primitiveImportance[i], 0.0f);
  }

  size_t hitCount = std::max<size_t>(_maxPrimitiveCount, 1);
  size_t hitBytes = hitCount * sizeof(uint32_t);
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
  if (_pPrimitiveHitBufferGPU && hitBytes > 0) {
    MTL::CommandBuffer *initCmd = _pCommandQueue->commandBuffer();
    if (initCmd) {
      MTL::BlitCommandEncoder *initBlit = initCmd->blitCommandEncoder();
      if (initBlit) {
        initBlit->fillBuffer(_pPrimitiveHitBufferGPU,
                             NS::Range::Make(0, hitBytes), 0);
        initBlit->endEncoding();
      }
      initCmd->commit();
      initCmd->waitUntilCompleted();
    }
  }

  _rayHitRebuildCooldown = 0;

  _maxBlasNodeCount = 1;
  _maxTlasNodeCount = 1;

  _pScene->buildBVH();
  size_t blasNodeCount = _pScene->getBVHNodeCount();
  _maxBlasNodeCount = std::max<size_t>(blasNodeCount, _maxBlasNodeCount);

  size_t fullTlasCount = 0;
  if (primCount > 0) {
    simd::float4 *tmp = _pScene->createTLASBuffer(fullTlasCount);
    delete[] tmp;
  }
  _maxTlasNodeCount = std::max<size_t>(fullTlasCount, _maxTlasNodeCount);
  _totalNodeCount = _maxBlasNodeCount + _maxTlasNodeCount;

  _allSceneObjects = _pScene->getObjects();
  size_t objectCount = _allSceneObjects.size();
  _objectBounds.resize(objectCount);
  _objectActive.assign(objectCount, false);
  _objectCooldown.assign(objectCount, 0);
  _objectImportance.assign(objectCount, 0.0f);
  _energySortedIndices.resize(objectCount);
  std::iota(_energySortedIndices.begin(), _energySortedIndices.end(), size_t(0));
  _residentObjectGpuResources.resize(objectCount);

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
    size_t last = first + obj.primitiveCount;
    for (size_t prim = first; prim < last && prim < _primitiveToObject.size();
         ++prim)
      _primitiveToObject[prim] = objectIndex;
  }

  updateResidency(true, true);
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

  std::vector<simd::float3> vertices;
  std::vector<uint32_t> indices;
  vertices.reserve((last - first) * 3);
  indices.reserve((last - first) * 3);

  for (size_t prim = first; prim < last; ++prim) {
    if (prim >= _allPrimitives.size())
      break;
    const Primitive &p = _allPrimitives[prim];
    if (p.type != PrimitiveType::Triangle)
      continue;

    uint32_t baseIndex = static_cast<uint32_t>(vertices.size());
    vertices.push_back(p.triangle.v0);
    vertices.push_back(p.triangle.v1);
    vertices.push_back(p.triangle.v2);
    indices.push_back(baseIndex + 0);
    indices.push_back(baseIndex + 1);
    indices.push_back(baseIndex + 2);
  }

  size_t triangleCount = indices.size() / 3;

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

  NS::UInteger vertexBytes =
      static_cast<NS::UInteger>(vertices.size() * sizeof(simd::float3));
  NS::UInteger indexBytes =
      static_cast<NS::UInteger>(indices.size() * sizeof(uint32_t));

  MTL::AccelerationStructureTriangleGeometryDescriptor *geometryDesc =
      MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
  geometryDesc->setOpaque(true);

  std::string blasLabel = "ObjectBLAS_" + std::to_string(objectIndex);
  std::string vertexLabel = "ObjectVertices_" + std::to_string(objectIndex);
  std::string indexLabel = "ObjectIndices_" + std::to_string(objectIndex);

  NS::Object *descriptorObjects[] = {geometryDesc};
  NS::Array *geometryArray =
      NS::Array::alloc()->init(descriptorObjects, 1);

  MTL::PrimitiveAccelerationStructureDescriptor *accelDesc =
      MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
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
    if (geometryArray)
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
    geometryDesc->setTriangleCount(triangleCount);
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
        if (geometryArray)
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

  MTL::AccelerationStructure *accelerationStructure =
      resident.resources.ensureAccelerationStructure(
          sizes.accelerationStructureSize, blasLabel.c_str());

  if (!accelerationStructure) {
    if (geometryArray)
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
      std::memcpy(vertexStaging->contents(), vertices.data(), vertexBytes);
      markBufferModified(vertexStaging, NS::Range::Make(0, vertexBytes));
    }
  }
  if (indexBytes > 0) {
    indexStaging =
        _pDevice->newBuffer(indexBytes, MTL::ResourceStorageModeShared);
    if (indexStaging) {
      std::memcpy(indexStaging->contents(), indices.data(), indexBytes);
      markBufferModified(indexStaging, NS::Range::Make(0, indexBytes));
    }
  }

  if ((vertexBytes > 0 && !vertexStaging) ||
      (indexBytes > 0 && !indexStaging)) {
    if (indexStaging)
      indexStaging->release();
    if (vertexStaging)
      vertexStaging->release();
    if (geometryArray)
      geometryArray->release();
    geometryDesc->release();
    accelDesc->release();
    cleanupPool();
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
    if (indexStaging)
      indexStaging->release();
    if (vertexStaging)
      vertexStaging->release();
    if (geometryArray)
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

  MTL::AccelerationStructureCommandEncoder *asEncoder =
      commandBuffer->accelerationStructureCommandEncoder();
  if (!asEncoder) {
    if (scratchBuffer)
      scratchBuffer->release();
    if (indexStaging)
      indexStaging->release();
    if (vertexStaging)
      vertexStaging->release();
    if (geometryArray)
      geometryArray->release();
    geometryDesc->release();
    accelDesc->release();
    cleanupPool();
    return false;
  }

  asEncoder->buildAccelerationStructure(accelerationStructure, accelDesc,
                                        scratchBuffer, 0);
  asEncoder->endEncoding();

  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  if (scratchBuffer)
    scratchBuffer->release();
  if (indexStaging)
    indexStaging->release();
  if (vertexStaging)
    vertexStaging->release();
  if (geometryArray)
    geometryArray->release();
  geometryDesc->release();
  accelDesc->release();

  resident.byteSize = totalHeapBytes;
  resident.triangleCount = triangleCount;
  resident.vertexCount = vertices.size();
  resident.vertexBufferOffset = 0;
  resident.indexBufferOffset = 0;
  resident.geometryValid = true;

  cleanupPool();
  return true;
}

bool Renderer::ensureDummyBlas() {
  if (_pDummyBlas)
    return true;

  if (!_pDevice || !_pCommandQueue)
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
    if (vertexZeroRequest.staging)
      vertexZeroRequest.staging->release();
    if (indexZeroRequest.staging)
      indexZeroRequest.staging->release();
    geometryDesc->release();
    return false;
  }

  NS::Object *geometryObjects[] = {geometryDesc};
  NS::Array *geometryArray = NS::Array::alloc()->init(geometryObjects, 1);

  MTL::PrimitiveAccelerationStructureDescriptor *accelDesc =
      MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
  accelDesc->setGeometryDescriptors(geometryArray);

  MTL::AccelerationStructureSizes sizes =
      _pDevice->accelerationStructureSizes(accelDesc);

  MTL::AccelerationStructure *structure =
      _dummyBlasResources.ensureAccelerationStructure(
          sizes.accelerationStructureSize, "DummyBLAS");

  if (!structure) {
    if (vertexZeroRequest.staging)
      vertexZeroRequest.staging->release();
    if (indexZeroRequest.staging)
      indexZeroRequest.staging->release();
    geometryArray->release();
    geometryDesc->release();
    accelDesc->release();
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
    if (vertexZeroRequest.staging)
      vertexZeroRequest.staging->release();
    if (indexZeroRequest.staging)
      indexZeroRequest.staging->release();
    geometryArray->release();
    geometryDesc->release();
    accelDesc->release();
    return false;
  }

  if (vertexZeroRequest.staging || indexZeroRequest.staging) {
    MTL::BlitCommandEncoder *blitEncoder = commandBuffer->blitCommandEncoder();
    if (!blitEncoder) {
      if (scratchBuffer)
        scratchBuffer->release();
      if (vertexZeroRequest.staging)
        vertexZeroRequest.staging->release();
      if (indexZeroRequest.staging)
        indexZeroRequest.staging->release();
      geometryArray->release();
      geometryDesc->release();
      accelDesc->release();
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
    if (vertexZeroRequest.staging)
      vertexZeroRequest.staging->release();
    if (indexZeroRequest.staging)
      indexZeroRequest.staging->release();
    geometryArray->release();
    geometryDesc->release();
    accelDesc->release();
    return false;
  }

  encoder->buildAccelerationStructure(structure, accelDesc, scratchBuffer, 0);
  encoder->endEncoding();

  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  if (vertexZeroRequest.staging)
    vertexZeroRequest.staging->release();
  if (indexZeroRequest.staging)
    indexZeroRequest.staging->release();
  if (scratchBuffer)
    scratchBuffer->release();
  geometryArray->release();
  geometryDesc->release();
  accelDesc->release();

  _pDummyBlas = structure;
  return true;
}

void Renderer::updateTopLevelAccelerationStructure(
    const std::vector<MTL::AccelerationStructureInstanceDescriptor> &descriptors,
    const std::vector<MTL::AccelerationStructure *> &structures) {
  if (!_pDevice || !_pCommandQueue)
    return;

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
    stagingBuffer =
        _pDevice->newBuffer(descriptorBytes, MTL::ResourceStorageModeShared);
    if (!stagingBuffer)
      return;

    std::memcpy(stagingBuffer->contents(), descriptors.data(), descriptorBytes);
    NS::UInteger stagingLength =
        static_cast<NS::UInteger>(descriptorBytes);
    markBufferModified(stagingBuffer, NS::Range::Make(0, stagingLength));
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
      if (stagingBuffer)
        stagingBuffer->release();
      return;
    }
    _pTlasStructure = structure;
  }

  NS::UInteger scratchSize = needsRebuild ? sizes.buildScratchBufferSize
                                          : sizes.refitScratchBufferSize;
  MTL::Buffer *scratchBuffer = nullptr;
  if (scratchSize > 0)
    scratchBuffer =
        _pDevice->newBuffer(scratchSize, MTL::ResourceStorageModePrivate);

  MTL::CommandBuffer *commandBuffer = _pCommandQueue->commandBuffer();
  if (!commandBuffer) {
    if (scratchBuffer)
      scratchBuffer->release();
    if (stagingBuffer)
      stagingBuffer->release();
    instanceDesc->release();
    return;
  }

  MTL::BlitCommandEncoder *blit = nullptr;
  auto ensureBlitEncoder = [&]() -> MTL::BlitCommandEncoder * {
    if (!blit)
      blit = commandBuffer->blitCommandEncoder();
    return blit;
  };

  if (stagingBuffer && descriptorBytes > 0) {
    MTL::BlitCommandEncoder *enc = ensureBlitEncoder();
    if (!enc) {
      if (scratchBuffer)
        scratchBuffer->release();
      stagingBuffer->release();
      instanceDesc->release();
      return;
    }
    enc->copyFromBuffer(stagingBuffer, 0, instanceBuffer, 0, descriptorBytes);
  } else if (needsDescriptorUpload && descriptorBytes == 0 && instanceBuffer &&
             instanceBuffer->length() > 0) {
    MTL::BlitCommandEncoder *enc = ensureBlitEncoder();
    if (!enc) {
      if (scratchBuffer)
        scratchBuffer->release();
      if (stagingBuffer)
        stagingBuffer->release();
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
    if (scratchBuffer)
      scratchBuffer->release();
    if (stagingBuffer)
      stagingBuffer->release();
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

  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  if (scratchBuffer)
    scratchBuffer->release();
  if (stagingBuffer)
    stagingBuffer->release();
  instanceDesc->release();

  _cachedInstanceDescriptors = descriptors;
  _cachedInstancedAccelerationStructures = instancedStructures;
}

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

  if (needFullUpload) {
    _cachedPrimitiveData.assign(totalPrimitiveCount * 3,
                                simd::float4{0.0f, 0.0f, 0.0f, 0.0f});
    _cachedMaterialData.assign(totalPrimitiveCount * 2,
                               simd::float4{0.0f, 0.0f, 0.0f, 0.0f});

    for (size_t i = 0; i < totalPrimitiveCount; ++i) {
      const Primitive &p = _allPrimitives[i];
      simd::float4 *primBase = &_cachedPrimitiveData[3 * i];
      simd::float4 *matBase = &_cachedMaterialData[2 * i];

      switch (p.type) {
      case PrimitiveType::Sphere: {
        primBase[0] =
            simd::make_float4(p.sphere.center, static_cast<float>(p.type));
        primBase[1] = simd::make_float4(
            simd::make_float3(p.sphere.radius, 0.0f, 0.0f), 0.0f);
        primBase[2] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        break;
      }
      case PrimitiveType::Rectangle: {
        primBase[0] =
            simd::make_float4(p.rectangle.center, static_cast<float>(p.type));
        primBase[1] = simd::make_float4(p.rectangle.u, 0.0f);
        primBase[2] = simd::make_float4(p.rectangle.v, 0.0f);
        break;
      }
      case PrimitiveType::Triangle: {
        primBase[0] =
            simd::make_float4(p.triangle.v0, static_cast<float>(p.type));
        primBase[1] = simd::make_float4(p.triangle.v1, 0.0f);
        primBase[2] = simd::make_float4(p.triangle.v2, 0.0f);
        break;
      }
      }

      const Material &m = p.material;
      matBase[0] = simd::make_float4(m.albedo, m.materialType);
      matBase[1] = simd::make_float4(m.emissionColor, m.emissionPower);
    }

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

  const auto &sceneObjects = _pScene->getObjects();
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

  if (!needFullUpload) {
    deduplicate(_recentlyActivated);
    deduplicate(_recentlyDeactivated);
    if (!_recentlyActivated.empty() && !_recentlyDeactivated.empty()) {
      for (size_t idx : _recentlyDeactivated) {
        auto it = std::lower_bound(_recentlyActivated.begin(),
                                   _recentlyActivated.end(), idx);
        if (it != _recentlyActivated.end() && *it == idx)
          _recentlyActivated.erase(it);
      }
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
    remapUpload.clear();
    compactPrimitiveData.clear();
    compactMaterialData.clear();
    compactPrimitiveIndices.clear();
    compactBVHNodes.clear();
    compactTriangleVertices.clear();
    compactTriangleIndices.clear();
    compactActiveMask.clear();

    compactPrimitiveData.reserve(_activePrimitiveCount * 3);
    compactMaterialData.reserve(_activePrimitiveCount * 2);
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
          switch (p.type) {
          case PrimitiveType::Sphere: {
            prim0 =
                simd::make_float4(p.sphere.center, static_cast<float>(p.type));
            prim1 = simd::make_float4(
                simd::make_float3(p.sphere.radius, 0.0f, 0.0f), 0.0f);
            prim2 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
            break;
          }
          case PrimitiveType::Rectangle: {
            prim0 = simd::make_float4(p.rectangle.center,
                                      static_cast<float>(p.type));
            prim1 = simd::make_float4(p.rectangle.u, 0.0f);
            prim2 = simd::make_float4(p.rectangle.v, 0.0f);
            break;
          }
          case PrimitiveType::Triangle: {
            prim0 =
                simd::make_float4(p.triangle.v0, static_cast<float>(p.type));
            prim1 = simd::make_float4(p.triangle.v1, 0.0f);
            prim2 = simd::make_float4(p.triangle.v2, 0.0f);
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

          const Material &m = p.material;
          compactMaterialData.push_back(
              simd::make_float4(m.albedo, m.materialType));
          compactMaterialData.push_back(
              simd::make_float4(m.emissionColor, m.emissionPower));
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
        switch (p.type) {
        case PrimitiveType::Sphere: {
          prim0 = simd::make_float4(p.sphere.center, static_cast<float>(p.type));
          prim1 = simd::make_float4(
              simd::make_float3(p.sphere.radius, 0.0f, 0.0f), 0.0f);
          prim2 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          break;
        }
        case PrimitiveType::Rectangle: {
          prim0 = simd::make_float4(p.rectangle.center, static_cast<float>(p.type));
          prim1 = simd::make_float4(p.rectangle.u, 0.0f);
          prim2 = simd::make_float4(p.rectangle.v, 0.0f);
          break;
        }
        case PrimitiveType::Triangle: {
          prim0 = simd::make_float4(p.triangle.v0, static_cast<float>(p.type));
          prim1 = simd::make_float4(p.triangle.v1, 0.0f);
          prim2 = simd::make_float4(p.triangle.v2, 0.0f);
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

        const Material &m = p.material;
        compactMaterialData.push_back(simd::make_float4(m.albedo, m.materialType));
        compactMaterialData.push_back(
            simd::make_float4(m.emissionColor, m.emissionPower));
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

    if (!shouldBeResident) {
      gpuResident.transitionToCold(instanceRecord);
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

  bool uploadAll = needFullUpload || useCompaction || compactionStateChanged;
  bool allowShrink = useCompaction;

  if (uploadAll) {
    size_t primitiveFloat4Count = primitiveSource->size();
    size_t primitiveBytes =
        std::max<size_t>(primitiveFloat4Count, size_t(1)) *
        sizeof(simd::float4);
    ensureBufferCapacity(_pSphereBuffer, primitiveBytes, _sphereBufferCapacity,
                         allowShrink);
    if (_pSphereBuffer) {
      simd::float4 *dst =
          static_cast<simd::float4 *>(_pSphereBuffer->contents());
      if (primitiveFloat4Count > 0)
        std::memcpy(dst, primitiveSource->data(),
                    primitiveFloat4Count * sizeof(simd::float4));
      else
        dst[0] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
      markBufferModified(_pSphereBuffer, NS::Range::Make(0, primitiveBytes));
    }

    size_t materialFloat4Count = materialSource->size();
    size_t materialBytes =
        std::max<size_t>(materialFloat4Count, size_t(1)) *
        sizeof(simd::float4);
    ensureBufferCapacity(_pSphereMaterialBuffer, materialBytes,
                         _sphereMaterialBufferCapacity, allowShrink);
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
                         _primitiveIndexBufferCapacity, allowShrink);
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
    ensureBufferCapacity(_pBVHBuffer, bvhBytes, _bvhBufferCapacity, allowShrink);
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
                         allowShrink);
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
                         _primitiveRemapBufferCapacity, allowShrink);
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
                         _triangleVertexBufferCapacity, allowShrink);
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
                         _triangleIndexBufferCapacity, allowShrink);
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
  }

  size_t instanceCount = std::max<size_t>(_instanceRecords.size(), size_t(1));
  size_t instanceBytes = instanceCount * sizeof(BlasInstanceRecord);
  ensureBufferCapacity(_pInstanceBuffer, instanceBytes, _instanceBufferCapacity,
                       allowShrink);
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
                       _geometryHandleBufferCapacity, allowShrink);
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
                       allowShrink);
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
                       _lightIndexBufferCapacity, allowShrink);
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
                       allowShrink);
  if (_pLightCdfBuffer) {
    float *dst = static_cast<float *>(_pLightCdfBuffer->contents());
    if (_cachedLightCdf.empty())
      dst[0] = 0.0f;
    else
      std::memcpy(dst, _cachedLightCdf.data(),
                  _cachedLightCdf.size() * sizeof(float));
    markBufferModified(_pLightCdfBuffer, NS::Range::Make(0, lightCdfBytes));
  }

  _residentNodeCount = _blasNodeCount + _tlasNodeCount;
  size_t activeBlasNodes = 0;
  size_t activeTlasNodes = 0;
  if (useCompaction) {
    activeBlasNodes = _blasNodeCount;
    activeTlasNodes = (_residentPrimitiveCount > 0) ? _tlasNodeCount : 0;
  } else if (totalPrimitiveCount > 0 && _blasNodeCount > 0) {
    size_t denom = std::max<size_t>(totalPrimitiveCount, size_t(1));
    activeBlasNodes = std::max<size_t>(
        size_t(1), (_blasNodeCount * _activePrimitiveCount + denom - 1) /
                        denom);
    activeTlasNodes = (_tlasNodeCount > 0 && _activePrimitiveCount > 0)
                          ? _tlasNodeCount
                          : size_t(0);
  }
  _activeNodeCount = activeBlasNodes + activeTlasNodes;

  _recentlyActivated.clear();
  _recentlyDeactivated.clear();

  _needsAccumulationReset = true;
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

MTL::Texture *Renderer::requestResidentTexture(ManagedTextureSlot &slot,
                                               MTL::CommandBuffer *cmd,
                                               MTL::BlitCommandEncoder *&blit) {
  if (slot.texture || !slot.descriptorValid || slot.width == 0 ||
      slot.height == 0)
    return slot.texture;

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
  if (slot.stagingBuffer && slot.stagingValid && cmd) {
    size_t totalBytes = textureByteSize(slot);
    if (totalBytes > 0 && slot.stagingCapacity >= totalBytes) {
      if (!blit)
        blit = cmd->blitCommandEncoder();
      if (blit) {
        size_t rowBytes = slot.width * bytesPerPixel(slot.pixelFormat);
        size_t alignedRowBytes = alignTo(rowBytes, 256);
        NS::UInteger bytesPerImage =
            static_cast<NS::UInteger>(alignedRowBytes * slot.height);
        MTL::Size size = MTL::Size::Make(slot.width, slot.height, 1);
        MTL::Origin origin = MTL::Origin::Make(0, 0, 0);
        blit->copyFromBuffer(slot.stagingBuffer, 0, alignedRowBytes,
                             bytesPerImage, size, slot.texture, 0, 0, origin);
        std::printf(
            "[TextureResidency] Restored slot %s from staging buffer (%zu bytes).\n",
            label, totalBytes);
        logged = true;
      } else {
        slot.stagingValid = false;
        _needsAccumulationReset = true;
        _accumulationTargetsNeedClear = true;
        std::printf(
            "[TextureResidency] Restored slot %s without staging data: failed to "
            "obtain blit encoder.\n",
            label);
        logged = true;
      }
    } else {
      slot.stagingValid = false;
      _needsAccumulationReset = true;
      _accumulationTargetsNeedClear = true;
      std::printf(
          "[TextureResidency] Restored slot %s without staging data: staging "
          "buffer capacity (%zu) insufficient for %zu bytes.\n",
          label, slot.stagingCapacity, totalBytes);
      logged = true;
    }
  }

  if (!logged) {
    std::printf(
        "[TextureResidency] Restored slot %s without staging data.\n", label);
  }

  return slot.texture;
}

bool Renderer::evictTextureSlot(ManagedTextureSlot &slot,
                                MTL::CommandBuffer *cmd,
                                MTL::BlitCommandEncoder *&blit) {
  if (!slot.texture)
    return false;

  const char *label = textureSlotLabel(slot);
  size_t totalBytes = textureByteSize(slot);
  if (totalBytes == 0 || !cmd) {
    std::string reason;
    if (totalBytes == 0)
      reason += "zero-sized texture";
    if (!cmd) {
      if (!reason.empty())
        reason += ", ";
      reason += "no command buffer";
    }
    std::printf(
        "[TextureResidency] Evicting slot %s: releasing without staging (%s).\n",
        label, reason.c_str());
    slot.texture->release();
    slot.texture = nullptr;
    slot.stagingValid = false;
    _needsAccumulationReset = true;
    _accumulationTargetsNeedClear = true;
    return true;
  }

  ensureBufferCapacity(slot.stagingBuffer, totalBytes, slot.stagingCapacity,
                       false, MTL::ResourceStorageModeShared);
  if (!slot.stagingBuffer) {
    std::printf(
        "[TextureResidency] Evicting slot %s: releasing without staging (failed "
        "to allocate staging buffer).\n",
        label);
    slot.texture->release();
    slot.texture = nullptr;
    slot.stagingValid = false;
    _needsAccumulationReset = true;
    _accumulationTargetsNeedClear = true;
    return true;
  }

  if (!blit)
    blit = cmd->blitCommandEncoder();
  if (!blit) {
    std::printf(
        "[TextureResidency] Evicting slot %s: releasing without staging (failed "
        "to create blit encoder).\n",
        label);
    slot.texture->release();
    slot.texture = nullptr;
    slot.stagingValid = false;
    _needsAccumulationReset = true;
    _accumulationTargetsNeedClear = true;
    return true;
  }

  size_t rowBytes = slot.width * bytesPerPixel(slot.pixelFormat);
  size_t alignedRowBytes = alignTo(rowBytes, 256);
  NS::UInteger bytesPerImage =
      static_cast<NS::UInteger>(alignedRowBytes * slot.height);
  MTL::Size size = MTL::Size::Make(slot.width, slot.height, 1);
  MTL::Origin origin = MTL::Origin::Make(0, 0, 0);
  blit->copyFromTexture(slot.texture, 0, 0, origin, size, slot.stagingBuffer, 0,
                        alignedRowBytes, bytesPerImage);

  std::printf(
      "[TextureResidency] Evicting slot %s: copied %zu bytes to staging buffer "
      "before release.\n",
      label, totalBytes);
  slot.stagingValid = true;
  slot.texture->release();
  slot.texture = nullptr;
  return true;
}

void Renderer::updateTextureResidency(MTL::CommandBuffer *cmd) {
  if (!cmd || _needsAccumulationReset)
    return;

  bool belowBudget = _residentPrimitiveCount < kTextureResidencyPrimitiveBudget;
  bool overMemory = currentGPUMemoryMB() > _textureResidencyMemoryCapMB;
  if (!belowBudget && !overMemory)
    return;

  std::array<ManagedTextureSlot *, 4> slots = {
      &_accumulationSlots[0], &_accumulationSlots[1], &_sampleCountSlot,
      &_sampleImportanceSlot};

  MTL::BlitCommandEncoder *blit = nullptr;
  for (ManagedTextureSlot *slot : slots)
    evictTextureSlot(*slot, cmd, blit);
  if (blit)
    blit->endEncoding();
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

  _needsAccumulationReset = true;
  _accumulationTargetsNeedClear = true;
}

bool Renderer::resetAccumulationTargets(MTL::CommandBuffer *cmd) {
  if (!cmd)
    return false;

  std::array<ManagedTextureSlot *, 4> slots = {
      &_accumulationSlots[0], &_accumulationSlots[1], &_sampleCountSlot,
      &_sampleImportanceSlot};

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
  if (viewChanged)
    recalculateViewport();

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
  Camera::deltaTime =
      _observerActive ? 0.0f : static_cast<float>(_deltaTimeSeconds);
  updateUniforms(cameraChanged);
  beginFrameMetrics();
  std::swap(_accumulationSlots[0], _accumulationSlots[1]);

  NS::AutoreleasePool *pPool = NS::AutoreleasePool::alloc()->init();

  MTL::CommandBuffer *pCmd = _pCommandQueue->commandBuffer();
  if (!pCmd) {
    if (_benchmarkEnabled && !_pendingBenchmarkSamples.empty())
      _pendingBenchmarkSamples.pop_back();
    if (_accumulationTargetsNeedClear) {
      _accumulationTargetsNeedClear = false;
      _needsAccumulationReset = false;
    }
    pPool->release();
    return;
  }

  bool belowBudget =
      _residentPrimitiveCount < kTextureResidencyPrimitiveBudget;
  double currentMemory = currentGPUMemoryMB();
  bool overCap = currentMemory > _textureResidencyMemoryCapMB;

  if (!_needsAccumulationReset && (belowBudget || overCap))
    updateTextureResidency(pCmd);

  bool allowResidency =
      _needsAccumulationReset || (!belowBudget && !overCap);

  MTL::BlitCommandEncoder *restoreBlit = nullptr;
  MTL::Texture *accum0 = _accumulationSlots[0].texture;
  MTL::Texture *accum1 = _accumulationSlots[1].texture;
  MTL::Texture *sampleCount = _sampleCountSlot.texture;
  MTL::Texture *sampleImportance = _sampleImportanceSlot.texture;
  if (allowResidency) {
    accum0 = requestResidentTexture(_accumulationSlots[0], pCmd, restoreBlit);
    accum1 = requestResidentTexture(_accumulationSlots[1], pCmd, restoreBlit);
    sampleCount = requestResidentTexture(_sampleCountSlot, pCmd, restoreBlit);
    sampleImportance =
        requestResidentTexture(_sampleImportanceSlot, pCmd, restoreBlit);
  }
  if (restoreBlit)
    restoreBlit->endEncoding();

  bool haveAllTextures =
      accum0 && accum1 && sampleCount && sampleImportance;

  if (_accumulationTargetsNeedClear && haveAllTextures) {
    if (resetAccumulationTargets(pCmd)) {
      _accumulationTargetsNeedClear = false;
      _needsAccumulationReset = false;
    }
  }

  pCmd->addCompletedHandler([this](MTL::CommandBuffer *cmd) {
    this->completeFrameMetrics(cmd);
  });

  if (!haveAllTextures) {
    MTL::Drawable *drawable = pView->currentDrawable();
    if (drawable)
      pCmd->presentDrawable(drawable);
    pCmd->commit();
    pPool->release();
    return;
  }

  updateAdaptiveSamplingMaps(pCmd);

  if (_pPathTracePSO && haveAllTextures) {
    MTL::ComputeCommandEncoder *pCompute = pCmd->computeCommandEncoder();
    if (pCompute) {
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

      NS::UInteger width = accum1 ? accum1->width() : 0;
      NS::UInteger height = accum1 ? accum1->height() : 0;
      if (width > 0 && height > 0) {
        NS::UInteger tgWidth = std::max<NS::UInteger>(1, _pPathTracePSO->threadExecutionWidth());
        NS::UInteger maxThreads = std::max<NS::UInteger>(tgWidth,
            _pPathTracePSO->maxTotalThreadsPerThreadgroup());
        NS::UInteger tgHeight = std::max<NS::UInteger>(1, maxThreads / tgWidth);
        MTL::Size threadsPerThreadgroup = MTL::Size::Make(tgWidth, tgHeight, 1);
        MTL::Size threadgroups = MTL::Size::Make(
            (width + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            (height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height, 1);
        pCompute->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
      }

      pCompute->endEncoding();
    }
  }

  MTL::RenderPassDescriptor *pRpd = pView->currentRenderPassDescriptor();
  MTL::RenderCommandEncoder *pEnc = pCmd->renderCommandEncoder(pRpd);

  pEnc->setRenderPipelineState(_pPSO);
  pEnc->setFragmentTexture(accum1, 0);

  pEnc->drawPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle,
                       NS::UInteger(0), NS::UInteger(6));

  if (_pOverlayPSO && _observerActive) {
    float nearDistance = kFrustumDebugNear;
    float baseDistance = std::max(_primaryCameraState.focalLength, 1.0f);
    float farDistance =
        std::max(baseDistance * kFrustumDebugFarMultiplier, nearDistance * 2.0f);

    auto corners = buildFrustumCorners(_primaryCameraState, nearDistance, farDistance);
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

  MTL::BlitCommandEncoder *pBlit = pCmd->blitCommandEncoder();
  bool performedRayHitReadback = false;
  if (pBlit && _pPrimitiveHitBufferGPU && _pPrimitiveHitReadback) {
    size_t bytes =
        std::min(_pPrimitiveHitBufferGPU->length(),
                 _pPrimitiveHitReadback->length());
    if (bytes > 0) {
      pBlit->copyFromBuffer(_pPrimitiveHitBufferGPU, 0, _pPrimitiveHitReadback, 0,
                            bytes);
      pBlit->fillBuffer(_pPrimitiveHitBufferGPU, NS::Range::Make(0, bytes), 0);
      performedRayHitReadback = true;
    }
  }
  if (pBlit)
    pBlit->endEncoding();

  pCmd->presentDrawable(pView->currentDrawable());
  pCmd->commit();

  if (performedRayHitReadback) {
    if (_lastRayHitCommandBuffer)
      _lastRayHitCommandBuffer->release();
    _lastRayHitCommandBuffer = pCmd;
    _lastRayHitCommandBuffer->retain();
    _rayHitCopyError = false;
  }

  pPool->release();
}

void Renderer::updateResidency(bool forceAllToggles, bool forceFullRebuild) {
  if (!_pScene)
    return;

  _framePrimitiveActivations = 0;
  _framePrimitiveDeactivations = 0;
  _frameObjectActivations = 0;
  _frameObjectDeactivations = 0;
  _frameStrategy = _pScene->getResidencyStrategy();

  for (uint32_t &cooldown : _primitiveCooldown)
    if (cooldown > 0)
      --cooldown;
  for (uint32_t &cooldown : _objectCooldown)
    if (cooldown > 0)
      --cooldown;
  if (_compactionCooldown > 0)
    --_compactionCooldown;

  bool changed = false;
  switch (_pScene->getResidencyStrategy()) {
  case ResidencyStrategy::EnergyImportance:
    changed = updateEnergyImportance(forceAllToggles);
    break;
  case ResidencyStrategy::RayHitBudget:
    changed = updateRayHitBudget(forceAllToggles);
    break;
  case ResidencyStrategy::ScreenSpaceFootprint:
    changed = updateScreenSpaceFootprint(forceAllToggles);
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

bool Renderer::updateAlwaysResident(bool /*forceAllToggles*/) {
  bool changed = false;

  for (size_t objectIndex = 0; objectIndex < _allSceneObjects.size(); ++objectIndex) {
    size_t toggled = setObjectActive(objectIndex, true);
    if (toggled > 0)
      changed = true;
  }

  for (size_t primIndex = 0; primIndex < _activePrimitive.size(); ++primIndex) {
    if (setPrimitiveActive(primIndex, true))
      changed = true;
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
  // distance thresholds.

  size_t toggles = 0;
  bool changed = false;

  const size_t objectCount = _allSceneObjects.size();
  std::vector<float> objectDistances(objectCount,
                                     std::numeric_limits<float>::max());
  std::vector<bool> objectBehind(objectCount, false);
  std::vector<size_t> sortedIndices(objectCount);
  simd::float3 forward = Camera::forward;
  float forwardLenSq = simd::length_squared(forward);
  bool forwardValid = forwardLenSq >= 1e-6f;
  if (forwardValid)
    forward /= std::sqrt(forwardLenSq);
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    const BoundingSphere &sphere =
        (objectIndex < _objectBounds.size())
            ? _objectBounds[objectIndex]
            : BoundingSphere{simd::make_float3(0.0f, 0.0f, 0.0f), 0.0f};
    simd::float3 toCenter = sphere.center - Camera::position;
    float dist = simd::length(toCenter) - sphere.radius;
    if (forwardValid) {
      float forwardDepth = simd::dot(toCenter, forward);
      if (forwardDepth + sphere.radius <= 0.0f) {
        objectDistances[objectIndex] = std::numeric_limits<float>::max();
        objectBehind[objectIndex] = true;
        sortedIndices[objectIndex] = objectIndex;
        continue;
      }
    }
    objectDistances[objectIndex] = std::max(dist, 0.0f);
    sortedIndices[objectIndex] = objectIndex;
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
    bool shouldBeActive =
        currentlyActive ? (dist <= _residencyConfig.lodExitDistance && !behind)
                         : (dist < _residencyConfig.lodEnterDistance && !behind);
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

  const SceneObject &obj = _allSceneObjects[objectIndex];
  size_t toggled = 0;
  size_t first = obj.firstPrimitive;
  size_t last = first + obj.primitiveCount;
  for (size_t prim = first; prim < last && prim < _activePrimitive.size(); ++prim)
    if (setPrimitiveActive(prim, active))
      ++toggled;

  if (objectIndex >= _objectActive.size())
    _objectActive.resize(objectIndex + 1, false);
  if (objectIndex >= _objectCooldown.size())
    _objectCooldown.resize(objectIndex + 1, 0);

  bool anyActive = false;
  for (size_t prim = first; prim < last && prim < _activePrimitive.size(); ++prim) {
    if (_activePrimitive[prim])
      anyActive = true;
  }

  bool newState = anyActive;
  bool fullyInactive = !anyActive;
  bool prevState = _objectActive[objectIndex];
  _objectActive[objectIndex] = newState;

  if (prevState != newState) {
    if (newState)
      ++_frameObjectActivations;
    else
      ++_frameObjectDeactivations;
  }

  if (toggled > 0 || prevState != newState || fullyInactive)
    _objectCooldown[objectIndex] = _residencyConfig.stateCooldownFrames;

  return toggled;
}

bool Renderer::updateEnergyImportance(bool forceAllToggles) {
  if (_activePrimitive.empty())
    return false;

  const size_t objectCount = _allSceneObjects.size();
  if (objectCount == 0) {
    // Fall back to primitive-level logic if no objects are available.
    const size_t primCount = _activePrimitive.size();
    std::vector<bool> desiredState(primCount, false);
    std::vector<size_t> sortedIndices(primCount);
    std::iota(sortedIndices.begin(), sortedIndices.end(), size_t(0));
    std::sort(sortedIndices.begin(), sortedIndices.end(),
              [this](size_t a, size_t b) {
                float scoreA = sanitizeSortValue(_primitiveImportance[a]);
                float scoreB = sanitizeSortValue(_primitiveImportance[b]);
                if (scoreA == scoreB)
                  return a < b;
                return scoreA > scoreB;
              });

    size_t minActive =
        std::min(primCount, _residencyConfig.energyMinActivePrimitives);

    if (_totalPrimitiveImportance <= 0.0f) {
      size_t enabledPrimitives = 0;
      for (size_t index : sortedIndices) {
        if (enabledPrimitives >= minActive)
          break;
        desiredState[index] = true;
        ++enabledPrimitives;
      }
    } else {
      float cumulative = 0.0f;
      float targetImportance =
          _totalPrimitiveImportance * _residencyConfig.energyTargetFraction;
      size_t enabled = 0;
      for (size_t index : sortedIndices) {
        if (enabled >= minActive && cumulative >= targetImportance)
          break;
        desiredState[index] = true;
        cumulative += std::max(_primitiveImportance[index], 0.0f);
        ++enabled;
      }

      for (size_t i = 0; i < minActive && i < sortedIndices.size(); ++i)
        desiredState[sortedIndices[i]] = true;
    }

    bool changed = false;
    size_t toggles = 0;
    for (size_t i = 0; i < primCount; ++i) {
      bool shouldBeActive = desiredState[i];
      if (shouldBeActive == _activePrimitive[i])
        continue;
      if (!forceAllToggles) {
        if (i < _primitiveCooldown.size() && _primitiveCooldown[i] > 0)
          continue;
        if (toggles >= _residencyConfig.energyMaxTogglesPerFrame)
          break;
      }
      if (setPrimitiveActive(i, shouldBeActive)) {
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

  std::vector<size_t> objectPrimitiveCounts(objectCount, 0);
  bool anyMeshGroups = false;
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    const SceneObject &obj = _allSceneObjects[objectIndex];
    size_t first = obj.firstPrimitive;
    size_t last = first + obj.primitiveCount;
    float totalImportance = 0.0f;
    size_t count = 0;
    float coverage = 0.0f;
    for (size_t prim = first; prim < last && prim < _primitiveImportance.size();
         ++prim) {
      totalImportance += std::max(_primitiveImportance[prim], 0.0f);
      if (applyVisibilityBoost && prim < _primitiveScreenCoverage.size())
        coverage += std::max(_primitiveScreenCoverage[prim], 0.0f);
      ++count;
    }
    if (applyVisibilityBoost) {
      float sphereCoverage = 0.0f;
      if (objectIndex < _objectBounds.size())
        sphereCoverage = coverageForSphere(_objectBounds[objectIndex]);
      float combinedCoverage = std::max(coverage, sphereCoverage);
      float visibilityFactor = combinedCoverage > 0.0f ? 1.0f : 0.0f;
      float multiplier = 1.0f + (visibilityBoost - 1.0f) * visibilityFactor;
      _objectImportance[objectIndex] = totalImportance * multiplier;
    } else {
      _objectImportance[objectIndex] = totalImportance;
    }
    objectPrimitiveCounts[objectIndex] = count;
    if (obj.meshGroupId >= 0)
      anyMeshGroups = true;
  }

  std::sort(_energySortedIndices.begin(), _energySortedIndices.end(),
            [this](size_t a, size_t b) {
              float scoreA = 0.0f;
              if (a < _objectImportance.size())
                scoreA = sanitizeSortValue(_objectImportance[a]);
              float scoreB = 0.0f;
              if (b < _objectImportance.size())
                scoreB = sanitizeSortValue(_objectImportance[b]);
              if (scoreA == scoreB)
                return a < b;
              return scoreA > scoreB;
            });

  const size_t primCount = _activePrimitive.size();
  const size_t minActivePrimitives =
      std::min(primCount, _residencyConfig.energyMinActivePrimitives);

  if (!anyMeshGroups) {
    std::vector<bool> desiredObjectState(objectCount, false);
    size_t primitivesEnabled = 0;

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
          _totalPrimitiveImportance * _residencyConfig.energyTargetFraction;
      for (size_t idx : _energySortedIndices) {
        if (idx >= objectPrimitiveCounts.size())
          continue;
        size_t count = objectPrimitiveCounts[idx];
        float importance =
            (idx < _objectImportance.size()) ? _objectImportance[idx] : 0.0f;
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
    for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
      bool shouldBeActive = desiredObjectState[objectIndex];
      bool currentlyActive =
          objectIndex < _objectActive.size() && _objectActive[objectIndex];
      if (shouldBeActive == currentlyActive)
        continue;

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
        continue;

      // Allow large object toggles to saturate the frame budget instead of
      // being rejected outright. We only stop once the current frame's budget
      // has already been consumed, matching the behaviour of other residency
      // passes.
      if (!forceAllToggles &&
          toggledPrimitiveCount >= _residencyConfig.energyMaxTogglesPerFrame)
        continue;

      size_t toggled = setObjectActive(objectIndex, shouldBeActive);
      if (toggled > 0) {
        toggledPrimitiveCount =
            std::min(toggledPrimitiveCount + toggled,
                     _residencyConfig.energyMaxTogglesPerFrame);
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
        if (setObjectActive(idx, true) > 0) {
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
        if (setObjectActive(fallbackObject, true) > 0)
          changed = true;
      } else {
        if (setPrimitiveActive(0, true))
          changed = true;
      }
    }

    return changed;
  }

  struct MeshGroupSummary {
    int meshId = -1;
    std::vector<size_t> objectIndices;
    float importance = 0.0f;
    size_t primitiveCount = 0;
  };

  std::vector<MeshGroupSummary> meshGroups;
  meshGroups.reserve(objectCount);
  std::unordered_map<int, size_t> meshToGroup;
  meshToGroup.reserve(objectCount);

  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    const SceneObject &obj = _allSceneObjects[objectIndex];
    int meshId = obj.meshGroupId;
    size_t groupIndex = std::numeric_limits<size_t>::max();
    if (meshId >= 0) {
      auto it = meshToGroup.find(meshId);
      if (it == meshToGroup.end()) {
        meshGroups.emplace_back();
        groupIndex = meshGroups.size() - 1;
        meshGroups[groupIndex].meshId = meshId;
        meshToGroup.emplace(meshId, groupIndex);
      } else {
        groupIndex = it->second;
      }
    } else {
      meshGroups.emplace_back();
      groupIndex = meshGroups.size() - 1;
      meshGroups[groupIndex].meshId = meshId;
    }

    MeshGroupSummary &group = meshGroups[groupIndex];
    group.objectIndices.push_back(objectIndex);
    group.importance += (objectIndex < _objectImportance.size())
                            ? _objectImportance[objectIndex]
                            : 0.0f;
    group.primitiveCount += (objectIndex < objectPrimitiveCounts.size())
                                ? objectPrimitiveCounts[objectIndex]
                                : size_t(0);
  }

  std::vector<float> meshGroupAverageImportance(meshGroups.size(), 0.0f);
  for (size_t groupIndex = 0; groupIndex < meshGroups.size(); ++groupIndex) {
    const MeshGroupSummary &group = meshGroups[groupIndex];
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
      const MeshGroupSummary &group = meshGroups[idx];
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
        _totalPrimitiveImportance * _residencyConfig.energyTargetFraction;
    for (size_t sortedPos = 0; sortedPos < meshSortedIndices.size();
         ++sortedPos) {
      size_t idx = meshSortedIndices[sortedPos];
      if (idx >= meshGroups.size())
        continue;
      const MeshGroupSummary &group = meshGroups[idx];
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
      const MeshGroupSummary &group = meshGroups[idx];
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
    // Expand the selection to include mesh siblings whose per-primitive
    // averages are comparable to the last group that satisfied the target.
    // The tolerance derives from the relative separation between the cutoff
    // group and its neighbours so groups with similar intensity ratios remain
    // together even when the absolute gap collapses toward zero.
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

    // Fall back to a gentle relative epsilon so identical neighbours are kept
    // together instead of being filtered out by a zero-width band.
    const float kRelativeFallback = 0.01f;
    allowedRatio = std::max(allowedRatio, kRelativeFallback);

    float allowedDelta = meshLastPrimaryAverage * allowedRatio;

    if (allowedDelta > 0.0f) {
      float epsilon = std::max(1e-5f, meshLastPrimaryAverage * 1e-3f);
      for (size_t groupIndex = 0; groupIndex < meshGroups.size();
           ++groupIndex) {
        if (desiredGroupState[groupIndex])
          continue;
        const MeshGroupSummary &group = meshGroups[groupIndex];
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
    for (size_t objectIndex : meshGroups[groupIndex].objectIndices) {
      if (objectIndex < desiredObjectState.size())
        desiredObjectState[objectIndex] = true;
    }
  }

  bool changed = false;
  size_t toggledPrimitiveCount = 0;
  for (size_t groupIndex = 0; groupIndex < meshGroups.size(); ++groupIndex) {
    const MeshGroupSummary &group = meshGroups[groupIndex];
    bool groupShouldBeActive = desiredGroupState[groupIndex];
    size_t groupToggleCount = 0;
    bool groupNeedsToggle = false;
    bool groupCanToggle = true;

    for (size_t objectIndex : group.objectIndices) {
      bool shouldBeActive =
          (objectIndex < desiredObjectState.size())
              ? desiredObjectState[objectIndex]
              : groupShouldBeActive;
      bool currentlyActive =
          (objectIndex < _objectActive.size()) ? _objectActive[objectIndex]
                                               : false;
      if (shouldBeActive == currentlyActive)
        continue;
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

    // Toggle all partitions belonging to the mesh together so the frame budget
    // is charged against the combined primitive count for the group, even when
    // the adaptive sibling pass activates several related meshes at once.
    if (!forceAllToggles &&
        toggledPrimitiveCount >= _residencyConfig.energyMaxTogglesPerFrame)
      continue;

    size_t toggledThisGroup = 0;
    for (size_t objectIndex : group.objectIndices) {
      bool shouldBeActive =
          (objectIndex < desiredObjectState.size())
              ? desiredObjectState[objectIndex]
              : groupShouldBeActive;
      bool currentlyActive =
          (objectIndex < _objectActive.size()) ? _objectActive[objectIndex]
                                               : false;
      if (shouldBeActive == currentlyActive)
        continue;
      toggledThisGroup += setObjectActive(objectIndex, shouldBeActive);
    }

    if (toggledThisGroup > 0) {
      toggledPrimitiveCount =
          std::min(toggledPrimitiveCount + toggledThisGroup,
                   _residencyConfig.energyMaxTogglesPerFrame);
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
      const MeshGroupSummary &group = meshGroups[idx];
      if (group.primitiveCount == 0)
        continue;
      size_t toggled = 0;
      for (size_t objectIndex : group.objectIndices)
        toggled += setObjectActive(objectIndex, true);
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
      for (size_t objectIndex : meshGroups[fallbackGroup].objectIndices)
        toggled += setObjectActive(objectIndex, true);
      if (toggled > 0)
        changed = true;
    } else {
      if (setPrimitiveActive(0, true))
        changed = true;
    }
  }

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

  const float screenArea = Camera::screenSize.x * Camera::screenSize.y;
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

  struct MeshGroupCoverage {
    int meshGroupId = -1;
    std::vector<size_t> objectIndices;
    float coverage = 0.0f;
    size_t primitiveCount = 0;
  };

  std::vector<MeshGroupCoverage> meshGroups;
  meshGroups.reserve(objectCount);
  std::unordered_map<int, size_t> meshToGroup;
  meshToGroup.reserve(objectCount);

  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    const SceneObject &obj = _allSceneObjects[objectIndex];
    size_t groupIndex = std::numeric_limits<size_t>::max();
    if (obj.meshGroupId >= 0) {
      auto it = meshToGroup.find(obj.meshGroupId);
      if (it == meshToGroup.end()) {
        meshGroups.emplace_back();
        groupIndex = meshGroups.size() - 1;
        meshGroups[groupIndex].meshGroupId = obj.meshGroupId;
        meshToGroup.emplace(obj.meshGroupId, groupIndex);
      } else {
        groupIndex = it->second;
      }
    } else {
      meshGroups.emplace_back();
      groupIndex = meshGroups.size() - 1;
      meshGroups[groupIndex].meshGroupId = obj.meshGroupId;
    }

    MeshGroupCoverage &group = meshGroups[groupIndex];
    group.objectIndices.push_back(objectIndex);
    group.coverage += objectCoverage[objectIndex];
    group.primitiveCount += objectPrimitiveTotals[objectIndex];
  }

  std::vector<size_t> sortedGroups(meshGroups.size());
  std::iota(sortedGroups.begin(), sortedGroups.end(), size_t(0));
  std::sort(sortedGroups.begin(), sortedGroups.end(),
            [&meshGroups](size_t a, size_t b) {
              float ca = (a < meshGroups.size())
                             ? sanitizeSortValue(meshGroups[a].coverage)
                             : 0.0f;
              float cb = (b < meshGroups.size())
                             ? sanitizeSortValue(meshGroups[b].coverage)
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
    const MeshGroupCoverage &group = meshGroups[groupIndex];
    size_t declaredPrimitiveCount = group.primitiveCount;
    if (declaredPrimitiveCount == 0)
      continue;

    float coverage = group.coverage;
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
    const MeshGroupCoverage &group = meshGroups[groupIndex];
    size_t declaredPrimitiveCount = group.primitiveCount;
    if (declaredPrimitiveCount == 0)
      continue;
    desiredGroupState[groupIndex] = true;
    primitivesEnabled += declaredPrimitiveCount;
    accumulatedCoverage += group.coverage;
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
    const MeshGroupCoverage &group = meshGroups[groupIndex];
    std::vector<size_t> objectsToToggle;
    objectsToToggle.reserve(group.objectIndices.size());
    bool canToggleGroup = true;

    for (size_t objectIndex : group.objectIndices) {
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
      const MeshGroupCoverage &group = meshGroups[groupIndex];
      if (group.primitiveCount == 0)
        continue;

      bool groupActivated = false;
      for (size_t objectIndex : group.objectIndices) {
        if (objectIndex >= _allSceneObjects.size())
          continue;
        if (setObjectActive(objectIndex, true) > 0) {
          changed = true;
          groupActivated = true;
        }
      }

      if (groupActivated) {
        fallbackPrimitives += group.primitiveCount;
        anyActivePrimitive = true;
      }

      if (anyActivePrimitive && fallbackPrimitives >= minActivePrimitives)
        break;
    }
  }

  return changed;
}

void Renderer::flushResidencyChanges(bool forceFullRebuild) {
  if (!forceFullRebuild && _recentlyActivated.empty() &&
      _recentlyDeactivated.empty()) {
    for (size_t objectIndex = 0;
         objectIndex < _residentObjectGpuResources.size(); ++objectIndex) {
      auto &resident = _residentObjectGpuResources[objectIndex];
      auto &record = _instanceRecords[objectIndex];
      bool pending = resident.pendingCommand &&
                     resident.pendingCommand->status() !=
                         MTL::CommandBufferStatusCompleted;
      if (resident.pendingCommand && !pending)
        resident.clearPendingCommand();

      if (!resident.isResident() && !pending) {
        resident.transitionToCold(record);
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

      const SceneObject &obj = _allSceneObjects[objectIndex];
      size_t first = obj.firstPrimitive;
      size_t last = first + obj.primitiveCount;
      bool anyActive = false;
      for (size_t prim = first; prim < last && prim < _activePrimitive.size();
           ++prim) {
        if (_activePrimitive[prim])
          anyActive = true;
      }

      bool newState = anyActive;
      bool fullyInactive = !anyActive;
      bool prevState = _objectActive[objectIndex];
      _objectActive[objectIndex] = newState;
      if (prevState != newState || fullyInactive)
        _objectCooldown[objectIndex] = _residencyConfig.stateCooldownFrames;
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

  if (!_pPrimitiveHitReadback)
    return;

  size_t bufferLength = _pPrimitiveHitReadback->length();
  uint32_t *hitPtr =
      static_cast<uint32_t *>(_pPrimitiveHitReadback->contents());
  if (!hitPtr)
    return;

  if (_rayHitCopyError) {
    std::memset(hitPtr, 0, bufferLength);
    _primitiveHitScores.clear();
    _primitiveHitLastFrame.clear();
    _rayHitCopyError = false;
    return;
  }

  if (!_pScene ||
      _pScene->getResidencyStrategy() != ResidencyStrategy::RayHitBudget) {
    std::memset(hitPtr, 0, bufferLength);
    _primitiveHitScores.clear();
    _primitiveHitLastFrame.clear();
    return;
  }

  size_t totalPrimitiveCount = _allPrimitives.size();
  if (totalPrimitiveCount == 0)
    return;

  size_t bufferCount = bufferLength / sizeof(uint32_t);
  size_t count = std::min(totalPrimitiveCount, bufferCount);
  if (count == 0)
    return;

  if (_primitiveHitScores.size() < totalPrimitiveCount)
    _primitiveHitScores.resize(totalPrimitiveCount, 0.0f);
  if (_primitiveHitLastFrame.size() < totalPrimitiveCount)
    _primitiveHitLastFrame.resize(totalPrimitiveCount, 0);

  parallelChunkedAsync(0, count, [&](size_t chunkStart, size_t chunkEnd) {
    for (size_t i = chunkStart; i < chunkEnd; ++i) {
      uint32_t hits = hitPtr[i];
      _primitiveHitLastFrame[i] = hits;
      _primitiveHitScores[i] =
          _primitiveHitScores[i] * _residencyConfig.rayHitDecay +
          static_cast<float>(hits);
      hitPtr[i] = 0;
    }
  });
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
    sample.textureMemoryCapMB = _textureResidencyMemoryCapMB;
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
    sample.overMemoryCap = sample.gpuMemoryMB > _textureResidencyMemoryCapMB;
    _pendingBenchmarkSamples.push_back(std::move(sample));
  }
}

void Renderer::completeFrameMetrics(MTL::CommandBuffer *pCmd) {
  auto cpuEnd = std::chrono::high_resolution_clock::now();
  _lastCPUTime = std::chrono::duration<double>(cpuEnd - _cpuStart).count();
  _lastGPUTime = pCmd->GPUEndTime() - pCmd->GPUStartTime();
  if (_lastGPUTime > 0.0) {
    _lastRaysPerSecond = static_cast<double>(_lastRayCount) / _lastGPUTime;
  } else {
    _lastRaysPerSecond = 0.0;
  }
  size_t offloaded = _totalNodeCount > _residentNodeCount ?
                         _totalNodeCount - _residentNodeCount :
                         0;
  printf("Resident nodes: %zu offloaded: %zu CPU: %.3f ms GPU: %.3f ms Rays/s: %.2f\n",
         _activeNodeCount, offloaded, _lastCPUTime * 1000.0,
         _lastGPUTime * 1000.0, _lastRaysPerSecond);

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
