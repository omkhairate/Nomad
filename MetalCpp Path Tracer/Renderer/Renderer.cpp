#include "Renderer.h"

#include "Camera.h"
#include "InputSystem.h"
#include "Scene.h"
#include "SceneLoader.h"
#include <Metal/MTLArgument.hpp>
#include <Metal/MTLComputePipeline.hpp>
#include <Metal/MTLAccelerationStructureTypes.hpp>
#include <algorithm>
#include <array>
#include <cassert>
#include <CoreFoundation/CoreFoundation.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <functional>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <numeric>
#include <simd/simd.h>
#include <string>
#include <utility>

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
  instanceRecord.triangleBase = 0;
  instanceRecord.triangleCount = 0;
  triangleBase = 0;
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
  instanceRecord.triangleBase = static_cast<uint32_t>(triangleBase);
  instanceRecord.triangleCount = static_cast<uint32_t>(triangleCount);
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

float luminance(const simd::float3 &c) {
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

float primitiveArea(const Primitive &p) {
  simd::float3 e1 = p.triangle.v1 - p.triangle.v0;
  simd::float3 e2 = p.triangle.v2 - p.triangle.v0;
  return 0.5f * simd::length(simd::cross(e1, e2));
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
  return (p.triangle.v0[axis] + p.triangle.v1[axis] + p.triangle.v2[axis]) /
         3.0f;
}

void primitiveBoundsLocal(const Primitive &p, simd::float3 &pMin,
                          simd::float3 &pMax) {
  const auto &t = p.triangle;
  pMin = simd::min(t.v0, simd::min(t.v1, t.v2));
  pMax = simd::max(t.v0, simd::max(t.v1, t.v2));
}

void markBufferModified(MTL::Buffer *buffer, NS::Range range) {
  if (!buffer)
    return;

  if (buffer->storageMode() == MTL::StorageModeManaged)
    buffer->didModifyRange(range);
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
  simd::float3 dir = toCenter / dist;
  float cosAngle = simd::dot(simd::normalize(Camera::forward), dir);
  if (cosAngle <= 0.0f)
    return false;
  float angle = acosf(cosAngle);
  float halfFov = (Camera::verticalFov * M_PI / 180.0f) * 0.5f;
  float radiusAngle = asinf(std::min(b.radius / dist, 1.0f));
  return angle <= halfFov + radiusAngle;
}

Renderer::Renderer(MTL::Device *pDevice)
    : _pDevice(pDevice->retain()), _pScene(new Scene()) {
  _pCommandQueue = _pDevice->newCommandQueue();
  _tlasHeap.initialize(_pDevice);
  _dummyBlasResources.initialize(_pDevice);

  Camera::reset();

  updateVisibleScene();
  buildShaders();
  buildTextures();

  recalculateViewport();
}

Renderer::~Renderer() {
  if (_pPrimitiveBuffer)
    _pPrimitiveBuffer->release();
  if (_pMaterialBuffer)
    _pMaterialBuffer->release();
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

  delete _pScene;
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
        std::filesystem::path(__FILE__).parent_path() / "../scene_distance_lod_sweep.xml";
    SceneLoader::LoadSceneFromXML(alt.string(), _pScene);
  }

  Camera::screenSize = _pScene->screenSize;

  if (!_pScene->cameraPath.empty()) {
    const auto &k = _pScene->cameraPath.front();
    Camera::position = k.position;
    Camera::forward = simd::normalize(k.lookAt - k.position);
    Camera::up = {0, 1, 0};
  }

  printf("Scene loaded: %zu total primitives (triangles only)\n",
         _pScene->getPrimitiveCount());

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
  _energySortedIndices.resize(primCount);
  _primitiveHitScores.assign(primCount, 0.0f);
  _primitiveHitLastFrame.assign(primCount, 0);
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
    simd::float3 c = (p.triangle.v0 + p.triangle.v1 + p.triangle.v2) / 3.0f;
    float r = simd::length(p.triangle.v0 - c);
    r = std::max(r, (float)simd::length(p.triangle.v1 - c));
    r = std::max(r, (float)simd::length(p.triangle.v2 - c));
    _primitiveBounds[i] = {c, r};
    _primitiveImportance[i] = primitiveImportance(p);
    _energySortedIndices[i] = i;
    if (i < _rayHitSortedIndices.size())
      _rayHitSortedIndices[i] = i;
    if (i < _screenCoverageSortedIndices.size())
      _screenCoverageSortedIndices[i] = i;
    _totalPrimitiveImportance += std::max(_primitiveImportance[i], 0.0f);
  }

  std::sort(_energySortedIndices.begin(), _energySortedIndices.end(),
            [this](size_t a, size_t b) {
              float scoreA = sanitizeSortValue(_primitiveImportance[a]);
              float scoreB = sanitizeSortValue(_primitiveImportance[b]);
              if (scoreA == scoreB)
                return a < b;
              return scoreA > scoreB;
            });

  size_t hitCount = std::max<size_t>(_maxPrimitiveCount, 1);
  size_t hitBytes = hitCount * sizeof(uint32_t);
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
    simd::float3 v0 = p.triangle.v0;
    simd::float3 v1 = p.triangle.v1;
    simd::float3 v2 = p.triangle.v2;

    vertices.push_back(v0);
    vertices.push_back(v1);
    vertices.push_back(v2);
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
      triangleCount > 0
          ? MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()
                ->init()
          : nullptr;
  if (geometryDesc)
    geometryDesc->setOpaque(true);

  std::string blasLabel = "ObjectBLAS_" + std::to_string(objectIndex);
  std::string vertexLabel = "ObjectVertices_" + std::to_string(objectIndex);
  std::string indexLabel = "ObjectIndices_" + std::to_string(objectIndex);

  auto releaseGeometryDescriptors = [&]() {
    if (geometryDesc) {
      geometryDesc->release();
      geometryDesc = nullptr;
    }
  };

  std::vector<NS::Object *> descriptorObjects;
  if (geometryDesc)
    descriptorObjects.push_back(geometryDesc);

  if (descriptorObjects.empty()) {
    releaseGeometryDescriptors();
    cleanupPool();
    return false;
  }

  NS::Array *geometryArray =
      NS::Array::alloc()->init(descriptorObjects.data(), descriptorObjects.size());

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

    if (vertexBytes > 0) {
      vertexBuffer = resident.resources.ensureVertexBuffer(
          vertexBytes, vertexLabel.c_str());
      if (!vertexBuffer)
        return false;
    } else {
      vertexBuffer = nullptr;
    }

    if (indexBytes > 0) {
      indexBuffer = resident.resources.ensureIndexBuffer(indexBytes,
                                                        indexLabel.c_str());
      if (!indexBuffer)
        return false;
    } else {
      indexBuffer = nullptr;
    }

    return true;
  };

  if (!requestGeometryBuffers()) {
    if (geometryArray)
      geometryArray->release();
    releaseGeometryDescriptors();
    accelDesc->release();
    cleanupPool();
    return false;
  }

  auto configureGeometryDescriptors = [&]() {
    if (geometryDesc) {
      geometryDesc->setVertexBuffer(vertexBuffer);
      geometryDesc->setVertexBufferOffset(0);
      geometryDesc->setVertexStride(sizeof(simd::float3));
      geometryDesc->setVertexFormat(
          MTL::AttributeFormat::AttributeFormatFloat3);
      geometryDesc->setIndexBuffer(indexBuffer);
      geometryDesc->setIndexBufferOffset(0);
      geometryDesc->setIndexType(MTL::IndexType::IndexTypeUInt32);
      geometryDesc->setTriangleCount(triangleCount);
    }
  };

  NS::UInteger totalHeapBytes = 0;
  NS::UInteger alignedAccelerationSize = 0;
  MTL::AccelerationStructureSizes sizes{};
  MTL::SizeAndAlign heapAlign{};

  while (true) {
    configureGeometryDescriptors();

    sizes = _pDevice->accelerationStructureSizes(accelDesc);
    heapAlign = _pDevice->heapAccelerationStructureSizeAndAlign(accelDesc);

    alignedAccelerationSize = static_cast<NS::UInteger>(
        alignTo(static_cast<size_t>(heapAlign.size),
                static_cast<size_t>(heapAlign.align)));
    alignedAccelerationSize =
        resident.resources.alignedHeapSize(alignedAccelerationSize);

    NS::UInteger requiredTotal =
        alignedAccelerationSize + alignedVertexSize + alignedIndexSize;

    if (requiredTotal > totalHeapBytes) {
      totalHeapBytes = requiredTotal;
      resident.resources.ensureHeapCapacity(totalHeapBytes);

      if (!requestGeometryBuffers()) {
        if (geometryArray)
          geometryArray->release();
        releaseGeometryDescriptors();
        accelDesc->release();
        cleanupPool();
        return false;
      }
      continue;
    }

    totalHeapBytes = requiredTotal;
    break;
  }

  configureGeometryDescriptors();

  MTL::AccelerationStructure *accelerationStructure =
      resident.resources.ensureAccelerationStructure(
          sizes.accelerationStructureSize, blasLabel.c_str());

  if (!accelerationStructure) {
    if (geometryArray)
      geometryArray->release();
    releaseGeometryDescriptors();
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
    releaseGeometryDescriptors();
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
    releaseGeometryDescriptors();
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
    releaseGeometryDescriptors();
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
  releaseGeometryDescriptors();
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

      primBase[0] =
          simd::make_float4(p.triangle.v0, static_cast<float>(p.type));
      primBase[1] = simd::make_float4(p.triangle.v1, 0.0f);
      primBase[2] = simd::make_float4(p.triangle.v2, 0.0f);

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

  size_t triangleLeafCursor = 0;

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
      record.triangleBase = static_cast<uint32_t>(triangleLeafCursor);
      bool anyActive = false;
      size_t first = obj.firstPrimitive;
      size_t last = first + obj.primitiveCount;
      size_t objectTriangleCount = 0;
      for (size_t prim = first; prim < last && prim < _allPrimitives.size(); ++prim) {
        ++objectTriangleCount;
        if (prim < _activePrimitive.size() && _activePrimitive[prim])
          anyActive = true;
      }
      record.triangleCount = static_cast<uint32_t>(objectTriangleCount);
      record.blasRootIndex = anyActive ? obj.blasRootIndex : -1;
      objectShouldBeResident[objectIndex] = anyActive;
      _instanceRecords[objectIndex] = record;
      triangleLeafCursor += objectTriangleCount;
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
      record.triangleBase = static_cast<uint32_t>(triangleLeafCursor);
      size_t objectTriangleCount = 0;

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
          simd::float4 prim0 =
              simd::make_float4(p.triangle.v0, static_cast<float>(p.type));
          simd::float4 prim1 = simd::make_float4(p.triangle.v1, 0.0f);
          simd::float4 prim2 = simd::make_float4(p.triangle.v2, 0.0f);
          size_t baseVertex = compactTriangleVertices.size();
          compactTriangleVertices.push_back(p.triangle.v0);
          compactTriangleVertices.push_back(p.triangle.v1);
          compactTriangleVertices.push_back(p.triangle.v2);
          compactTriangleIndices.push_back(
              simd::make_uint3(static_cast<uint32_t>(baseVertex),
                                static_cast<uint32_t>(baseVertex + 1),
                                static_cast<uint32_t>(baseVertex + 2)));
          ++objectTriangleCount;

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
        record.triangleCount = static_cast<uint32_t>(objectTriangleCount);
        if (activeCount == 0) {
          for (const auto &edit : residentIndexEdits)
            if (edit.first < _primitiveToResidentIndex.size())
              _primitiveToResidentIndex[edit.first] = edit.second;
          objectShouldBeResident[objectIndex] = false;
          _instanceRecords[objectIndex] = record;
          objectTriangleCount = 0;
          ++blasCacheSkipped;
          continue;
        }

        size_t nodeBase = compactBVHNodes.size() / 2;
        int32_t cachedRootIndex =
            static_cast<int32_t>(nodeBase + obj.cachedBlasRootIndex);

        bool encounteredEmptyLeaf = false;
        std::vector<simd::float4> rebuiltNodes;
        rebuiltNodes.reserve(obj.cachedBlasNodes.size() * 2);

        for (const BVHNode &node : obj.cachedBlasNodes) {
          BVHNode adjusted = node;
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
            if (newCount == 0) {
              encounteredEmptyLeaf = true;
              break;
            }
            if (newLeftFirst < 0)
              newLeftFirst = static_cast<int>(record.primitiveIndexBase);
            assert(newCount > 0 &&
                   "Cached BLAS leaf remapped to zero primitives");
            adjusted.leftFirst = newLeftFirst;
            adjusted.count = newCount;
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
          rebuiltNodes.push_back(simd::make_float4(adjusted.boundsMin, leftBits));
          rebuiltNodes.push_back(simd::make_float4(adjusted.boundsMax, rightBits));
        }

        if (!encounteredEmptyLeaf) {
          record.blasRootIndex = cachedRootIndex;
          compactBVHNodes.insert(compactBVHNodes.end(), rebuiltNodes.begin(),
                                 rebuiltNodes.end());
          objectShouldBeResident[objectIndex] = true;
          _instanceRecords[objectIndex] = record;
          triangleLeafCursor += objectTriangleCount;
          ++blasCacheReused;
          assert(!encounteredEmptyLeaf &&
                 "Cached BLAS leaf with zero primitives reached upload");
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
        objectTriangleCount = 0;
        record.triangleCount = 0;
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
        record.triangleCount = 0;
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
        simd::float4 prim0 =
            simd::make_float4(p.triangle.v0, static_cast<float>(p.type));
        simd::float4 prim1 = simd::make_float4(p.triangle.v1, 0.0f);
        simd::float4 prim2 = simd::make_float4(p.triangle.v2, 0.0f);
        size_t baseVertex = compactTriangleVertices.size();
        compactTriangleVertices.push_back(p.triangle.v0);
        compactTriangleVertices.push_back(p.triangle.v1);
        compactTriangleVertices.push_back(p.triangle.v2);
        compactTriangleIndices.push_back(
            simd::make_uint3(static_cast<uint32_t>(baseVertex),
                              static_cast<uint32_t>(baseVertex + 1),
                              static_cast<uint32_t>(baseVertex + 2)));
        ++objectTriangleCount;

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
      record.triangleCount = static_cast<uint32_t>(objectTriangleCount);
      objectShouldBeResident[objectIndex] = record.primitiveCount > 0;
      _instanceRecords[objectIndex] = record;
      if (record.primitiveCount > 0) {
        triangleLeafCursor += objectTriangleCount;
      }
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

    gpuResident.triangleBase = instanceRecord.triangleBase;

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
      desc.intersectionFunctionTableOffset = 0;
      handle.triangleBase = _instanceRecords[objectIndex].triangleBase;
      handle.triangleCount = _instanceRecords[objectIndex].triangleCount;
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
    ensureBufferCapacity(_pPrimitiveBuffer, primitiveBytes,
                         _primitiveBufferCapacity, allowShrink);
    if (_pPrimitiveBuffer) {
      simd::float4 *dst =
          static_cast<simd::float4 *>(_pPrimitiveBuffer->contents());
      if (primitiveFloat4Count > 0)
        std::memcpy(dst, primitiveSource->data(),
                    primitiveFloat4Count * sizeof(simd::float4));
      else
        dst[0] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
      markBufferModified(_pPrimitiveBuffer,
                         NS::Range::Make(0, primitiveBytes));
    }

    size_t materialFloat4Count = materialSource->size();
    size_t materialBytes =
        std::max<size_t>(materialFloat4Count, size_t(1)) *
        sizeof(simd::float4);
    ensureBufferCapacity(_pMaterialBuffer, materialBytes,
                         _materialBufferCapacity, allowShrink);
    if (_pMaterialBuffer) {
      simd::float4 *dst =
          static_cast<simd::float4 *>(_pMaterialBuffer->contents());
      if (materialFloat4Count > 0)
        std::memcpy(dst, materialSource->data(),
                    materialFloat4Count * sizeof(simd::float4));
      else {
        dst[0] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (materialBytes >= 2 * sizeof(simd::float4))
          dst[1] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
      markBufferModified(_pMaterialBuffer,
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

bool Renderer::updateCamera() {
  const auto &path = _pScene->cameraPath;

  // If the scene defines a camera path, advance along it every render pass
  if (!path.empty()) {
    if (_animationFrame <= path.front().frame) {
      Camera::position = path.front().position;
      simd::float3 look = path.front().lookAt;
      Camera::forward = simd::normalize(look - Camera::position);
    } else if (_animationFrame >= path.back().frame) {
      Camera::position = path.back().position;
      simd::float3 look = path.back().lookAt;
      Camera::forward = simd::normalize(look - Camera::position);
    } else {
      for (size_t i = 0; i + 1 < path.size(); ++i) {
        const auto &k0 = path[i];
        const auto &k1 = path[i + 1];
        if (_animationFrame >= k0.frame && _animationFrame <= k1.frame) {
          float t =
              float(_animationFrame - k0.frame) / float(k1.frame - k0.frame);
          Camera::position = k0.position + t * (k1.position - k0.position);
          simd::float3 look = k0.lookAt + t * (k1.lookAt - k0.lookAt);
          Camera::forward = simd::normalize(look - Camera::position);
          break;
        }
      }
    }

    Camera::up = {0, 1, 0};
    // Reset the frame delta while scripted motion drives the camera so the
    // timeline remains frame-locked.
    _deltaTimeSeconds = 0.0;
    Camera::deltaTime = 0.0f;
    InputSystem::clearInputs();

    // Update view dependent data for the new camera transform
    recalculateViewport();

    // Move to the next keyframe for the following frame
    _animationFrame++;
    return true;
  }

  // Fall back to interactive controls when no keyframes are present
  bool changed = Camera::transformWithInputs();
  if (changed) {
    recalculateViewport();
  }
  return changed;
}

void Renderer::updateUniforms() {
  UniformsData &u = *((UniformsData *)_pUniformsBuffer->contents());

  bool cameraChanged = updateCamera();
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
  updateResidency();
  updateUniforms();
  beginFrameMetrics();
  std::swap(_accumulationSlots[0], _accumulationSlots[1]);

  NS::AutoreleasePool *pPool = NS::AutoreleasePool::alloc()->init();

  MTL::CommandBuffer *pCmd = _pCommandQueue->commandBuffer();
  if (!pCmd) {
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

      if (useAccelerationStructureLayout)
        pCompute->setAccelerationStructure(_pTlasStructure, 0);
      else
        pCompute->setAccelerationStructure(nullptr, 0);

      pCompute->setBuffer(useAccelerationStructureLayout ? _pGeometryHandleBuffer
                                                         : nullptr,
                          0, 1);
      pCompute->setBuffer(_pPrimitiveBuffer, 0, 2);
      pCompute->setBuffer(_pMaterialBuffer, 0, 3);
      pCompute->setBuffer(_pUniformsBuffer, 0, 4);
      pCompute->setBuffer(_pActiveBuffer, 0, 5);
      pCompute->setBuffer(_pLightIndexBuffer, 0, 6);
      pCompute->setBuffer(_pLightCdfBuffer, 0, 7);
      pCompute->setBuffer(_pPrimitiveRemapBuffer, 0, 8);
      pCompute->setBuffer(_pPrimitiveHitBufferGPU, 0, 9);
      pCompute->setBuffer(_pInstanceBuffer, 0, 10);
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

  pEnc->endEncoding();

  MTL::BlitCommandEncoder *pBlit = pCmd->blitCommandEncoder();
  if (_pPrimitiveHitBufferGPU && _pPrimitiveHitReadback) {
    size_t bytes =
        std::min(_pPrimitiveHitBufferGPU->length(),
                 _pPrimitiveHitReadback->length());
    if (bytes > 0) {
      pBlit->copyFromBuffer(_pPrimitiveHitBufferGPU, 0, _pPrimitiveHitReadback, 0,
                            bytes);
      pBlit->fillBuffer(_pPrimitiveHitBufferGPU, NS::Range::Make(0, bytes), 0);
    }
  }
  pBlit->endEncoding();

  pCmd->presentDrawable(pView->currentDrawable());
  pCmd->commit();

  pPool->release();
}

void Renderer::updateResidency(bool forceAllToggles, bool forceFullRebuild) {
  if (!_pScene)
    return;

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
  case ResidencyStrategy::DistanceLOD:
  default:
    changed = updateLODByDistance(forceAllToggles);
    break;
  }

  if (changed || forceFullRebuild)
    flushResidencyChanges(forceFullRebuild);
}

bool Renderer::updateLODByDistance(bool forceAllToggles) {
  // Use hysteresis so objects do not flicker when hovering near the activation
  // boundary. Inactive objects only become active once the camera is closer
  // than LOD_ENTER_DISTANCE, while active objects stay active until the camera
  // has moved beyond LOD_EXIT_DISTANCE.

  size_t toggles = 0;
  bool changed = false;

  const size_t objectCount = _allSceneObjects.size();
  std::vector<float> objectDistances(objectCount,
                                     std::numeric_limits<float>::max());
  std::vector<size_t> sortedIndices(objectCount);
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    const BoundingSphere &sphere =
        (objectIndex < _objectBounds.size())
            ? _objectBounds[objectIndex]
            : BoundingSphere{simd::make_float3(0.0f, 0.0f, 0.0f), 0.0f};
    float dist = simd::length(sphere.center - Camera::position) - sphere.radius;
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
    bool shouldBeActive =
        currentlyActive ? dist <= _residencyConfig.lodExitDistance
                         : dist < _residencyConfig.lodEnterDistance;
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

  if (toggled > 0 || prevState != newState || fullyInactive)
    _objectCooldown[objectIndex] = _residencyConfig.stateCooldownFrames;

  return toggled;
}

bool Renderer::updateEnergyImportance(bool forceAllToggles) {
  if (_activePrimitive.empty())
    return false;

  const size_t primCount = _activePrimitive.size();
  std::vector<bool> desiredState(primCount, false);
  size_t minActive =
      std::min(primCount, _residencyConfig.energyMinActivePrimitives);

  if (_totalPrimitiveImportance <= 0.0f) {
    for (size_t i = 0; i < minActive && i < _energySortedIndices.size(); ++i)
      desiredState[_energySortedIndices[i]] = true;
  } else {
    float cumulative = 0.0f;
    float targetImportance =
        _totalPrimitiveImportance * _residencyConfig.energyTargetFraction;
    size_t enabled = 0;
    for (size_t rank = 0; rank < _energySortedIndices.size(); ++rank) {
      size_t index = _energySortedIndices[rank];
      if (enabled >= minActive && cumulative >= targetImportance)
        break;
      desiredState[index] = true;
      cumulative += std::max(_primitiveImportance[index], 0.0f);
      ++enabled;
    }

    for (size_t i = 0; i < minActive && i < _energySortedIndices.size(); ++i)
      desiredState[_energySortedIndices[i]] = true;
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
    size_t fallback = !_energySortedIndices.empty() ? _energySortedIndices.front()
                                                    : size_t(0);
    if (setPrimitiveActive(fallback, true))
      changed = true;
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

  const auto &hitScores = _primitiveHitScoresSnapshot;

  std::sort(_rayHitSortedIndices.begin(), _rayHitSortedIndices.end(),
            [&hitScores](size_t a, size_t b) {
              float rawA = (a < hitScores.size()) ? hitScores[a] : 0.0f;
              float rawB = (b < hitScores.size()) ? hitScores[b] : 0.0f;
              float scoreA = sanitizeSortValue(rawA);
              float scoreB = sanitizeSortValue(rawB);
              if (scoreA == scoreB)
                return a < b;
              return scoreA > scoreB;
            });

  const size_t minActive =
      std::min(primCount, _residencyConfig.rayHitMinActivePrimitives);
  size_t targetActive = static_cast<size_t>(
      std::ceil(primCount * _residencyConfig.rayHitTargetFraction));
  targetActive = std::max(targetActive, minActive);
  targetActive = std::min(targetActive, primCount);

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

  for (size_t i = 0; i < primCount; ++i) {
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
          float radiusPixels =
              (b.radius / depth) / tanHalfFov * (Camera::screenSize.y * 0.5f);
          radiusPixels = std::max(radiusPixels, 0.0f);
          float area = static_cast<float>(M_PI) * radiusPixels * radiusPixels;
          float angleFactor = std::max(cosAngle, 0.0f);
          coverage = std::min(area * angleFactor, screenArea);
        }
      }
    }
    _primitiveScreenCoverage[i] = coverage;
  }

  std::sort(_screenCoverageSortedIndices.begin(),
            _screenCoverageSortedIndices.end(), [this](size_t a, size_t b) {
              float rawA = (a < _primitiveScreenCoverage.size())
                               ? _primitiveScreenCoverage[a]
                               : 0.0f;
              float rawB = (b < _primitiveScreenCoverage.size())
                               ? _primitiveScreenCoverage[b]
                               : 0.0f;
              float ca = sanitizeSortValue(rawA);
              float cb = sanitizeSortValue(rawB);
              if (ca == cb)
                return a < b;
              return ca > cb;
            });

  std::vector<bool> desired(primCount, false);
  const size_t minActive =
      std::min(primCount, _residencyConfig.screenFootprintMinActivePrimitives);
  float accumulated = 0.0f;
  size_t enabled = 0;
  float targetCoverage =
      screenArea * _residencyConfig.screenFootprintTargetFraction;
  for (size_t idx : _screenCoverageSortedIndices) {
    if (idx >= _primitiveScreenCoverage.size())
      continue;
    float coverage = _primitiveScreenCoverage[idx];
    if (enabled >= minActive && accumulated >= targetCoverage)
      break;
    if (enabled >= minActive &&
        coverage < _residencyConfig.screenFootprintMinPixelCoverage)
      break;
    if (coverage <= 0.0f && enabled >= minActive)
      break;
    desired[idx] = true;
    accumulated += coverage;
    ++enabled;
  }

  for (size_t i = 0; i < minActive && i < _screenCoverageSortedIndices.size(); ++i)
    desired[_screenCoverageSortedIndices[i]] = true;

  size_t toggles = 0;
  bool changed = false;
  for (size_t i = 0; i < primCount; ++i) {
    bool shouldBeActive = desired[i];
    if (shouldBeActive == _activePrimitive[i])
      continue;
    if (!forceAllToggles) {
      if (i < _primitiveCooldown.size() && _primitiveCooldown[i] > 0)
        continue;
      if (toggles >= _residencyConfig.screenFootprintMaxTogglesPerFrame)
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
    size_t fallback = !_screenCoverageSortedIndices.empty()
                          ? _screenCoverageSortedIndices.front()
                          : size_t(0);
    if (setPrimitiveActive(fallback, true))
      changed = true;
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

void Renderer::processRayHitCounters() {
  if (!_pPrimitiveHitReadback)
    return;

  size_t bufferLength = _pPrimitiveHitReadback->length();
  uint32_t *hitPtr =
      static_cast<uint32_t *>(_pPrimitiveHitReadback->contents());
  if (!hitPtr)
    return;

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

  for (size_t i = 0; i < count; ++i) {
    uint32_t hits = hitPtr[i];
    _primitiveHitLastFrame[i] = hits;
    _primitiveHitScores[i] =
        _primitiveHitScores[i] * _residencyConfig.rayHitDecay +
        static_cast<float>(hits);
    hitPtr[i] = 0;
  }
}

void Renderer::beginFrameMetrics() {
  _cpuStart = std::chrono::high_resolution_clock::now();
  _lastRayCount = static_cast<size_t>(Camera::screenSize.x * Camera::screenSize.y);
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
}

double Renderer::lastCPUTime() const { return _lastCPUTime; }
double Renderer::lastGPUTime() const { return _lastGPUTime; }
double Renderer::lastRaysPerSecond() const { return _lastRaysPerSecond; }
size_t Renderer::activeNodeCount() const { return _activeNodeCount; }
size_t Renderer::residentNodeCount() const { return _residentNodeCount; }
size_t Renderer::totalNodeCount() const { return _totalNodeCount; }
