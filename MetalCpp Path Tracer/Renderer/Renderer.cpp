#include "Renderer.h"

#include "Camera.h"
#include "InputSystem.h"
#include "Scene.h"
#include "SceneLoader.h"
#include <algorithm>
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
  uint32_t padding0 = 0;
  uint32_t padding1 = 0;
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

float luminance(const simd::float3 &c) {
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
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

  Camera::reset();

  updateVisibleScene();
  buildShaders();
  buildTextures();

  recalculateViewport();
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
  if (_pPrimitiveHitBufferGPU)
    _pPrimitiveHitBufferGPU->release();
  if (_pPrimitiveHitReadback)
    _pPrimitiveHitReadback->release();
  if (_pLightIndexBuffer)
    _pLightIndexBuffer->release();
  if (_pLightCdfBuffer)
    _pLightCdfBuffer->release();

  for (int i = 0; i < 2; i++)
    if (_accumulationTargets[i])
      _accumulationTargets[i]->release();

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

  NS::Error *pError = nullptr;
  MTL::Library *pLibrary = _pDevice->newDefaultLibrary();

  if (!pLibrary) {
    __builtin_printf("Failed to load Metal library\n");
    assert(false);
  }

  MTL::Function *pVertexFn = pLibrary->newFunction(
      NS::String::string("vertexMain", UTF8StringEncoding));
  MTL::Function *pFragFn = pLibrary->newFunction(
      NS::String::string("fragmentMain", UTF8StringEncoding));

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

  pVertexFn->release();
  pFragFn->release();
  pDesc->release();
  pLibrary->release();
}

void Renderer::updateVisibleScene() {
  if (!SceneLoader::LoadSceneFromXML("scene.xml", _pScene)) {
    std::filesystem::path alt =
        std::filesystem::path(__FILE__).parent_path() / "../scene.xml";
    SceneLoader::LoadSceneFromXML(alt.string(), _pScene);
  }

  Camera::screenSize = _pScene->screenSize;

  if (!_pScene->cameraPath.empty()) {
    const auto &k = _pScene->cameraPath.front();
    Camera::position = k.position;
    Camera::forward = simd::normalize(k.lookAt - k.position);
    Camera::up = {0, 1, 0};
  }

  printf("Scene loaded: %zu total primitives (%zu spheres, %zu triangles, %zu "
         "rectangles)\n",
         _pScene->getPrimitiveCount(), _pScene->getSphereCount(),
         _pScene->getTriangleCount(), _pScene->getRectangleCount());

  _residencyConfig = _pScene->getResidencyParameters();

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
  _allSceneObjects = _pScene->getObjects();
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

  size_t objectCount = _allSceneObjects.size();
  _objectBounds.resize(objectCount);
  _objectActive.assign(objectCount, false);
  _objectCooldown.assign(objectCount, 0);

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
    _energySortedIndices[i] = i;
    if (i < _rayHitSortedIndices.size())
      _rayHitSortedIndices[i] = i;
    if (i < _screenCoverageSortedIndices.size())
      _screenCoverageSortedIndices[i] = i;
    _totalPrimitiveImportance += std::max(_primitiveImportance[i], 0.0f);
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

  _pUniformsBuffer->didModifyRange(NS::Range::Make(0, sizeof(UniformsData)));

}

void Renderer::rebuildResidentResources(bool forceFullRebuild) {
  size_t totalPrimitiveCount = _allPrimitives.size();
  size_t cachedTotalPrimitiveCount = _cachedTotalPrimitiveCount;

  if (forceFullRebuild) {
    _residentCompacted = false;
    _compactionCooldown = 0;
  }

  const size_t uniformsDataSize = sizeof(UniformsData);
  if (!_pUniformsBuffer) {
    _pUniformsBuffer =
        _pDevice->newBuffer(uniformsDataSize, MTL::ResourceStorageModeManaged);
    if (_pUniformsBuffer)
      _pUniformsBuffer->didModifyRange(
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
  std::vector<simd::float4> compactTLASNodes;
  std::vector<simd::float3> compactTriangleVertices;
  std::vector<simd::uint3> compactTriangleIndices;
  std::vector<BVHNode> compactBVHStructs;

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
  } else {
    size_t residentPrimitiveCount = _activePrimitiveCount;
    remapUpload.resize(residentPrimitiveCount);
    compactPrimitiveData.assign(residentPrimitiveCount * 3,
                                simd::float4{0.0f, 0.0f, 0.0f, 0.0f});
    compactMaterialData.assign(residentPrimitiveCount * 2,
                               simd::float4{0.0f, 0.0f, 0.0f, 0.0f});
    compactPrimitiveIndices.resize(residentPrimitiveCount);

    std::vector<Primitive> compactPrimitives;
    compactPrimitives.reserve(residentPrimitiveCount);

    size_t triangleVertexBase = 0;
    for (size_t localIndex = 0; localIndex < residentPrimitiveCount;
         ++localIndex) {
      size_t globalIndex = activeIndices[localIndex];
      remapUpload[localIndex] = static_cast<uint32_t>(globalIndex);
      if (globalIndex < _primitiveToResidentIndex.size())
        _primitiveToResidentIndex[globalIndex] =
            static_cast<int32_t>(localIndex);

      const Primitive &p = _allPrimitives[globalIndex];
      compactPrimitives.push_back(p);
      compactPrimitiveIndices[localIndex] = static_cast<int>(localIndex);

      simd::float4 *primBase = &compactPrimitiveData[3 * localIndex];
      simd::float4 *matBase = &compactMaterialData[2 * localIndex];

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
        compactTriangleVertices.push_back(p.triangle.v0);
        compactTriangleVertices.push_back(p.triangle.v1);
        compactTriangleVertices.push_back(p.triangle.v2);
        compactTriangleIndices.push_back(simd::make_uint3(
            static_cast<uint32_t>(triangleVertexBase),
            static_cast<uint32_t>(triangleVertexBase + 1),
            static_cast<uint32_t>(triangleVertexBase + 2)));
        triangleVertexBase += 3;
        break;
      }
      }

      const Material &m = p.material;
      matBase[0] = simd::make_float4(m.albedo, m.materialType);
      matBase[1] = simd::make_float4(m.emissionColor, m.emissionPower);
    }

    size_t compactBlasCount = 0;
    simd::float4 *compactBVHRaw = _pScene->createBVHBuffer(
        compactPrimitives, compactPrimitiveIndices, compactBlasCount,
        compactBVHStructs);
    if (compactBVHRaw && compactBlasCount > 0) {
      compactBVHNodes.assign(compactBVHRaw,
                             compactBVHRaw + compactBlasCount * 2);
      delete[] compactBVHRaw;
    } else {
      compactBVHNodes.clear();
      compactBVHStructs.clear();
    }

    size_t compactTlasCount = 0;
    simd::float4 *compactTLASRaw = _pScene->createTLASBuffer(
        compactTlasCount, compactPrimitives, compactBVHStructs);
    if (compactTLASRaw && compactTlasCount > 0) {
      compactTLASNodes.assign(compactTLASRaw,
                              compactTLASRaw + compactTlasCount * 2);
      delete[] compactTLASRaw;
    } else {
      compactTLASNodes.clear();
    }

    compactActiveMask.assign(residentPrimitiveCount, 1);

    primitiveSource = &compactPrimitiveData;
    materialSource = &compactMaterialData;
    primitiveIndexSource = &compactPrimitiveIndices;
    bvhSource = &compactBVHNodes;
    tlasSource = &compactTLASNodes;
    triangleVertexSource = &compactTriangleVertices;
    triangleIndexSource = &compactTriangleIndices;

    _residentPrimitiveCount = residentPrimitiveCount;
    _residentTriangleCount = compactTriangleIndices.size();
    _blasNodeCount = compactBVHNodes.size() / 2;
    _tlasNodeCount = compactTLASNodes.size() / 2;

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
      _pSphereBuffer->didModifyRange(NS::Range::Make(0, primitiveBytes));
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
      _pSphereMaterialBuffer->didModifyRange(NS::Range::Make(0, materialBytes));
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
      _pPrimitiveIndexBuffer->didModifyRange(
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
      _pBVHBuffer->didModifyRange(NS::Range::Make(0, bvhBytes));
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
      _pTLASBuffer->didModifyRange(NS::Range::Make(0, tlasBytes));
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
      _pPrimitiveRemapBuffer->didModifyRange(
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
      _pTriangleVertexBuffer->didModifyRange(NS::Range::Make(0, vertexBytes));
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
      _pTriangleIndexBuffer->didModifyRange(NS::Range::Make(0, indexBytes));
    }

    _residentBuffersInitialized = true;
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
      _pActiveBuffer->didModifyRange(NS::Range::Make(0, activeBytes));
    } else if (uploadAll) {
      if (totalPrimitiveCount > 0)
        std::memcpy(activePtr, _cpuActiveMask.data(),
                    totalPrimitiveCount * sizeof(uint8_t));
      else
        activePtr[0] = 0;
      _pActiveBuffer->didModifyRange(NS::Range::Make(0, activeBytes));
    } else {
      auto updateMask = [&](const std::vector<size_t> &indices) {
        for (size_t idx : indices) {
          if (idx >= totalPrimitiveCount)
            continue;
          bool active = idx < _activePrimitive.size() && _activePrimitive[idx];
          uint8_t value = active ? 1 : 0;
          _cpuActiveMask[idx] = value;
          activePtr[idx] = value;
          _pActiveBuffer->didModifyRange(NS::Range::Make(idx, 1));
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
    _pLightIndexBuffer->didModifyRange(NS::Range::Make(0, lightIndexBytes));
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
    _pLightCdfBuffer->didModifyRange(NS::Range::Make(0, lightCdfBytes));
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
}




void Renderer::buildTextures() {
  MTL::TextureDescriptor *textureDescriptor =
      MTL::TextureDescriptor::alloc()->init();

  textureDescriptor->setPixelFormat(MTL::PixelFormat::PixelFormatRGBA32Float);
  textureDescriptor->setTextureType(MTL::TextureType::TextureType2D);
  textureDescriptor->setWidth(Camera::screenSize.x);
  textureDescriptor->setHeight(Camera::screenSize.y);
  textureDescriptor->setStorageMode(MTL::StorageMode::StorageModePrivate);
  textureDescriptor->setUsage(MTL::TextureUsageShaderRead |
                              MTL::TextureUsageShaderWrite);

  for (uint i = 0; i < 2; i++)
    _accumulationTargets[i] = _pDevice->newTexture(textureDescriptor);
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

  if (updateCamera()) {
    u.frameCount = 0;
    u.randomSeed = {randomFloat(), randomFloat(), randomFloat()};
  } else {
    u.frameCount++;
  }

  u.primitiveCount = _residentPrimitiveCount;
  u.triangleCount = _residentTriangleCount;
  u.totalPrimitiveCount = _allPrimitives.size();
  u.tlasNodeCount = _tlasNodeCount;
  u.blasNodeCount = _blasNodeCount;
  u.maxRayDepth = _pScene->maxRayDepth;
  u.debugAS = InputSystem::debugAS;
  u.lightCount = static_cast<uint32_t>(_lightCount);
  u.lightTotalWeight = _lightTotalWeight;

  _pUniformsBuffer->didModifyRange(NS::Range::Make(0, sizeof(UniformsData)));
}

void Renderer::draw(MTK::View *pView) {
  updateResidency();
  updateUniforms();
  beginFrameMetrics();
  std::swap(_accumulationTargets[0], _accumulationTargets[1]);

  NS::AutoreleasePool *pPool = NS::AutoreleasePool::alloc()->init();

  MTL::CommandBuffer *pCmd = _pCommandQueue->commandBuffer();
  pCmd->addCompletedHandler([this](MTL::CommandBuffer *cmd) {
    this->completeFrameMetrics(cmd);
  });
  MTL::RenderPassDescriptor *pRpd = pView->currentRenderPassDescriptor();
  MTL::RenderCommandEncoder *pEnc = pCmd->renderCommandEncoder(pRpd);

  pEnc->setRenderPipelineState(_pPSO);

  // Always bind something for each slot
  pEnc->setFragmentBuffer(_pBVHBuffer, 0, 0); // New!
  pEnc->setFragmentBuffer(_pSphereBuffer, 0, 1);
  pEnc->setFragmentBuffer(_pSphereMaterialBuffer, 0, 2);
  pEnc->setFragmentBuffer(_pUniformsBuffer, 0, 3);
  pEnc->setFragmentBuffer(_pTriangleVertexBuffer, 0, 4);
  pEnc->setFragmentBuffer(_pTriangleIndexBuffer, 0, 5);
  pEnc->setFragmentBuffer(_pPrimitiveIndexBuffer, 0, 6);
  pEnc->setFragmentBuffer(_pTLASBuffer, 0, 7);
  pEnc->setFragmentBuffer(_pActiveBuffer, 0, 8);
  pEnc->setFragmentBuffer(_pLightIndexBuffer, 0, 9);
  pEnc->setFragmentBuffer(_pLightCdfBuffer, 0, 10);
  pEnc->setFragmentBuffer(_pPrimitiveRemapBuffer, 0, 11);
  pEnc->setFragmentBuffer(_pPrimitiveHitBufferGPU, 0, 12);

  pEnc->setFragmentTexture(_accumulationTargets[0], 0);
  pEnc->setFragmentTexture(_accumulationTargets[1], 1);

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

  size_t activePrimitiveCount = 0;
  for (bool active : _activePrimitive)
    if (active)
      ++activePrimitiveCount;

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

    if (!forceAllToggles &&
        toggles + togglesNeeded > _residencyConfig.lodMaxTogglesPerFrame)
      continue;

    if (!forceAllToggles && !shouldBeActive && activePrimitiveCount > 0 &&
        togglesNeeded >= activePrimitiveCount)
      continue;

    size_t toggled = setObjectActive(objectIndex, shouldBeActive);
    if (toggled > 0) {
      toggles += toggled;
      changed = true;
      if (shouldBeActive) {
        activePrimitiveCount += toggled;
      } else {
        activePrimitiveCount =
            (toggled >= activePrimitiveCount) ? 0 : activePrimitiveCount - toggled;
      }
    }
  }

  if (activePrimitiveCount == 0 && !_activePrimitive.empty()) {
    // Ensure at least one primitive remains visible to avoid a blank scene
    if (setPrimitiveActive(0, true)) {
      changed = true;
      ++activePrimitiveCount;
    }
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

  std::sort(_rayHitSortedIndices.begin(), _rayHitSortedIndices.end(),
            [this](size_t a, size_t b) {
              float rawA =
                  (a < _primitiveHitScores.size()) ? _primitiveHitScores[a] : 0.0f;
              float rawB =
                  (b < _primitiveHitScores.size()) ? _primitiveHitScores[b] : 0.0f;
              float scoreA = sanitizeSortValue(rawA);
              float scoreB = sanitizeSortValue(rawB);
              if (scoreA == scoreB)
                return a < b;
              return scoreA > scoreB;
            });

  const size_t primCount = _activePrimitive.size();
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
    // No state changes to apply this frame.
    return;
  }
  rebuildResidentResources(forceFullRebuild);
}

void Renderer::drawableSizeWillChange(MTK::View *pView, CGSize size) {
  for (uint i = 0; i < 2; i++)
    if (_accumulationTargets[i])
      _accumulationTargets[i]->release();

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
      out << ",\"object\":" << objectIndex << ",\"blasRoot\":" << second;
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

  size_t totalPrimitiveCount = _allPrimitives.size();
  if (totalPrimitiveCount == 0)
    return;

  size_t bufferCount = _pPrimitiveHitReadback->length() / sizeof(uint32_t);
  size_t count = std::min(totalPrimitiveCount, bufferCount);
  if (count == 0)
    return;

  uint32_t *hitPtr =
      static_cast<uint32_t *>(_pPrimitiveHitReadback->contents());
  if (!hitPtr)
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
  processRayHitCounters();
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
