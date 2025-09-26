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
  size_t previousPrimitiveCount = _residentPrimitiveCount;
  size_t previousTriangleCount = _residentTriangleCount;
  size_t previousLightCount = _lightCount;
  size_t previousBlasCount = _blasNodeCount;
  size_t previousTlasCount = _tlasNodeCount;

  const size_t uniformsDataSize = sizeof(UniformsData);
  if (!_pUniformsBuffer) {
    _pUniformsBuffer =
        _pDevice->newBuffer(uniformsDataSize, MTL::ResourceStorageModeManaged);
    if (_pUniformsBuffer)
      _pUniformsBuffer->didModifyRange(
          NS::Range::Make(0, uniformsDataSize));
  }

  if (_primitiveToResidentIndex.size() < _activePrimitive.size())
    _primitiveToResidentIndex.assign(_activePrimitive.size(), -1);

  auto deduplicate = [](std::vector<size_t> &values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
  };

  if (forceFullRebuild) {
    _residentPrimitives.clear();
    _residentRemap.clear();
    for (size_t i = 0; i < _activePrimitive.size(); ++i) {
      if (_activePrimitive[i]) {
        size_t slot = _residentPrimitives.size();
        _residentPrimitives.push_back(_allPrimitives[i]);
        _residentRemap.push_back(static_cast<uint32_t>(i));
        if (i < _primitiveToResidentIndex.size())
          _primitiveToResidentIndex[i] = static_cast<int32_t>(slot);
      } else if (i < _primitiveToResidentIndex.size()) {
        _primitiveToResidentIndex[i] = -1;
      }
    }
    _recentlyActivated.clear();
    _recentlyDeactivated.clear();
  } else {
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

    for (size_t idx : _recentlyDeactivated) {
      if (idx >= _primitiveToResidentIndex.size())
        continue;
      int32_t slot = _primitiveToResidentIndex[idx];
      if (slot < 0 || _residentPrimitives.empty() ||
          static_cast<size_t>(slot) >= _residentPrimitives.size()) {
        _primitiveToResidentIndex[idx] = -1;
        continue;
      }
      size_t last = _residentPrimitives.size() - 1;
      if (static_cast<size_t>(slot) != last) {
        _residentPrimitives[slot] = _residentPrimitives[last];
        _residentRemap[slot] = _residentRemap[last];
        size_t movedIndex = _residentRemap[slot];
        if (movedIndex < _primitiveToResidentIndex.size())
          _primitiveToResidentIndex[movedIndex] = static_cast<int32_t>(slot);
      }
      _residentPrimitives.pop_back();
      _residentRemap.pop_back();
      _primitiveToResidentIndex[idx] = -1;
    }

    for (size_t idx : _recentlyActivated) {
      if (idx >= _activePrimitive.size() || !_activePrimitive[idx])
        continue;
      if (idx >= _primitiveToResidentIndex.size())
        _primitiveToResidentIndex.resize(idx + 1, -1);
      if (_primitiveToResidentIndex[idx] >= 0)
        continue;
      size_t slot = _residentPrimitives.size();
      _residentPrimitives.push_back(_allPrimitives[idx]);
      _residentRemap.push_back(static_cast<uint32_t>(idx));
      _primitiveToResidentIndex[idx] = static_cast<int32_t>(slot);
    }

    _recentlyActivated.clear();
    _recentlyDeactivated.clear();
  }

  _residentPrimitiveCount = _residentPrimitives.size();

  std::vector<simd::float4> primitiveBuffer(_residentPrimitiveCount * 3,
                                            simd::float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<simd::float4> materialBuffer(_residentPrimitiveCount * 2,
                                           simd::float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<simd::float3> triangleVertices;
  std::vector<simd::uint3> triangleIndices;
  triangleVertices.reserve(_residentPrimitiveCount * 3);
  triangleIndices.reserve(_residentPrimitiveCount);

  std::vector<uint32_t> lightIndices;
  std::vector<float> lightCdf;
  lightIndices.reserve(_residentPrimitiveCount);
  lightCdf.reserve(_residentPrimitiveCount);

  float totalLightWeight = 0.0f;
  size_t vertexCursor = 0;

  for (size_t i = 0; i < _residentPrimitiveCount; ++i) {
    const Primitive &p = _residentPrimitives[i];
    simd::float4 *primBase = &primitiveBuffer[3 * i];
    simd::float4 *matBase = &materialBuffer[2 * i];

    switch (p.type) {
    case PrimitiveType::Sphere: {
      primBase[0] =
          simd::make_float4(p.sphere.center, static_cast<float>(p.type));
      primBase[1] =
          simd::make_float4(simd::make_float3(p.sphere.radius, 0.0f, 0.0f), 0.0f);
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
      triangleVertices.push_back(p.triangle.v0);
      triangleVertices.push_back(p.triangle.v1);
      triangleVertices.push_back(p.triangle.v2);
      triangleIndices.push_back(
          simd::make_uint3(vertexCursor, vertexCursor + 1, vertexCursor + 2));
      vertexCursor += 3;
      break;
    }
    }

    const Material &m = p.material;
    matBase[0] = simd::make_float4(m.albedo, m.materialType);
    matBase[1] = simd::make_float4(m.emissionColor, m.emissionPower);

    float emissionStrength = m.emissionPower * luminance(m.emissionColor);
    if (emissionStrength > 0.0f) {
      float area = primitiveArea(p);
      if (area > 0.0f) {
        float weight = area * emissionStrength;
        if (weight > 0.0f) {
          totalLightWeight += weight;
          lightIndices.push_back(static_cast<uint32_t>(i));
          lightCdf.push_back(totalLightWeight);
        }
      }
    }
  }

  _residentTriangleCount = triangleIndices.size();
  _lightCount = lightIndices.size();
  _lightTotalWeight = totalLightWeight;

  std::vector<int> primitiveIndices(_residentPrimitiveCount);
  std::iota(primitiveIndices.begin(), primitiveIndices.end(), 0);

  std::vector<BVHNode> bvhNodes;
  if (_residentPrimitiveCount > 0) {
    bvhNodes.reserve(_residentPrimitiveCount * 2);
    buildBVHRecursive(_residentPrimitives, primitiveIndices, bvhNodes, 0,
                      _residentPrimitiveCount);
  }

  _blasNodeCount = bvhNodes.size();

  std::vector<TLASNode> tlasNodes;
  if (_residentPrimitiveCount > 0 && !bvhNodes.empty()) {
    TLASNode root;
    root.boundsMin = bvhNodes.front().boundsMin;
    root.boundsMax = bvhNodes.front().boundsMax;
    root.leftChild = -1;
    root.rightChild = 0;
    tlasNodes.push_back(root);
  }

  _tlasNodeCount = tlasNodes.size();

  _residentNodeCount = _blasNodeCount + _tlasNodeCount;

  std::vector<simd::float4> bvhPacked(_blasNodeCount * 2,
                                      simd::float4{0.0f, 0.0f, 0.0f, 0.0f});
  for (size_t i = 0; i < _blasNodeCount; ++i) {
    const BVHNode &node = bvhNodes[i];
    float leftBits = 0.0f;
    float countBits = 0.0f;
    std::memcpy(&leftBits, &node.leftFirst, sizeof(int));
    std::memcpy(&countBits, &node.count, sizeof(int));
    bvhPacked[2 * i] = simd::make_float4(node.boundsMin, leftBits);
    bvhPacked[2 * i + 1] = simd::make_float4(node.boundsMax, countBits);
  }

  std::vector<simd::float4> tlasPacked(_tlasNodeCount * 2,
                                       simd::float4{0.0f, 0.0f, 0.0f, 0.0f});
  for (size_t i = 0; i < _tlasNodeCount; ++i) {
    const TLASNode &node = tlasNodes[i];
    float leftBits = 0.0f;
    float rightBits = 0.0f;
    std::memcpy(&leftBits, &node.leftChild, sizeof(int));
    std::memcpy(&rightBits, &node.rightChild, sizeof(int));
    tlasPacked[2 * i] = simd::make_float4(node.boundsMin, leftBits);
    tlasPacked[2 * i + 1] = simd::make_float4(node.boundsMax, rightBits);
  }

  size_t primitiveFloat4Count = _residentPrimitiveCount * 3;
  size_t previousPrimitiveFloat4Count = previousPrimitiveCount * 3;
  ensureBufferCapacity(_pSphereBuffer,
                       std::max<size_t>(primitiveFloat4Count, 1) *
                           sizeof(simd::float4),
                       _sphereBufferCapacity, true);
  if (_pSphereBuffer) {
    size_t primitiveCapacity = _pSphereBuffer->length() / sizeof(simd::float4);
    if (primitiveCapacity == 0 && _pSphereBuffer->length() > 0)
      primitiveCapacity = 1;
    if (simd::float4 *primitiveDst =
            static_cast<simd::float4 *>(_pSphereBuffer->contents())) {
      BufferCountInfo primitiveInfo =
          prepareBufferCounts(primitiveFloat4Count, previousPrimitiveFloat4Count,
                              primitiveCapacity);
      size_t copyCount = std::min(primitiveFloat4Count, primitiveInfo.current);
      if (copyCount > 0) {
        std::memcpy(primitiveDst, primitiveBuffer.data(),
                    copyCount * sizeof(simd::float4));
      } else if (primitiveCapacity > 0) {
        primitiveDst[0] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
      if (primitiveInfo.zeroCount > 0) {
        std::memset(primitiveDst + primitiveInfo.zeroStart, 0,
                    primitiveInfo.zeroCount * sizeof(simd::float4));
      }
      if (primitiveInfo.touched > 0) {
        _pSphereBuffer->didModifyRange(NS::Range::Make(
            0, primitiveInfo.touched * sizeof(simd::float4)));
      }
    }
  }

  size_t materialFloat4Count = _residentPrimitiveCount * 2;
  size_t previousMaterialFloat4Count = previousPrimitiveCount * 2;
  ensureBufferCapacity(_pSphereMaterialBuffer,
                       std::max<size_t>(materialFloat4Count, 1) *
                           sizeof(simd::float4),
                       _sphereMaterialBufferCapacity, true);
  if (_pSphereMaterialBuffer) {
    size_t materialCapacity =
        _pSphereMaterialBuffer->length() / sizeof(simd::float4);
    if (materialCapacity == 0 && _pSphereMaterialBuffer->length() > 0)
      materialCapacity = 1;
    if (simd::float4 *materialDst =
            static_cast<simd::float4 *>(_pSphereMaterialBuffer->contents())) {
      BufferCountInfo materialInfo = prepareBufferCounts(
          materialFloat4Count, previousMaterialFloat4Count, materialCapacity);
      size_t copyCount = std::min(materialFloat4Count, materialInfo.current);
      if (copyCount > 0) {
        std::memcpy(materialDst, materialBuffer.data(),
                    copyCount * sizeof(simd::float4));
      } else if (materialCapacity > 0) {
        materialDst[0] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (materialCapacity > 1)
          materialDst[1] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
      if (materialInfo.zeroCount > 0) {
        std::memset(materialDst + materialInfo.zeroStart, 0,
                    materialInfo.zeroCount * sizeof(simd::float4));
      }
      if (materialInfo.touched > 0) {
        _pSphereMaterialBuffer->didModifyRange(NS::Range::Make(
            0, materialInfo.touched * sizeof(simd::float4)));
      }
    }
  }

  ensureBufferCapacity(_pPrimitiveIndexBuffer,
                       std::max<size_t>(_residentPrimitiveCount, 1) *
                           sizeof(int),
                       _primitiveIndexBufferCapacity, true);
  if (_pPrimitiveIndexBuffer) {
    size_t indexCapacity = _pPrimitiveIndexBuffer->length() / sizeof(int);
    if (indexCapacity == 0 && _pPrimitiveIndexBuffer->length() > 0)
      indexCapacity = 1;
    if (int *indexPtr =
            static_cast<int *>(_pPrimitiveIndexBuffer->contents())) {
      BufferCountInfo indexInfo = prepareBufferCounts(
          _residentPrimitiveCount, previousPrimitiveCount, indexCapacity);
      size_t copyCount = std::min(_residentPrimitiveCount, indexInfo.current);
      if (copyCount > 0) {
        std::memcpy(indexPtr, primitiveIndices.data(),
                    copyCount * sizeof(int));
      } else if (indexCapacity > 0) {
        indexPtr[0] = 0;
      }
      if (indexInfo.zeroCount > 0) {
        std::memset(indexPtr + indexInfo.zeroStart, 0,
                    indexInfo.zeroCount * sizeof(int));
      }
      if (indexInfo.touched > 0) {
        _pPrimitiveIndexBuffer->didModifyRange(
            NS::Range::Make(0, indexInfo.touched * sizeof(int)));
      }
    }
  }

  ensureBufferCapacity(_pBVHBuffer,
                       std::max<size_t>(_blasNodeCount * 2, size_t(1)) *
                           sizeof(simd::float4),
                       _bvhBufferCapacity, true);
  if (_pBVHBuffer) {
    size_t bvhCapacity = _pBVHBuffer->length() / sizeof(simd::float4);
    if (bvhCapacity == 0 && _pBVHBuffer->length() > 0)
      bvhCapacity = 1;
    if (simd::float4 *bvhPtr =
            static_cast<simd::float4 *>(_pBVHBuffer->contents())) {
      size_t blasFloat4Count = _blasNodeCount * 2;
      size_t previousBlasFloat4Count = previousBlasCount * 2;
      BufferCountInfo bvhInfo = prepareBufferCounts(
          blasFloat4Count, previousBlasFloat4Count, bvhCapacity);
      size_t copyCount = std::min(blasFloat4Count, bvhInfo.current);
      if (copyCount > 0) {
        std::memcpy(bvhPtr, bvhPacked.data(),
                    copyCount * sizeof(simd::float4));
      } else if (bvhCapacity > 0) {
        bvhPtr[0] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (bvhCapacity > 1)
          bvhPtr[1] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
      if (bvhInfo.zeroCount > 0) {
        std::memset(bvhPtr + bvhInfo.zeroStart, 0,
                    bvhInfo.zeroCount * sizeof(simd::float4));
      }
      if (bvhInfo.touched > 0) {
        _pBVHBuffer->didModifyRange(NS::Range::Make(
            0, bvhInfo.touched * sizeof(simd::float4)));
      }
    }
  }

  ensureBufferCapacity(_pTLASBuffer,
                       std::max<size_t>(_tlasNodeCount * 2, size_t(1)) *
                           sizeof(simd::float4),
                       _tlasBufferCapacity, true);
  if (_pTLASBuffer) {
    size_t tlasCapacity = _pTLASBuffer->length() / sizeof(simd::float4);
    if (tlasCapacity == 0 && _pTLASBuffer->length() > 0)
      tlasCapacity = 1;
    if (simd::float4 *tlasPtr =
            static_cast<simd::float4 *>(_pTLASBuffer->contents())) {
      size_t tlasFloat4Count = _tlasNodeCount * 2;
      size_t previousTlasFloat4Count = previousTlasCount * 2;
      BufferCountInfo tlasInfo = prepareBufferCounts(
          tlasFloat4Count, previousTlasFloat4Count, tlasCapacity);
      size_t copyCount = std::min(tlasFloat4Count, tlasInfo.current);
      if (copyCount > 0) {
        std::memcpy(tlasPtr, tlasPacked.data(),
                    copyCount * sizeof(simd::float4));
      } else if (tlasCapacity > 0) {
        tlasPtr[0] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (tlasCapacity > 1)
          tlasPtr[1] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
      if (tlasInfo.zeroCount > 0) {
        std::memset(tlasPtr + tlasInfo.zeroStart, 0,
                    tlasInfo.zeroCount * sizeof(simd::float4));
      }
      if (tlasInfo.touched > 0) {
        _pTLASBuffer->didModifyRange(NS::Range::Make(
            0, tlasInfo.touched * sizeof(simd::float4)));
      }
    }
  }

  ensureBufferCapacity(_pActiveBuffer,
                       std::max<size_t>(_residentPrimitiveCount, size_t(1)) *
                           sizeof(uint8_t),
                       _activeBufferCapacity, true);
  if (_pActiveBuffer) {
    size_t activeCapacity = _pActiveBuffer->length() / sizeof(uint8_t);
    if (activeCapacity == 0 && _pActiveBuffer->length() > 0)
      activeCapacity = 1;
    if (uint8_t *activePtr =
            static_cast<uint8_t *>(_pActiveBuffer->contents())) {
      BufferCountInfo activeInfo = prepareBufferCounts(
          _residentPrimitiveCount, previousPrimitiveCount, activeCapacity);
      if (activeInfo.touched > 0) {
        std::memset(activePtr, 0, activeInfo.touched * sizeof(uint8_t));
        if (activeInfo.current > 0) {
          std::memset(activePtr, 1, activeInfo.current * sizeof(uint8_t));
        }
        _pActiveBuffer->didModifyRange(
            NS::Range::Make(0, activeInfo.touched * sizeof(uint8_t)));
      }
    }
  }

  ensureBufferCapacity(_pPrimitiveRemapBuffer,
                       std::max<size_t>(_residentRemap.size(), size_t(1)) *
                           sizeof(uint32_t),
                       _primitiveRemapBufferCapacity, true);
  if (_pPrimitiveRemapBuffer) {
    size_t remapCapacity =
        _pPrimitiveRemapBuffer->length() / sizeof(uint32_t);
    if (remapCapacity == 0 && _pPrimitiveRemapBuffer->length() > 0)
      remapCapacity = 1;
    if (uint32_t *remapPtr =
            static_cast<uint32_t *>(_pPrimitiveRemapBuffer->contents())) {
      BufferCountInfo remapInfo = prepareBufferCounts(
          _residentRemap.size(), previousPrimitiveCount, remapCapacity);
      if (remapInfo.touched > 0) {
        std::memset(remapPtr, 0, remapInfo.touched * sizeof(uint32_t));
        size_t copyCount =
            std::min(_residentRemap.size(), remapInfo.current);
        if (copyCount > 0) {
          std::memcpy(remapPtr, _residentRemap.data(),
                      copyCount * sizeof(uint32_t));
        }
        _pPrimitiveRemapBuffer->didModifyRange(NS::Range::Make(
            0, remapInfo.touched * sizeof(uint32_t)));
      }
    }
  }

  ensureBufferCapacity(_pLightIndexBuffer,
                       std::max<size_t>(_lightCount, size_t(1)) *
                           sizeof(uint32_t),
                       _lightIndexBufferCapacity, true);
  if (_pLightIndexBuffer) {
    size_t lightIndexCapacity =
        _pLightIndexBuffer->length() / sizeof(uint32_t);
    if (lightIndexCapacity == 0 && _pLightIndexBuffer->length() > 0)
      lightIndexCapacity = 1;
    if (uint32_t *lightIndexPtr =
            static_cast<uint32_t *>(_pLightIndexBuffer->contents())) {
      BufferCountInfo lightIndexInfo = prepareBufferCounts(
          _lightCount, previousLightCount, lightIndexCapacity);
      if (lightIndexInfo.touched > 0) {
        std::memset(lightIndexPtr, 0,
                    lightIndexInfo.touched * sizeof(uint32_t));
        size_t copyCount = std::min(_lightCount, lightIndexInfo.current);
        if (copyCount > 0) {
          std::memcpy(lightIndexPtr, lightIndices.data(),
                      copyCount * sizeof(uint32_t));
        }
        _pLightIndexBuffer->didModifyRange(NS::Range::Make(
            0, lightIndexInfo.touched * sizeof(uint32_t)));
      }
    }
  }

  ensureBufferCapacity(_pLightCdfBuffer,
                       std::max<size_t>(_lightCount, size_t(1)) * sizeof(float),
                       _lightCdfBufferCapacity, true);
  if (_pLightCdfBuffer) {
    size_t lightCdfCapacity = _pLightCdfBuffer->length() / sizeof(float);
    if (lightCdfCapacity == 0 && _pLightCdfBuffer->length() > 0)
      lightCdfCapacity = 1;
    if (float *lightCdfPtr =
            static_cast<float *>(_pLightCdfBuffer->contents())) {
      BufferCountInfo lightCdfInfo = prepareBufferCounts(
          _lightCount, previousLightCount, lightCdfCapacity);
      if (lightCdfInfo.touched > 0) {
        std::memset(lightCdfPtr, 0,
                    lightCdfInfo.touched * sizeof(float));
        size_t copyCount = std::min(_lightCount, lightCdfInfo.current);
        if (copyCount > 0) {
          std::memcpy(lightCdfPtr, lightCdf.data(),
                      copyCount * sizeof(float));
        }
        _pLightCdfBuffer->didModifyRange(NS::Range::Make(
            0, lightCdfInfo.touched * sizeof(float)));
      }
    }
  }

  size_t vertexCount = triangleVertices.size();
  size_t indexCount = triangleIndices.size();
  size_t previousVertexCount = previousTriangleCount * 3;
  size_t previousIndexCount = previousTriangleCount;

  ensureBufferCapacity(_pTriangleVertexBuffer,
                       std::max<size_t>(vertexCount, size_t(1)) *
                           sizeof(simd::float3),
                       _triangleVertexBufferCapacity, true);
  if (_pTriangleVertexBuffer) {
    size_t vertexCapacity =
        _pTriangleVertexBuffer->length() / sizeof(simd::float3);
    if (vertexCapacity == 0 && _pTriangleVertexBuffer->length() > 0)
      vertexCapacity = 1;
    if (simd::float3 *vertexPtr = static_cast<simd::float3 *>(
            _pTriangleVertexBuffer->contents())) {
      BufferCountInfo vertexInfo = prepareBufferCounts(
          vertexCount, previousVertexCount, vertexCapacity);
      size_t copyCount = std::min(vertexCount, vertexInfo.current);
      if (copyCount > 0) {
        std::memcpy(vertexPtr, triangleVertices.data(),
                    copyCount * sizeof(simd::float3));
      } else if (vertexCapacity > 0) {
        vertexPtr[0] = simd::float3{0.0f, 0.0f, 0.0f};
      }
      if (vertexInfo.zeroCount > 0) {
        std::memset(vertexPtr + vertexInfo.zeroStart, 0,
                    vertexInfo.zeroCount * sizeof(simd::float3));
      }
      if (vertexInfo.touched > 0) {
        _pTriangleVertexBuffer->didModifyRange(NS::Range::Make(
            0, vertexInfo.touched * sizeof(simd::float3)));
      }
    }
  }

  ensureBufferCapacity(_pTriangleIndexBuffer,
                       std::max<size_t>(indexCount, size_t(1)) *
                           sizeof(simd::uint3),
                       _triangleIndexBufferCapacity, true);
  if (_pTriangleIndexBuffer) {
    size_t indexCapacity =
        _pTriangleIndexBuffer->length() / sizeof(simd::uint3);
    if (indexCapacity == 0 && _pTriangleIndexBuffer->length() > 0)
      indexCapacity = 1;
    if (simd::uint3 *indexPtr = static_cast<simd::uint3 *>(
            _pTriangleIndexBuffer->contents())) {
      BufferCountInfo indexInfo = prepareBufferCounts(
          indexCount, previousIndexCount, indexCapacity);
      size_t copyCount = std::min(indexCount, indexInfo.current);
      if (copyCount > 0) {
        std::memcpy(indexPtr, triangleIndices.data(),
                    copyCount * sizeof(simd::uint3));
      } else if (indexCapacity > 0) {
        indexPtr[0] = simd::make_uint3(0, 0, 0);
      }
      if (indexInfo.zeroCount > 0) {
        std::memset(indexPtr + indexInfo.zeroStart, 0,
                    indexInfo.zeroCount * sizeof(simd::uint3));
      }
      if (indexInfo.touched > 0) {
        _pTriangleIndexBuffer->didModifyRange(NS::Range::Make(
            0, indexInfo.touched * sizeof(simd::uint3)));
      }
    }
  }

  size_t newActiveCount = _residentNodeCount;
  if (newActiveCount != _activeNodeCount) {
    _activeNodeCount = newActiveCount;
    printf("Resident nodes: %zu\n", _activeNodeCount);
  }
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
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    const BoundingSphere &sphere =
        (objectIndex < _objectBounds.size())
            ? _objectBounds[objectIndex]
            : BoundingSphere{simd::make_float3(0.0f, 0.0f, 0.0f), 0.0f};
    float dist = simd::length(sphere.center - Camera::position) - sphere.radius;
    dist = std::max(dist, 0.0f);

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

    size_t toggled = setObjectActive(objectIndex, shouldBeActive);
    if (toggled > 0) {
      toggles += toggled;
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
