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

// Distance thresholds that implement a small hysteresis band for LOD toggles
// along with the minimum number of frames primitives must wait before they can
// flip state again.
constexpr float LOD_ENTER_DISTANCE = 225.0f;
constexpr float LOD_EXIT_DISTANCE = 275.0f;
constexpr uint32_t LOD_STATE_COOLDOWN_FRAMES = 5;
constexpr float ENERGY_TARGET_FRACTION = 0.9f;
constexpr size_t ENERGY_MIN_ACTIVE_PRIMITIVES = 16;
constexpr float RAY_HIT_DECAY = 0.85f;
constexpr float RAY_HIT_TARGET_FRACTION = 0.6f;
constexpr size_t RAY_HIT_MIN_ACTIVE_PRIMITIVES = 16;
constexpr size_t RAY_HIT_MAX_TOGGLES_PER_FRAME = 12;
constexpr uint32_t RAY_HIT_REBUILD_COOLDOWN_FRAMES = 6;
constexpr float SCREEN_FOOTPRINT_TARGET_FRACTION = 0.65f;
constexpr float SCREEN_FOOTPRINT_MIN_PIXEL_COVERAGE = 32.0f;
constexpr size_t SCREEN_FOOTPRINT_MIN_ACTIVE_PRIMITIVES = 16;
constexpr size_t SCREEN_FOOTPRINT_MAX_TOGGLES_PER_FRAME = 10;

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

} // namespace

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
  if (_pPrimitiveHitBuffer)
    _pPrimitiveHitBuffer->release();
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

  printf("LOD activation threshold: %.1f, deactivation threshold: %.1f (cooldown "
         "%u frames)\n",
         LOD_ENTER_DISTANCE, LOD_EXIT_DISTANCE, LOD_STATE_COOLDOWN_FRAMES);

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
  _activePrimitive.assign(primCount, true);
  _primitiveCooldown.assign(primCount, 0);
  _primitiveBounds.resize(primCount);
  _primitiveImportance.assign(primCount, 0.0f);
  _energySortedIndices.resize(primCount);
  _primitiveHitScores.assign(primCount, 0.0f);
  _primitiveHitLastFrame.assign(primCount, 0);
  _rayHitSortedIndices.resize(primCount);
  _primitiveScreenCoverage.assign(primCount, 0.0f);
  _screenCoverageSortedIndices.resize(primCount);
  _totalPrimitiveImportance = 0.0f;
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

  std::sort(_energySortedIndices.begin(), _energySortedIndices.end(),
            [this](size_t a, size_t b) {
              return _primitiveImportance[a] > _primitiveImportance[b];
            });

  if (_pPrimitiveHitBuffer) {
    _pPrimitiveHitBuffer->release();
    _pPrimitiveHitBuffer = nullptr;
  }
  size_t hitCount = primCount > 0 ? primCount : 1;
  _pPrimitiveHitBuffer =
      _pDevice->newBuffer(hitCount * sizeof(uint32_t),
                          MTL::ResourceStorageModeManaged);
  if (uint32_t *hitPtr = static_cast<uint32_t *>(_pPrimitiveHitBuffer->contents())) {
    std::memset(hitPtr, 0, hitCount * sizeof(uint32_t));
    _pPrimitiveHitBuffer->didModifyRange(
        NS::Range::Make(0, hitCount * sizeof(uint32_t)));
  }

  _rayHitRebuildCooldown = 0;

  _pScene->buildBVH();

  size_t fullTlasCount = 0;
  if (primCount > 0) {
    simd::float4 *tmp = _pScene->createTLASBuffer(fullTlasCount);
    delete[] tmp;
  }
  _totalNodeCount = fullTlasCount + primCount;

  rebuildResidentResources();
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

void Renderer::buildBuffers(const Scene &scene) {
  const size_t primitiveCount = scene.getPrimitiveCount();
  const size_t uniformsDataSize = sizeof(UniformsData);

  if (!_pUniformsBuffer) {
    _pUniformsBuffer =
        _pDevice->newBuffer(uniformsDataSize, MTL::ResourceStorageModeManaged);
    _pUniformsBuffer->didModifyRange(NS::Range::Make(0, uniformsDataSize));
  }

  if (_pSphereBuffer) {
    _pSphereBuffer->release();
    _pSphereBuffer = nullptr;
  }
  if (_pSphereMaterialBuffer) {
    _pSphereMaterialBuffer->release();
    _pSphereMaterialBuffer = nullptr;
  }

  simd::float4 *primitiveBuffer = nullptr;
  simd::float4 *materialBuffer = nullptr;
  if (primitiveCount > 0) {
    primitiveBuffer = scene.createTransformsBuffer();
    materialBuffer = scene.createMaterialsBuffer();
  }

  const size_t primitiveSize = primitiveCount * 3 * sizeof(simd::float4);
  const size_t materialSize = primitiveCount * 2 * sizeof(simd::float4);

  size_t primitiveAlloc =
      primitiveSize > 0 ? primitiveSize : sizeof(simd::float4);
  size_t materialAlloc = materialSize > 0 ? materialSize : sizeof(simd::float4);

  _pSphereBuffer =
      _pDevice->newBuffer(primitiveAlloc, MTL::ResourceStorageModeManaged);
  _pSphereMaterialBuffer =
      _pDevice->newBuffer(materialAlloc, MTL::ResourceStorageModeManaged);

  if (primitiveCount > 0) {
    memcpy(_pSphereBuffer->contents(), primitiveBuffer, primitiveSize);
    memcpy(_pSphereMaterialBuffer->contents(), materialBuffer, materialSize);
    _pSphereBuffer->didModifyRange(NS::Range::Make(0, primitiveSize));
    _pSphereMaterialBuffer->didModifyRange(NS::Range::Make(0, materialSize));
  }

  delete[] primitiveBuffer;
  delete[] materialBuffer;

  if (_pLightIndexBuffer) {
    _pLightIndexBuffer->release();
    _pLightIndexBuffer = nullptr;
  }
  if (_pLightCdfBuffer) {
    _pLightCdfBuffer->release();
    _pLightCdfBuffer = nullptr;
  }

  std::vector<uint32_t> lightIndices;
  std::vector<float> lightCdf;
  float totalWeight = 0.0f;

  if (primitiveCount > 0) {
    const auto &primitives = scene.getPrimitives();
    lightIndices.reserve(primitives.size());
    lightCdf.reserve(primitives.size());

    for (size_t i = 0; i < primitives.size(); ++i) {
      const Primitive &p = primitives[i];
      const Material &m = p.material;
      float emissionStrength = m.emissionPower * luminance(m.emissionColor);
      if (emissionStrength <= 0.0f)
        continue;

      float area = primitiveArea(p);
      if (area <= 0.0f)
        continue;

      float weight = area * emissionStrength;
      if (weight <= 0.0f)
        continue;

      totalWeight += weight;
      lightIndices.push_back(static_cast<uint32_t>(i));
      lightCdf.push_back(totalWeight);
    }
  }

  _lightCount = lightIndices.size();
  _lightTotalWeight = totalWeight;

  size_t lightCount = _lightCount > 0 ? _lightCount : 1;
  _pLightIndexBuffer =
      _pDevice->newBuffer(lightCount * sizeof(uint32_t),
                          MTL::ResourceStorageModeManaged);
  _pLightCdfBuffer =
      _pDevice->newBuffer(lightCount * sizeof(float),
                          MTL::ResourceStorageModeManaged);

  if (_lightCount > 0) {
    memcpy(_pLightIndexBuffer->contents(), lightIndices.data(),
           _lightCount * sizeof(uint32_t));
    _pLightIndexBuffer->didModifyRange(
        NS::Range::Make(0, _lightCount * sizeof(uint32_t)));

    memcpy(_pLightCdfBuffer->contents(), lightCdf.data(),
           _lightCount * sizeof(float));
    _pLightCdfBuffer->didModifyRange(
        NS::Range::Make(0, _lightCount * sizeof(float)));
  } else {
    uint32_t dummyIndex = 0;
    float dummyCdf = 0.0f;
    memcpy(_pLightIndexBuffer->contents(), &dummyIndex, sizeof(uint32_t));
    memcpy(_pLightCdfBuffer->contents(), &dummyCdf, sizeof(float));
    _pLightIndexBuffer->didModifyRange(NS::Range::Make(0, sizeof(uint32_t)));
    _pLightCdfBuffer->didModifyRange(NS::Range::Make(0, sizeof(float)));
  }

  if (_pActiveBuffer) {
    _pActiveBuffer->release();
    _pActiveBuffer = nullptr;
  }
  size_t activeAlloc = primitiveCount > 0 ? primitiveCount : 1;
  _pActiveBuffer =
      _pDevice->newBuffer(activeAlloc * sizeof(uint8_t),
                          MTL::ResourceStorageModeManaged);
  if (primitiveCount > 0) {
    std::vector<uint8_t> activeBytes(primitiveCount, 1);
    memcpy(_pActiveBuffer->contents(), activeBytes.data(),
           primitiveCount * sizeof(uint8_t));
    _pActiveBuffer->didModifyRange(
        NS::Range::Make(0, primitiveCount * sizeof(uint8_t)));
  } else {
    uint8_t inactive = 0;
    memcpy(_pActiveBuffer->contents(), &inactive, sizeof(uint8_t));
    _pActiveBuffer->didModifyRange(NS::Range::Make(0, sizeof(uint8_t)));
  }

  std::vector<simd::float3> vertices;
  std::vector<simd::uint3> indices;
  scene.createTriangleBuffers(vertices, indices);

  if (_pTriangleVertexBuffer) {
    _pTriangleVertexBuffer->release();
    _pTriangleVertexBuffer = nullptr;
  }
  if (_pTriangleIndexBuffer) {
    _pTriangleIndexBuffer->release();
    _pTriangleIndexBuffer = nullptr;
  }

  if (!vertices.empty()) {
    size_t vertexSize = vertices.size() * sizeof(simd::float3);
    _pTriangleVertexBuffer = _pDevice->newBuffer(
        vertices.data(), vertexSize, MTL::ResourceStorageModeManaged);
    _pTriangleVertexBuffer->didModifyRange(NS::Range::Make(0, vertexSize));
  } else {
    simd::float3 dummyVertex = {0, 0, 0};
    _pTriangleVertexBuffer = _pDevice->newBuffer(
        &dummyVertex, sizeof(simd::float3), MTL::ResourceStorageModeManaged);
  }

  if (!indices.empty()) {
    size_t indexSize = indices.size() * sizeof(simd::uint3);
    _pTriangleIndexBuffer = _pDevice->newBuffer(
        indices.data(), indexSize, MTL::ResourceStorageModeManaged);
    _pTriangleIndexBuffer->didModifyRange(NS::Range::Make(0, indexSize));
  } else {
    simd::uint3 dummyIndex = {0, 0, 0};
    _pTriangleIndexBuffer = _pDevice->newBuffer(
        &dummyIndex, sizeof(simd::uint3), MTL::ResourceStorageModeManaged);
  }
}

void Renderer::rebuildResidentResources() {
  Scene activeScene;
  activeScene.screenSize = _pScene->screenSize;
  activeScene.maxRayDepth = _pScene->maxRayDepth;

  std::vector<uint32_t> remap;
  remap.reserve(_allPrimitives.size());

  for (const auto &object : _allSceneObjects) {
    std::vector<Primitive> activePrims;
    activePrims.reserve(object.primitiveCount);
    std::vector<uint32_t> objectRemap;
    objectRemap.reserve(object.primitiveCount);
    for (size_t i = 0; i < object.primitiveCount; ++i) {
      size_t primIndex = object.firstPrimitive + i;
      if (primIndex < _activePrimitive.size() &&
          _activePrimitive[primIndex]) {
        activePrims.push_back(_allPrimitives[primIndex]);
        objectRemap.push_back(static_cast<uint32_t>(primIndex));
      }
    }
    if (!activePrims.empty()) {
      activeScene.addObjectSilent(activePrims);
      remap.insert(remap.end(), objectRemap.begin(), objectRemap.end());
    }
  }

  activeScene.buildBVH();

  rebuildAccelerationStructures(activeScene);
  buildBuffers(activeScene);

  if (_pPrimitiveRemapBuffer) {
    _pPrimitiveRemapBuffer->release();
    _pPrimitiveRemapBuffer = nullptr;
  }

  size_t remapCount = remap.empty() ? 1 : remap.size();
  _pPrimitiveRemapBuffer =
      _pDevice->newBuffer(remapCount * sizeof(uint32_t),
                          MTL::ResourceStorageModeManaged);
  if (uint32_t *remapPtr =
          static_cast<uint32_t *>(_pPrimitiveRemapBuffer->contents())) {
    if (!remap.empty()) {
      std::memcpy(remapPtr, remap.data(), remap.size() * sizeof(uint32_t));
      _pPrimitiveRemapBuffer->didModifyRange(
          NS::Range::Make(0, remap.size() * sizeof(uint32_t)));
    } else {
      uint32_t zero = 0;
      std::memcpy(remapPtr, &zero, sizeof(uint32_t));
      _pPrimitiveRemapBuffer->didModifyRange(
          NS::Range::Make(0, sizeof(uint32_t)));
    }
  }

  _residentPrimitiveCount = activeScene.getPrimitiveCount();
  _residentTriangleCount = activeScene.getTriangleCount();
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
  pEnc->setFragmentBuffer(_pPrimitiveHitBuffer, 0, 12);

  pEnc->setFragmentTexture(_accumulationTargets[0], 0);
  pEnc->setFragmentTexture(_accumulationTargets[1], 1);

  pEnc->drawPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle,
                       NS::UInteger(0), NS::UInteger(6));

  pEnc->endEncoding();

  MTL::BlitCommandEncoder *pBlit = pCmd->blitCommandEncoder();
  if (_pPrimitiveHitBuffer)
    pBlit->synchronizeResource(_pPrimitiveHitBuffer);
  pBlit->endEncoding();

  pCmd->presentDrawable(pView->currentDrawable());
  pCmd->commit();

  pPool->release();
}

void Renderer::updateResidency() {
  if (!_pScene)
    return;

  for (uint32_t &cooldown : _primitiveCooldown)
    if (cooldown > 0)
      --cooldown;

  switch (_pScene->getResidencyStrategy()) {
  case ResidencyStrategy::EnergyImportance:
    updateEnergyImportance();
    break;
  case ResidencyStrategy::RayHitBudget:
    updateRayHitBudget();
    break;
  case ResidencyStrategy::ScreenSpaceFootprint:
    updateScreenSpaceFootprint();
    break;
  case ResidencyStrategy::DistanceLOD:
  default:
    updateLODByDistance();
    break;
  }
}

void Renderer::updateLODByDistance() {
  // Use hysteresis so primitives do not flicker when hovering near the
  // activation boundary. Inactive primitives only become active once the
  // camera is closer than LOD_ENTER_DISTANCE, while active primitives stay
  // active until the camera has moved beyond LOD_EXIT_DISTANCE.

  size_t activeCount = 0;
  bool changed = false;
  for (size_t g = 0; g < _allPrimitives.size(); ++g) {
    float dist =
        simd::length(_primitiveBounds[g].center - Camera::position) -
        _primitiveBounds[g].radius;
    dist = std::max(dist, 0.0f);
    bool currentlyActive = g < _activePrimitive.size() && _activePrimitive[g];
    bool shouldBeActive = currentlyActive ? dist <= LOD_EXIT_DISTANCE
                                          : dist < LOD_ENTER_DISTANCE;
    bool canToggle =
        g >= _primitiveCooldown.size() || _primitiveCooldown[g] == 0;
    if (canToggle && shouldBeActive != currentlyActive &&
        setPrimitiveActive(g, shouldBeActive))
      changed = true;
    if (_activePrimitive[g])
      activeCount++;
  }

  if (activeCount == 0 && !_activePrimitive.empty()) {
    // Ensure at least one primitive remains visible to avoid a blank scene
    if (setPrimitiveActive(0, true))
      changed = true;
    activeCount = 1;
  }

  if (changed)
    rebuildResidentResources();
}

void Renderer::updateEnergyImportance() {
  if (_activePrimitive.empty())
    return;

  const size_t primCount = _activePrimitive.size();
  std::vector<bool> desiredState(primCount, false);
  size_t minActive = std::min(primCount, ENERGY_MIN_ACTIVE_PRIMITIVES);

  if (_totalPrimitiveImportance <= 0.0f) {
    for (size_t i = 0; i < minActive && i < _energySortedIndices.size(); ++i)
      desiredState[_energySortedIndices[i]] = true;
  } else {
    float cumulative = 0.0f;
    float targetImportance = _totalPrimitiveImportance * ENERGY_TARGET_FRACTION;
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
  for (size_t i = 0; i < primCount; ++i) {
    bool shouldBeActive = desiredState[i];
    if (shouldBeActive != _activePrimitive[i])
      if (setPrimitiveActive(i, shouldBeActive))
        changed = true;
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

  if (changed)
    rebuildResidentResources();
}

void Renderer::updateRayHitBudget() {
  if (_activePrimitive.empty())
    return;

  if (_rayHitSortedIndices.size() != _activePrimitive.size()) {
    _rayHitSortedIndices.resize(_activePrimitive.size());
    std::iota(_rayHitSortedIndices.begin(), _rayHitSortedIndices.end(), size_t(0));
  }
  if (_primitiveHitScores.size() < _activePrimitive.size())
    _primitiveHitScores.resize(_activePrimitive.size(), 0.0f);

  if (_rayHitRebuildCooldown > 0) {
    --_rayHitRebuildCooldown;
    return;
  }

  std::sort(_rayHitSortedIndices.begin(), _rayHitSortedIndices.end(),
            [this](size_t a, size_t b) {
              float scoreA =
                  (a < _primitiveHitScores.size()) ? _primitiveHitScores[a] : 0.0f;
              float scoreB =
                  (b < _primitiveHitScores.size()) ? _primitiveHitScores[b] : 0.0f;
              if (scoreA == scoreB)
                return a < b;
              return scoreA > scoreB;
            });

  const size_t primCount = _activePrimitive.size();
  const size_t minActive = std::min(primCount, RAY_HIT_MIN_ACTIVE_PRIMITIVES);
  size_t targetActive =
      static_cast<size_t>(std::ceil(primCount * RAY_HIT_TARGET_FRACTION));
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
    if (i < _primitiveCooldown.size() && _primitiveCooldown[i] > 0)
      continue;
    if (toggles >= RAY_HIT_MAX_TOGGLES_PER_FRAME)
      break;
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

  if (changed) {
    rebuildResidentResources();
    _rayHitRebuildCooldown = RAY_HIT_REBUILD_COOLDOWN_FRAMES;
  }
}

void Renderer::updateScreenSpaceFootprint() {
  if (_activePrimitive.empty())
    return;

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
              float ca = (a < _primitiveScreenCoverage.size())
                             ? _primitiveScreenCoverage[a]
                             : 0.0f;
              float cb = (b < _primitiveScreenCoverage.size())
                             ? _primitiveScreenCoverage[b]
                             : 0.0f;
              if (ca == cb)
                return a < b;
              return ca > cb;
            });

  std::vector<bool> desired(primCount, false);
  const size_t minActive =
      std::min(primCount, SCREEN_FOOTPRINT_MIN_ACTIVE_PRIMITIVES);
  float accumulated = 0.0f;
  size_t enabled = 0;
  float targetCoverage = screenArea * SCREEN_FOOTPRINT_TARGET_FRACTION;
  for (size_t idx : _screenCoverageSortedIndices) {
    if (idx >= _primitiveScreenCoverage.size())
      continue;
    float coverage = _primitiveScreenCoverage[idx];
    if (enabled >= minActive && accumulated >= targetCoverage)
      break;
    if (enabled >= minActive && coverage < SCREEN_FOOTPRINT_MIN_PIXEL_COVERAGE)
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
    if (i < _primitiveCooldown.size() && _primitiveCooldown[i] > 0)
      continue;
    if (toggles >= SCREEN_FOOTPRINT_MAX_TOGGLES_PER_FRAME)
      break;
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

  if (changed)
    rebuildResidentResources();
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
  if (index < _primitiveCooldown.size())
    _primitiveCooldown[index] = LOD_STATE_COOLDOWN_FRAMES;
  return true;
}

void Renderer::rebuildAccelerationStructures(const Scene &scene) {
  size_t newBlasCount = scene.getBVHNodeCount();
  _blasNodeCount = newBlasCount;

  if (_pBVHBuffer) {
    _pBVHBuffer->release();
    _pBVHBuffer = nullptr;
  }

  if (newBlasCount > 0) {
    simd::float4 *bvhData = scene.createBVHBuffer();
    _pBVHBuffer = _pDevice->newBuffer(
        bvhData, sizeof(simd::float4) * newBlasCount * 2,
        MTL::ResourceStorageModeManaged);
    _pBVHBuffer->didModifyRange(NS::Range::Make(0, _pBVHBuffer->length()));
    delete[] bvhData;
  } else {
    _pBVHBuffer =
        _pDevice->newBuffer(sizeof(simd::float4), MTL::ResourceStorageModeManaged);
  }

  size_t tlasCount = 0;
  simd::float4 *tlasData = scene.createTLASBuffer(tlasCount);
  _tlasNodeCount = tlasCount;

  if (_pTLASBuffer) {
    _pTLASBuffer->release();
    _pTLASBuffer = nullptr;
  }

  if (tlasData && tlasCount > 0) {
    _pTLASBuffer = _pDevice->newBuffer(
        tlasData, sizeof(simd::float4) * tlasCount * 2,
        MTL::ResourceStorageModeManaged);
    _pTLASBuffer->didModifyRange(NS::Range::Make(0, _pTLASBuffer->length()));
  } else {
    _pTLASBuffer =
        _pDevice->newBuffer(sizeof(simd::float4), MTL::ResourceStorageModeManaged);
  }
  delete[] tlasData;

  if (_pPrimitiveIndexBuffer) {
    _pPrimitiveIndexBuffer->release();
    _pPrimitiveIndexBuffer = nullptr;
  }

  size_t indexCount = scene.getPrimitiveIndices().size();
  if (indexCount > 0) {
    int *rawIndices = scene.createPrimitiveIndexBuffer();
    size_t indexSize = indexCount * sizeof(int);
    _pPrimitiveIndexBuffer = _pDevice->newBuffer(
        rawIndices, indexSize, MTL::ResourceStorageModeManaged);
    _pPrimitiveIndexBuffer->didModifyRange(NS::Range::Make(0, indexSize));
    delete[] rawIndices;
  } else {
    _pPrimitiveIndexBuffer =
        _pDevice->newBuffer(sizeof(int), MTL::ResourceStorageModeManaged);
  }

  size_t newActiveCount = _tlasNodeCount + scene.getPrimitiveCount();
  if (newActiveCount != _activeNodeCount) {
    _activeNodeCount = newActiveCount;
    printf("Active nodes: %zu\n", _activeNodeCount);
  }
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
  if (!_pPrimitiveHitBuffer)
    return;

  size_t totalPrimitiveCount = _allPrimitives.size();
  if (totalPrimitiveCount == 0)
    return;

  size_t bufferCount = _pPrimitiveHitBuffer->length() / sizeof(uint32_t);
  size_t count = std::min(totalPrimitiveCount, bufferCount);
  if (count == 0)
    return;

  uint32_t *hitPtr = static_cast<uint32_t *>(_pPrimitiveHitBuffer->contents());
  if (!hitPtr)
    return;

  if (_primitiveHitScores.size() < totalPrimitiveCount)
    _primitiveHitScores.resize(totalPrimitiveCount, 0.0f);
  if (_primitiveHitLastFrame.size() < totalPrimitiveCount)
    _primitiveHitLastFrame.resize(totalPrimitiveCount, 0);

  for (size_t i = 0; i < count; ++i) {
    uint32_t hits = hitPtr[i];
    _primitiveHitLastFrame[i] = hits;
    _primitiveHitScores[i] = _primitiveHitScores[i] * RAY_HIT_DECAY +
                            static_cast<float>(hits);
    hitPtr[i] = 0;
  }

  _pPrimitiveHitBuffer->didModifyRange(
      NS::Range::Make(0, count * sizeof(uint32_t)));
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
  size_t offloaded = _totalNodeCount > _activeNodeCount ?
                         _totalNodeCount - _activeNodeCount :
                         0;
  printf(
      "Nodes active: %zu offloaded: %zu CPU: %.3f ms GPU: %.3f ms Rays/s: %.2f\n",
      _activeNodeCount, offloaded, _lastCPUTime * 1000.0,
      _lastGPUTime * 1000.0, _lastRaysPerSecond);
}

double Renderer::lastCPUTime() const { return _lastCPUTime; }
double Renderer::lastGPUTime() const { return _lastGPUTime; }
double Renderer::lastRaysPerSecond() const { return _lastRaysPerSecond; }
size_t Renderer::activeNodeCount() const { return _activeNodeCount; }
size_t Renderer::totalNodeCount() const { return _totalNodeCount; }
