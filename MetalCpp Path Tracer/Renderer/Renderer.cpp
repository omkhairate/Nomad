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
#include <simd/simd.h>
#include <string>
#include <utility>

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

  _rebuildThread = std::thread(&Renderer::rebuildWorkerLoop, this);

  updateVisibleScene();
  buildShaders();
  buildTextures();

  recalculateViewport();
}

Renderer::~Renderer() {
  {
    std::lock_guard<std::mutex> lock(_workerMutex);
    _workerExit = true;
  }
  _workerCv.notify_all();
  if (_rebuildThread.joinable())
    _rebuildThread.join();

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

  // Store full primitive list and initialize tracking
  _allPrimitives = _pScene->getPrimitives();
  _allSceneObjects = _pScene->getObjects();
  size_t primCount = _allPrimitives.size();
  _activePrimitive.assign(primCount, true);
  _primitiveBounds.resize(primCount);
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
  }

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
  std::vector<uint8_t> snapshot;
  {
    std::lock_guard<std::mutex> lock(_primitiveMutex);
    snapshot.resize(_activePrimitive.size());
    for (size_t i = 0; i < _activePrimitive.size(); ++i)
      snapshot[i] = _activePrimitive[i] ? 1 : 0;
  }

  PreparedResources resources = buildResourcesForMask(snapshot);
  commitPreparedResources(resources);

  {
    std::lock_guard<std::mutex> lock(_workerMutex);
    _lodAppliedVersion = _lodRequestedVersion;
    _resourceBuffers[_activeResourceIndex].version = _lodAppliedVersion;
    _resourceBuffers[_activeResourceIndex].ready = false;
  }
  _lodDirty.store(false, std::memory_order_release);
}

void Renderer::enqueueLODRebuild(const std::vector<uint8_t> &mask) {
  {
    std::lock_guard<std::mutex> lock(_workerMutex);
    _pendingActiveMask = mask;
    ++_lodRequestedVersion;
    if (_readyResourceIndex != -1 && _resourceBuffers[_readyResourceIndex].version <
                                         _lodRequestedVersion) {
      _resourceBuffers[_readyResourceIndex].ready = false;
      _resourceBuffers[_readyResourceIndex].data = PreparedResources();
      _resourceBuffers[_readyResourceIndex].version = 0;
      _readyResourceIndex = -1;
    }
  }
  _lodDirty.store(true, std::memory_order_release);
  _workerCv.notify_one();
}

void Renderer::processPendingResources() {
  PreparedResources readyResources;
  size_t readyVersion = 0;
  size_t latestRequested = 0;
  int readyIndex = -1;
  {
    std::lock_guard<std::mutex> lock(_workerMutex);
    latestRequested = _lodRequestedVersion;
    if (_readyResourceIndex != -1 &&
        _resourceBuffers[_readyResourceIndex].ready) {
      readyIndex = _readyResourceIndex;
      readyVersion = _resourceBuffers[readyIndex].version;
      readyResources = std::move(_resourceBuffers[readyIndex].data);
      _resourceBuffers[readyIndex].ready = false;
      _resourceBuffers[readyIndex].version = 0;
      _readyResourceIndex = -1;
    }
  }

  if (readyIndex == -1)
    return;

  if (readyVersion < latestRequested) {
    // Drop stale payload; a newer request is pending.
    return;
  }

  commitPreparedResources(readyResources);

  bool clearDirty = false;
  {
    std::lock_guard<std::mutex> lock(_workerMutex);
    _activeResourceIndex = readyIndex;
    _lodAppliedVersion = readyVersion;
    clearDirty = (_lodRequestedVersion == readyVersion);
  }

  if (clearDirty)
    _lodDirty.store(false, std::memory_order_release);
}

Renderer::PreparedResources
Renderer::buildResourcesForMask(const std::vector<uint8_t> &mask) {
  PreparedResources result;
  result.activeMask = mask;

  Scene activeScene;
  activeScene.screenSize = _pScene->screenSize;
  activeScene.maxRayDepth = _pScene->maxRayDepth;

  for (const auto &object : _allSceneObjects) {
    std::vector<Primitive> activePrims;
    activePrims.reserve(object.primitiveCount);
    for (size_t i = 0; i < object.primitiveCount; ++i) {
      size_t primIndex = object.firstPrimitive + i;
      if (primIndex < mask.size() && mask[primIndex])
        activePrims.push_back(_allPrimitives[primIndex]);
    }
    if (!activePrims.empty())
      activeScene.addObjectSilent(activePrims);
  }

  activeScene.buildBVH();

  result.primitiveCount = activeScene.getPrimitiveCount();
  result.triangleCount = activeScene.getTriangleCount();
  result.blasNodeCount = activeScene.getBVHNodeCount();

  if (result.primitiveCount > 0) {
    simd::float4 *transforms = activeScene.createTransformsBuffer();
    result.transforms.assign(transforms,
                             transforms + result.primitiveCount * 3);
    delete[] transforms;

    simd::float4 *materials = activeScene.createMaterialsBuffer();
    result.materials.assign(materials,
                            materials + result.primitiveCount * 2);
    delete[] materials;
  }

  if (result.blasNodeCount > 0) {
    simd::float4 *bvhData = activeScene.createBVHBuffer();
    result.bvhData.assign(bvhData, bvhData + result.blasNodeCount * 2);
    delete[] bvhData;
  }

  size_t tlasCount = 0;
  simd::float4 *tlasData = activeScene.createTLASBuffer(tlasCount);
  if (tlasData && tlasCount > 0) {
    result.tlasData.assign(tlasData, tlasData + tlasCount * 2);
  }
  delete[] tlasData;
  result.tlasNodeCount = tlasCount;
  result.activeNodeCount = result.tlasNodeCount + result.primitiveCount;

  size_t indexCount = activeScene.getPrimitiveIndices().size();
  if (indexCount > 0) {
    int *rawIndices = activeScene.createPrimitiveIndexBuffer();
    result.primitiveIndices.assign(rawIndices, rawIndices + indexCount);
    delete[] rawIndices;
  }

  activeScene.createTriangleBuffers(result.triangleVertices,
                                    result.triangleIndices);

  const auto &primitives = activeScene.getPrimitives();
  result.lightIndices.reserve(primitives.size());
  result.lightCdf.reserve(primitives.size());
  result.lightTotalWeight = 0.0f;

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

    result.lightTotalWeight += weight;
    result.lightIndices.push_back(static_cast<uint32_t>(i));
    result.lightCdf.push_back(result.lightTotalWeight);
  }

  result.lightCount = result.lightIndices.size();

  return result;
}

void Renderer::commitPreparedResources(const PreparedResources &resources) {
  _residentPrimitiveCount = resources.primitiveCount;
  _residentTriangleCount = resources.triangleCount;
  _blasNodeCount = resources.blasNodeCount;
  _tlasNodeCount = resources.tlasNodeCount;
  _activeNodeCount = resources.activeNodeCount;
  _lightCount = resources.lightCount;
  _lightTotalWeight = resources.lightTotalWeight;

  if (_pSphereBuffer) {
    _pSphereBuffer->release();
    _pSphereBuffer = nullptr;
  }
  if (_pSphereMaterialBuffer) {
    _pSphereMaterialBuffer->release();
    _pSphereMaterialBuffer = nullptr;
  }

  size_t transformSize = resources.transforms.size() * sizeof(simd::float4);
  if (transformSize > 0) {
    _pSphereBuffer = _pDevice->newBuffer(resources.transforms.data(),
                                         transformSize,
                                         MTL::ResourceStorageModeManaged);
    _pSphereBuffer->didModifyRange(NS::Range::Make(0, transformSize));
  } else {
    simd::float4 dummy = {0, 0, 0, 0};
    _pSphereBuffer = _pDevice->newBuffer(&dummy, sizeof(simd::float4),
                                         MTL::ResourceStorageModeManaged);
  }

  size_t materialSize = resources.materials.size() * sizeof(simd::float4);
  if (materialSize > 0) {
    _pSphereMaterialBuffer =
        _pDevice->newBuffer(resources.materials.data(), materialSize,
                            MTL::ResourceStorageModeManaged);
    _pSphereMaterialBuffer->didModifyRange(NS::Range::Make(0, materialSize));
  } else {
    simd::float4 dummy = {0, 0, 0, 0};
    _pSphereMaterialBuffer = _pDevice->newBuffer(
        &dummy, sizeof(simd::float4), MTL::ResourceStorageModeManaged);
  }

  if (_pLightIndexBuffer) {
    _pLightIndexBuffer->release();
    _pLightIndexBuffer = nullptr;
  }
  if (_pLightCdfBuffer) {
    _pLightCdfBuffer->release();
    _pLightCdfBuffer = nullptr;
  }

  size_t lightCount = resources.lightIndices.size();
  size_t lightAlloc = lightCount > 0 ? lightCount : 1;
  _pLightIndexBuffer = _pDevice->newBuffer(lightAlloc * sizeof(uint32_t),
                                           MTL::ResourceStorageModeManaged);
  _pLightCdfBuffer = _pDevice->newBuffer(lightAlloc * sizeof(float),
                                         MTL::ResourceStorageModeManaged);

  if (lightCount > 0) {
    memcpy(_pLightIndexBuffer->contents(), resources.lightIndices.data(),
           lightCount * sizeof(uint32_t));
    _pLightIndexBuffer->didModifyRange(
        NS::Range::Make(0, lightCount * sizeof(uint32_t)));

    memcpy(_pLightCdfBuffer->contents(), resources.lightCdf.data(),
           lightCount * sizeof(float));
    _pLightCdfBuffer->didModifyRange(
        NS::Range::Make(0, lightCount * sizeof(float)));
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

  size_t activeCount = resources.activeMask.size();
  size_t activeAlloc = activeCount > 0 ? activeCount : 1;
  _pActiveBuffer =
      _pDevice->newBuffer(activeAlloc * sizeof(uint8_t),
                          MTL::ResourceStorageModeManaged);
  if (activeCount > 0) {
    memcpy(_pActiveBuffer->contents(), resources.activeMask.data(),
           activeCount * sizeof(uint8_t));
    _pActiveBuffer->didModifyRange(
        NS::Range::Make(0, activeCount * sizeof(uint8_t)));
  } else {
    uint8_t inactive = 0;
    memcpy(_pActiveBuffer->contents(), &inactive, sizeof(uint8_t));
    _pActiveBuffer->didModifyRange(NS::Range::Make(0, sizeof(uint8_t)));
  }

  if (_pTriangleVertexBuffer) {
    _pTriangleVertexBuffer->release();
    _pTriangleVertexBuffer = nullptr;
  }
  if (_pTriangleIndexBuffer) {
    _pTriangleIndexBuffer->release();
    _pTriangleIndexBuffer = nullptr;
  }

  if (!resources.triangleVertices.empty()) {
    size_t vertexSize = resources.triangleVertices.size() * sizeof(simd::float3);
    _pTriangleVertexBuffer = _pDevice->newBuffer(
        resources.triangleVertices.data(), vertexSize,
        MTL::ResourceStorageModeManaged);
    _pTriangleVertexBuffer->didModifyRange(NS::Range::Make(0, vertexSize));
  } else {
    simd::float3 dummyVertex = {0, 0, 0};
    _pTriangleVertexBuffer = _pDevice->newBuffer(
        &dummyVertex, sizeof(simd::float3), MTL::ResourceStorageModeManaged);
  }

  if (!resources.triangleIndices.empty()) {
    size_t indexSize = resources.triangleIndices.size() * sizeof(simd::uint3);
    _pTriangleIndexBuffer = _pDevice->newBuffer(
        resources.triangleIndices.data(), indexSize,
        MTL::ResourceStorageModeManaged);
    _pTriangleIndexBuffer->didModifyRange(NS::Range::Make(0, indexSize));
  } else {
    simd::uint3 dummyIndex = {0, 0, 0};
    _pTriangleIndexBuffer = _pDevice->newBuffer(
        &dummyIndex, sizeof(simd::uint3), MTL::ResourceStorageModeManaged);
  }

  if (_pBVHBuffer) {
    _pBVHBuffer->release();
    _pBVHBuffer = nullptr;
  }
  if (resources.bvhData.empty()) {
    _pBVHBuffer = _pDevice->newBuffer(sizeof(simd::float4),
                                      MTL::ResourceStorageModeManaged);
  } else {
    size_t bvhSize = resources.bvhData.size() * sizeof(simd::float4);
    _pBVHBuffer = _pDevice->newBuffer(resources.bvhData.data(), bvhSize,
                                      MTL::ResourceStorageModeManaged);
    _pBVHBuffer->didModifyRange(NS::Range::Make(0, bvhSize));
  }

  if (_pTLASBuffer) {
    _pTLASBuffer->release();
    _pTLASBuffer = nullptr;
  }
  if (resources.tlasData.empty()) {
    _pTLASBuffer = _pDevice->newBuffer(sizeof(simd::float4),
                                       MTL::ResourceStorageModeManaged);
  } else {
    size_t tlasSize = resources.tlasData.size() * sizeof(simd::float4);
    _pTLASBuffer = _pDevice->newBuffer(resources.tlasData.data(), tlasSize,
                                       MTL::ResourceStorageModeManaged);
    _pTLASBuffer->didModifyRange(NS::Range::Make(0, tlasSize));
  }

  if (_pPrimitiveIndexBuffer) {
    _pPrimitiveIndexBuffer->release();
    _pPrimitiveIndexBuffer = nullptr;
  }

  if (!resources.primitiveIndices.empty()) {
    size_t indexSize = resources.primitiveIndices.size() * sizeof(int);
    _pPrimitiveIndexBuffer = _pDevice->newBuffer(
        resources.primitiveIndices.data(), indexSize,
        MTL::ResourceStorageModeManaged);
    _pPrimitiveIndexBuffer->didModifyRange(NS::Range::Make(0, indexSize));
  } else {
    _pPrimitiveIndexBuffer = _pDevice->newBuffer(sizeof(int),
                                                 MTL::ResourceStorageModeManaged);
  }
}

void Renderer::rebuildWorkerLoop() {
  while (true) {
    std::vector<uint8_t> mask;
    size_t targetVersion = 0;
    int targetIndex = -1;

    {
      std::unique_lock<std::mutex> lock(_workerMutex);
      _workerCv.wait(lock, [this]() {
        return _workerExit ||
               (_lodRequestedVersion > _lodBuildingVersion);
      });

      if (_workerExit)
        break;

      targetVersion = _lodRequestedVersion;
      mask = _pendingActiveMask;
      targetIndex = 1 - _activeResourceIndex;
      if (_readyResourceIndex == targetIndex) {
        _resourceBuffers[targetIndex].ready = false;
        _resourceBuffers[targetIndex].data = PreparedResources();
        _resourceBuffers[targetIndex].version = 0;
        _readyResourceIndex = -1;
      }
      _inFlightResourceIndex = targetIndex;
      _lodBuildingVersion = targetVersion;
      _rebuildInProgress = true;
    }

    PreparedResources resources = buildResourcesForMask(mask);

    {
      std::lock_guard<std::mutex> lock(_workerMutex);
      if (_workerExit)
        break;

      ResourceBuffer &buffer = _resourceBuffers[targetIndex];
      buffer.data = std::move(resources);
      buffer.version = targetVersion;
      buffer.ready = true;
      _readyResourceIndex = targetIndex;
      _inFlightResourceIndex = -1;
      _rebuildInProgress = false;
    }

    _workerCv.notify_all();
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
  updateLODByDistance();
  processPendingResources();
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

  pEnc->setFragmentTexture(_accumulationTargets[0], 0);
  pEnc->setFragmentTexture(_accumulationTargets[1], 1);

  pEnc->drawPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle,
                       NS::UInteger(0), NS::UInteger(6));

  pEnc->endEncoding();

  MTL::BlitCommandEncoder *pBlit = pCmd->blitCommandEncoder();
  pBlit->endEncoding();

  pCmd->presentDrawable(pView->currentDrawable());
  pCmd->commit();

  pPool->release();
}

void Renderer::updateLODByDistance() {
  // Keep primitives active until the camera is reasonably far away.
  // Using a larger threshold prevents the entire scene from being culled
  // when starting far from the origin.
  const float FULL_DETAIL_DISTANCE = 250.0f;
  bool changed = false;
  std::vector<uint8_t> snapshot;
  {
    std::lock_guard<std::mutex> lock(_primitiveMutex);
    snapshot.resize(_activePrimitive.size());
    size_t activeCount = 0;
    for (size_t g = 0; g < _activePrimitive.size(); ++g) {
      float dist =
          simd::length(_primitiveBounds[g].center - Camera::position) -
          _primitiveBounds[g].radius;
      dist = std::max(dist, 0.0f);
      bool shouldBeActive = dist < FULL_DETAIL_DISTANCE;
      if (_activePrimitive[g] != shouldBeActive) {
        _activePrimitive[g] = shouldBeActive;
        changed = true;
      }
      snapshot[g] = _activePrimitive[g] ? 1 : 0;
      if (_activePrimitive[g])
        activeCount++;
    }

    if (activeCount == 0 && !_activePrimitive.empty()) {
      if (!_activePrimitive[0]) {
        _activePrimitive[0] = true;
        snapshot[0] = 1;
        changed = true;
      }
    }
  }

  if (changed) {
    enqueueLODRebuild(snapshot);
  }
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
  std::vector<uint8_t> snapshot;
  {
    std::lock_guard<std::mutex> lock(_primitiveMutex);
    if (index >= _activePrimitive.size())
      return false;
    if (_activePrimitive[index] == active)
      return false;
    _activePrimitive[index] = active;
    snapshot.resize(_activePrimitive.size());
    for (size_t i = 0; i < _activePrimitive.size(); ++i)
      snapshot[i] = _activePrimitive[i] ? 1 : 0;
  }

  enqueueLODRebuild(snapshot);
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
