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
  _bufferedPrimitiveCount = _pScene->getPrimitiveCount();
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

  rebuildAccelerationStructures();
  buildBuffers();
  _bufferedPrimitiveCount = _pScene->getPrimitiveCount();
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

  printf("viewportU: (%f, %f, %f)\n", viewportU.x, viewportU.y, viewportU.z);
  printf("viewportV: (%f, %f, %f)\n", viewportV.x, viewportV.y, viewportV.z);
  printf("firstPixel: (%f, %f, %f)\n", firstPixelPosition.x,
         firstPixelPosition.y, firstPixelPosition.z);
}

void Renderer::buildBuffers() {
  const size_t primitiveCount = _pScene->getPrimitiveCount();
  const size_t uniformsDataSize = sizeof(UniformsData);

  // Uniforms
  if (_pUniformsBuffer)
    _pUniformsBuffer->release();
  if (!ensureBudget(uniformsDataSize))
    return;
  _pUniformsBuffer =
      _pDevice->newBuffer(uniformsDataSize, MTL::ResourceStorageModeManaged);
  _pUniformsBuffer->didModifyRange(NS::Range::Make(0, uniformsDataSize));

  // Destroy previous
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
    primitiveBuffer =
        _pScene->createTransformsBuffer(); // 3 float4s per primitive
    materialBuffer =
        _pScene->createMaterialsBuffer(); // 2 float4s per primitive
  }

  const size_t primitiveSize = primitiveCount * 3 * sizeof(simd::float4);
  const size_t materialSize = primitiveCount * 2 * sizeof(simd::float4);

  size_t primitiveAlloc =
      primitiveSize > 0 ? primitiveSize : sizeof(simd::float4);
  size_t materialAlloc = materialSize > 0 ? materialSize : sizeof(simd::float4);
  if (!ensureBudget(primitiveAlloc + materialAlloc)) {
    delete[] primitiveBuffer;
    delete[] materialBuffer;
    return;
  }
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

  // Active mask buffer (1 byte per primitive)
  if (_pActiveBuffer) {
    _pActiveBuffer->release();
    _pActiveBuffer = nullptr;
  }
  size_t activeSize = primitiveCount * sizeof(uint8_t);
  size_t activeAlloc = activeSize > 0 ? activeSize : sizeof(uint8_t);
  if (!ensureBudget(activeAlloc))
    return;
  _pActiveBuffer =
      _pDevice->newBuffer(activeAlloc, MTL::ResourceStorageModeManaged);
  if (primitiveCount > 0) {
    std::vector<uint8_t> activeBytes(primitiveCount, 1);
    memcpy(_pActiveBuffer->contents(), activeBytes.data(), activeSize);
    _pActiveBuffer->didModifyRange(NS::Range::Make(0, activeSize));
  }

  // Dummy triangle buffer bindings
  simd::float3 dummyVertex = {0, 0, 0};
  simd::uint3 dummyIndex = {0, 0, 0};
  if (!ensureBudget(sizeof(simd::float3)))
    return;
  _pTriangleVertexBuffer = _pDevice->newBuffer(
      &dummyVertex, sizeof(simd::float3), MTL::ResourceStorageModeManaged);
  if (!ensureBudget(sizeof(simd::uint3)))
    return;
  _pTriangleIndexBuffer = _pDevice->newBuffer(&dummyIndex, sizeof(simd::uint3),
                                              MTL::ResourceStorageModeManaged);
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
  size_t texSize = static_cast<size_t>(Camera::screenSize.x * Camera::screenSize.y *
                                       4 * sizeof(float));
  for (uint i = 0; i < 2; i++) {
    if (!ensureBudget(texSize)) {
      textureDescriptor->release();
      return;
    }
    _accumulationTargets[i] = _pDevice->newTexture(textureDescriptor);
  }
  textureDescriptor->release();
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

  u.primitiveCount = _pScene->getPrimitiveCount();
  u.triangleCount = _pScene->getTriangleCount();
  u.totalPrimitiveCount = _allPrimitives.size();
  u.tlasNodeCount = _tlasNodeCount;
  u.blasNodeCount = _blasNodeCount;
  u.maxRayDepth = _pScene->maxRayDepth;
  u.debugAS = InputSystem::debugAS;

  _pUniformsBuffer->didModifyRange(NS::Range::Make(0, sizeof(UniformsData)));
}

void Renderer::draw(MTK::View *pView) {
  updateLODByDistance();
  if (_pendingSync && ++_framesSinceSyncRequest >= kSyncDelayFrames) {
    syncBuffersToActive();
    _pendingSync = false;
  }
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
  // Determine residency based solely on whether primitives are visible.
  // Primitives outside the camera frustum are offloaded and will be
  // reloaded once they come back into view.
  bool changed = false;
  size_t activeCount = 0;
  for (size_t g = 0; g < _allPrimitives.size(); ++g) {
    bool shouldBeActive = isInView(_primitiveBounds[g]);
    if (_activePrimitive[g] != shouldBeActive) {
      _activePrimitive[g] = shouldBeActive;
      changed = true;
    }
    if (_activePrimitive[g])
      activeCount++;
  }

  if (activeCount == 0 && !_activePrimitive.empty()) {
    _activePrimitive[0] = true;
    activeCount = 1;
    changed = true;
  }

  if (changed && activeCount != _bufferedPrimitiveCount) {
    _pendingSync = true;
    _framesSinceSyncRequest = 0;
  }
}

void Renderer::syncBuffersToActive() {
  Scene *newScene = new Scene();
  newScene->screenSize = _pScene->screenSize;
  newScene->maxRayDepth = _pScene->maxRayDepth;
  newScene->cameraPath = _pScene->cameraPath;

  for (size_t i = 0; i < _allPrimitives.size(); ++i)
    if (_activePrimitive[i])
      newScene->addPrimitive(_allPrimitives[i]);

  delete _pScene;
  _pScene = newScene;

  rebuildAccelerationStructures();
  buildBuffers();
  _bufferedPrimitiveCount = _pScene->getPrimitiveCount();
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

void Renderer::rebuildAccelerationStructures() {
  _pScene->buildBVH();

  size_t newBlasCount = _pScene->getBVHNodeCount();
  _blasNodeCount = newBlasCount;

  simd::float4 *bvhData = _pScene->createBVHBuffer();
  if (_pBVHBuffer)
    _pBVHBuffer->release();
  size_t bvhSize = sizeof(simd::float4) * newBlasCount * 2;
  if (!ensureBudget(bvhSize)) {
    delete[] bvhData;
    return;
  }
  _pBVHBuffer =
      _pDevice->newBuffer(bvhData, bvhSize, MTL::ResourceStorageModeManaged);
  _pBVHBuffer->didModifyRange(NS::Range::Make(0, _pBVHBuffer->length()));
  delete[] bvhData;

  size_t tlasCount = 0;
  simd::float4 *tlasData = _pScene->createTLASBuffer(tlasCount);
  _tlasNodeCount = tlasCount;
  _totalNodeCount = _tlasNodeCount + _allPrimitives.size();
  size_t oldActiveCount = _activeNodeCount;
  size_t activePrim = 0;
  for (bool a : _activePrimitive)
    if (a)
      activePrim++;
  _activeNodeCount = _tlasNodeCount + activePrim;
  if (_activeNodeCount != oldActiveCount) {
    printf("Active nodes: %zu\n", _activeNodeCount);
  }
  if (_pTLASBuffer)
    _pTLASBuffer->release();
  if (tlasData && tlasCount > 0) {
    size_t tlasSize = sizeof(simd::float4) * tlasCount * 2;
    if (!ensureBudget(tlasSize)) {
      delete[] tlasData;
      return;
    }
    _pTLASBuffer =
        _pDevice->newBuffer(tlasData, tlasSize, MTL::ResourceStorageModeManaged);
    _pTLASBuffer->didModifyRange(NS::Range::Make(0, _pTLASBuffer->length()));
  } else {
    if (!ensureBudget(1)) {
      delete[] tlasData;
      return;
    }
    _pTLASBuffer = _pDevice->newBuffer(1, MTL::ResourceStorageModeManaged);
  }
  delete[] tlasData;

  size_t indexCount = _pScene->getPrimitiveCount();
  size_t indexSize = indexCount * sizeof(int);
  if (_pPrimitiveIndexBuffer)
    _pPrimitiveIndexBuffer->release();
  if (indexCount > 0) {
    if (!ensureBudget(indexSize))
      return;
    int *rawIndices = _pScene->createPrimitiveIndexBuffer();
    _pPrimitiveIndexBuffer = _pDevice->newBuffer(
        rawIndices, indexSize, MTL::ResourceStorageModeManaged);
    _pPrimitiveIndexBuffer->didModifyRange(NS::Range::Make(0, indexSize));
    delete[] rawIndices;
  } else {
    if (!ensureBudget(sizeof(int)))
      return;
    _pPrimitiveIndexBuffer =
        _pDevice->newBuffer(sizeof(int), MTL::ResourceStorageModeManaged);
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
    int childIndex = reinterpret_cast<const int *>(&bmin)[3];
    out << "    {\"child\":" << childIndex << ",\"min\":[" << bmin.x << ","
        << bmin.y << "," << bmin.z << "],\"max\":[" << bmax.x << "," << bmax.y
        << "," << bmax.z << "]}";
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

void Renderer::setGPUMemoryBudgetMB(double mb) {
  if (mb <= 0.0)
    _gpuMemoryBudget = std::numeric_limits<size_t>::max();
  else
    _gpuMemoryBudget = static_cast<size_t>(mb * 1024.0 * 1024.0);
}

bool Renderer::ensureBudget(size_t bytes) const {
  if (_gpuMemoryBudget == std::numeric_limits<size_t>::max())
    return true;
  size_t current = _pDevice->currentAllocatedSize();
  if (current + bytes > _gpuMemoryBudget) {
    printf("GPU memory budget exceeded: requested %zu bytes (current %zu, budget %zu)\n",
           bytes, current, _gpuMemoryBudget);
    return false;
  }
  return true;
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
