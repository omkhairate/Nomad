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

  for (auto &inst : _blasInstances) {
    if (inst.buffer)
      inst.buffer->release();
  }

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

  // ✅ Unified buffer
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
    std::vector<uint8_t> activeBytes(primitiveCount);
    for (size_t i = 0; i < primitiveCount; ++i)
      activeBytes[i] = _activePrimitive[i] ? 1 : 0;
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
  // Keep primitives active until the camera is reasonably far away.
  // Using a larger threshold prevents the entire scene from being culled
  // when starting far from the origin.
  const float FULL_DETAIL_DISTANCE = 250.0f;
  size_t activeCount = 0;
  for (size_t g = 0; g < _allPrimitives.size(); ++g) {
    float dist =
        simd::length(_primitiveBounds[g].center - Camera::position) -
        _primitiveBounds[g].radius;
    dist = std::max(dist, 0.0f);
    bool shouldBeActive = dist < FULL_DETAIL_DISTANCE;
    if (_activePrimitive[g] != shouldBeActive)
      setPrimitiveActive(g, shouldBeActive);
    if (_activePrimitive[g])
      activeCount++;
  }

  if (activeCount == 0 && !_activePrimitive.empty()) {
    // Ensure at least one primitive remains visible to avoid a blank scene
    setPrimitiveActive(0, true);
    activeCount = 1;
  }

  size_t newActiveNodes = _tlasNodeCount + activeCount;
  if (newActiveNodes != _activeNodeCount) {
    _activeNodeCount = newActiveNodes;
    printf("Active nodes: %zu\n", _activeNodeCount);
  }

  if (_tlasDirty)
    rebuildTLAS();
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

bool Renderer::ensureBLASResident(size_t index) {
  if (index >= _blasInstances.size())
    return false;

  BLASInstance &inst = _blasInstances[index];
  if (inst.buffer)
    return true;

  if (inst.nodeCount == 0)
    return false;

  size_t nodeBytes = sizeof(simd::float4) * inst.nodeCount * 2;
  if (nodeBytes == 0)
    nodeBytes = sizeof(simd::float4);

  if (!ensureBudget(nodeBytes))
    return false;

  inst.buffer =
      _pDevice->newBuffer(nodeBytes, MTL::ResourceStorageModeManaged);
  if (!inst.buffer)
    return false;

  if (index < _blasCPUData.size()) {
    simd::float4 *dst = static_cast<simd::float4 *>(inst.buffer->contents());
    dst[0] = _blasCPUData[index].nodeMin;
    dst[1] = _blasCPUData[index].nodeMax;
    inst.buffer->didModifyRange(NS::Range::Make(0, nodeBytes));
  }

  return true;
}

void Renderer::releaseBLAS(size_t index) {
  if (index >= _blasInstances.size())
    return;

  BLASInstance &inst = _blasInstances[index];
  if (inst.buffer) {
    inst.buffer->release();
    inst.buffer = nullptr;
  }
  inst.nodeOffset = 0;
}

void Renderer::rebuildTLAS() {
  std::vector<simd::float4> aggregatedNodes;
  std::vector<simd::float4> tlasData;
  size_t totalNodes = 0;
  size_t activeCount = 0;
  size_t previousActiveNodes = _activeNodeCount;

  aggregatedNodes.reserve(_activePrimitive.size() * 2);
  tlasData.reserve(_activePrimitive.size() * 2);

  for (size_t i = 0; i < _activePrimitive.size(); ++i) {
    if (!_activePrimitive[i])
      continue;

    if (!ensureBLASResident(i))
      continue;

    BLASInstance &inst = _blasInstances[i];
    if (!inst.buffer)
      continue;

    inst.nodeOffset = static_cast<uint32_t>(totalNodes);

    simd::float4 *src =
        static_cast<simd::float4 *>(inst.buffer->contents());
    for (uint32_t n = 0; n < inst.nodeCount; ++n) {
      aggregatedNodes.push_back(src[2 * n + 0]);
      aggregatedNodes.push_back(src[2 * n + 1]);
    }

    simd::float4 minNode = src[0];
    uint32_t offset = inst.nodeOffset;
    std::memcpy(&minNode.w, &offset, sizeof(uint32_t));
    simd::float4 maxNode = src[1];
    maxNode.w = 0.0f;

    tlasData.push_back(minNode);
    tlasData.push_back(maxNode);

    totalNodes += inst.nodeCount;
    activeCount++;
  }

  if (_pBVHBuffer) {
    _pBVHBuffer->release();
    _pBVHBuffer = nullptr;
  }

  size_t bvhSize = aggregatedNodes.size() * sizeof(simd::float4);
  if (bvhSize == 0)
    bvhSize = sizeof(simd::float4);
  if (!ensureBudget(bvhSize))
    bvhSize = sizeof(simd::float4);

  _pBVHBuffer =
      _pDevice->newBuffer(bvhSize, MTL::ResourceStorageModeManaged);
  if (!aggregatedNodes.empty()) {
    memcpy(_pBVHBuffer->contents(), aggregatedNodes.data(),
           aggregatedNodes.size() * sizeof(simd::float4));
    _pBVHBuffer->didModifyRange(
        NS::Range::Make(0, aggregatedNodes.size() * sizeof(simd::float4)));
  }

  if (_pTLASBuffer) {
    _pTLASBuffer->release();
    _pTLASBuffer = nullptr;
  }

  size_t tlasSize = tlasData.size() * sizeof(simd::float4);
  if (tlasSize == 0)
    tlasSize = sizeof(simd::float4);
  if (!ensureBudget(tlasSize))
    tlasSize = sizeof(simd::float4);

  _pTLASBuffer =
      _pDevice->newBuffer(tlasSize, MTL::ResourceStorageModeManaged);
  if (!tlasData.empty()) {
    memcpy(_pTLASBuffer->contents(), tlasData.data(),
           tlasData.size() * sizeof(simd::float4));
    _pTLASBuffer->didModifyRange(
        NS::Range::Make(0, tlasData.size() * sizeof(simd::float4)));
  }

  _tlasCPUData = tlasData;

  _blasNodeCount = totalNodes;
  _tlasNodeCount = activeCount;
  _totalNodeCount = _tlasNodeCount + _allPrimitives.size();
  _activeNodeCount = _tlasNodeCount + activeCount;
  if (_activeNodeCount != previousActiveNodes)
    printf("Active nodes: %zu\n", _activeNodeCount);
  _tlasDirty = false;
}

void Renderer::setPrimitiveActive(size_t index, bool active) {
  if (index >= _activePrimitive.size())
    return;
  if (_activePrimitive[index] == active)
    return;
  if (active) {
    if (!ensureBLASResident(index))
      return;
  } else {
    releaseBLAS(index);
  }
  _activePrimitive[index] = active;
  _tlasDirty = true;
  if (_pActiveBuffer) {
    uint8_t *mask = static_cast<uint8_t *>(_pActiveBuffer->contents());
    mask[index] = active ? 1 : 0;
    _pActiveBuffer->didModifyRange(NS::Range::Make(index, 1));
  }
}

void Renderer::rebuildAccelerationStructures() {
  _pScene->buildBVH();

  for (auto &inst : _blasInstances) {
    if (inst.buffer)
      inst.buffer->release();
  }
  _blasInstances.clear();
  _blasCPUData.clear();

  size_t primitiveCount = _pScene->getPrimitiveCount();
  const auto &scenePrims = _pScene->getPrimitives();
  _allPrimitives = scenePrims;
  _primitiveBounds.resize(primitiveCount);
  _activePrimitive.resize(primitiveCount, true);
  _blasCPUData.resize(primitiveCount);
  _blasInstances.resize(primitiveCount);

  auto encodeInt = [](int value) {
    float encoded;
    std::memcpy(&encoded, &value, sizeof(int));
    return encoded;
  };

  for (size_t i = 0; i < primitiveCount; ++i) {
    const Primitive &p = scenePrims[i];

    simd::float3 bMin = simd::make_float3(0.0f, 0.0f, 0.0f);
    simd::float3 bMax = simd::make_float3(0.0f, 0.0f, 0.0f);
    BoundingSphere bounds{};

    if (p.type == PrimitiveType::Sphere) {
      const auto &s = p.sphere;
      simd::float3 radius =
          simd::make_float3(s.radius, s.radius, s.radius);
      bMin = s.center - radius;
      bMax = s.center + radius;
      bounds.center = s.center;
      bounds.radius = s.radius;
    } else if (p.type == PrimitiveType::Triangle) {
      bMin = simd::min(p.triangle.v0,
                       simd::min(p.triangle.v1, p.triangle.v2));
      bMax = simd::max(p.triangle.v0,
                       simd::max(p.triangle.v1, p.triangle.v2));
      simd::float3 center =
          (p.triangle.v0 + p.triangle.v1 + p.triangle.v2) / 3.0f;
      float r = simd::length(p.triangle.v0 - center);
      r = std::max(r, (float)simd::length(p.triangle.v1 - center));
      r = std::max(r, (float)simd::length(p.triangle.v2 - center));
      bounds.center = center;
      bounds.radius = r;
    } else {
      const auto &r = p.rectangle;
      simd::float3 c1 = r.center - r.u - r.v;
      simd::float3 c2 = r.center - r.u + r.v;
      simd::float3 c3 = r.center + r.u - r.v;
      simd::float3 c4 = r.center + r.u + r.v;
      bMin = simd::min(simd::min(c1, c2), simd::min(c3, c4));
      bMax = simd::max(simd::max(c1, c2), simd::max(c3, c4));
      bounds.center = r.center;
      bounds.radius = simd::length(r.u) + simd::length(r.v);
    }

    int leftFirst = static_cast<int>(i);
    int count = 1;

    _blasCPUData[i].nodeMin =
        simd::make_float4(bMin, encodeInt(leftFirst));
    _blasCPUData[i].nodeMax = simd::make_float4(bMax, encodeInt(count));

    _blasInstances[i].nodeCount = 1;
    _blasInstances[i].nodeOffset = 0;

    if (i < _primitiveBounds.size())
      _primitiveBounds[i] = bounds;

    if (i < _activePrimitive.size() && _activePrimitive[i])
      ensureBLASResident(i);
  }

  _tlasDirty = true;
  rebuildTLAS();

  size_t indexCount = primitiveCount;
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

  size_t tlasCount = _tlasCPUData.size() / 2;
  out << "  \"tlas\": [\n";
  for (size_t i = 0; i < tlasCount; ++i) {
    const simd::float4 &bmin = _tlasCPUData[2 * i];
    const simd::float4 &bmax = _tlasCPUData[2 * i + 1];
    uint32_t offset = 0;
    std::memcpy(&offset, &bmin.w, sizeof(uint32_t));
    out << "    {\"index\":" << i << ",\"offset\":" << offset
        << ",\"min\":[" << bmin.x << "," << bmin.y << "," << bmin.z
        << "],\"max\":[" << bmax.x << "," << bmax.y << "," << bmax.z
        << "]}";
    if (i + 1 < tlasCount)
      out << ",\n";
    else
      out << "\n";
  }
  out << "  ],\n";

  out << "  \"blas\": [\n";
  for (size_t i = 0; i < _blasCPUData.size(); ++i) {
    const auto &n = _blasCPUData[i];
    int leftFirst = 0;
    int count = 0;
    std::memcpy(&leftFirst, &n.nodeMin.w, sizeof(int));
    std::memcpy(&count, &n.nodeMax.w, sizeof(int));
    bool resident = i < _blasInstances.size() && _blasInstances[i].buffer;
    out << "    {\"index\":" << i << ",\"leftFirst\":" << leftFirst
        << ",\"count\":" << count << ",\"resident\":"
        << (resident ? "true" : "false") << ",\"min\":[" << n.nodeMin.x
        << "," << n.nodeMin.y << "," << n.nodeMin.z << "],\"max\":["
        << n.nodeMax.x << "," << n.nodeMax.y << "," << n.nodeMax.z
        << "]}";
    if (i + 1 < _blasCPUData.size())
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
