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
#include <unordered_map>
#include <unordered_set>
#include <CoreFoundation/CoreFoundation.h>

using namespace MetalCppPathTracer;

static constexpr size_t kMaxInstanceCapacity = 16384;
static constexpr size_t kTLASNodeFootprintBytes = sizeof(simd::float4) * 2;
static constexpr int kMaxTLASLeafInstances = 8;
static constexpr size_t kDefaultBudgetBytes = 256ull * 1024ull * 1024ull;
static constexpr double kBudgetDecreaseFactor = 0.8;
static constexpr double kBudgetIncreaseFactor = 1.1;
static constexpr double kBudgetIncreaseThreshold = 0.6;
static constexpr double kBudgetDecreaseThreshold = 1.2;
static constexpr size_t kPrimitiveStride = 4;

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
  uint32_t residentInstanceCount;
  uint32_t totalInstanceCount;
  uint32_t lightCount;
  uint32_t paddingUniforms[3];
};

struct InstanceMetadataCPU {
  uint32_t primitiveCount;
  uint32_t blasNodeCount;
  uint32_t rootNodeIndex;
  uint32_t padding;
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

Renderer::Renderer(MTL::Device *pDevice)
    : _pDevice(pDevice->retain()), _pScene(new Scene()) {
  _pCommandQueue = _pDevice->newCommandQueue();

  uint64_t recommended = _pDevice->recommendedMaxWorkingSetSize();
  if (recommended > 0 &&
      recommended < std::numeric_limits<uint64_t>::max()) {
    _recommendedBudget = static_cast<size_t>(recommended);
    _gpuMemoryBudget = _recommendedBudget;
    double mb = static_cast<double>(_gpuMemoryBudget) / (1024.0 * 1024.0);
    printf("GPU memory budget defaulting to %.2f MB (recommended).\n", mb);
  } else {
    _recommendedBudget = kDefaultBudgetBytes;
    _gpuMemoryBudget = _recommendedBudget;
    double mb = static_cast<double>(_gpuMemoryBudget) / (1024.0 * 1024.0);
    printf("GPU memory budget defaulting to %.2f MB (fallback).\n", mb);
  }

  Camera::reset();

  buildShaders();
  updateVisibleScene();
  buildTextures();

  recalculateViewport();
}

Renderer::~Renderer() {
  for (size_t i = 0; i < _instances.size(); ++i)
    releaseInstanceResources(i);

  if (_pTriangleVertexBuffer)
    _pTriangleVertexBuffer->release();
  if (_pTriangleIndexBuffer)
    _pTriangleIndexBuffer->release();
  if (_pUniformsBuffer)
    _pUniformsBuffer->release();
  if (_pTLASBuffer)
    _pTLASBuffer->release();
  if (_pTLASInstanceIndexBuffer)
    _pTLASInstanceIndexBuffer->release();
  if (_pInstanceMetaBuffer)
    _pInstanceMetaBuffer->release();
  if (_pInstanceArgBuffer)
    _pInstanceArgBuffer->release();
  if (_pInstanceArgElementEncoder)
    _pInstanceArgElementEncoder->release();
  if (_pInstanceArgEncoder)
    _pInstanceArgEncoder->release();
  if (_pLightBuffer) {
    _pLightBuffer->release();
    _pLightBuffer = nullptr;
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

  if (_pInstanceArgEncoder) {
    _pInstanceArgEncoder->release();
    _pInstanceArgEncoder = nullptr;
  }
  if (_pInstanceArgElementEncoder) {
    _pInstanceArgElementEncoder->release();
    _pInstanceArgElementEncoder = nullptr;
  }
  _instanceArgStride = 0;
  _pInstanceArgEncoder = pFragFn->newArgumentEncoder(NS::UInteger(3));
  if (_pInstanceArgEncoder) {
    // Build an element encoder that matches InstanceResources { id(0)..id(3) }
    MTL::ArgumentDescriptor* d0 = MTL::ArgumentDescriptor::argumentDescriptor();
    d0->setIndex(NS::UInteger(0));
    d0->setDataType(MTL::DataType::DataTypePointer);
    d0->setAccess(MTL::BindingAccess::ArgumentAccessReadOnly);
    MTL::ArgumentDescriptor* d1 = MTL::ArgumentDescriptor::argumentDescriptor();
    d1->setIndex(NS::UInteger(1));
    d1->setDataType(MTL::DataType::DataTypePointer);
    d1->setAccess(MTL::BindingAccess::ArgumentAccessReadOnly);
    MTL::ArgumentDescriptor* d2 = MTL::ArgumentDescriptor::argumentDescriptor();
    d2->setIndex(NS::UInteger(2));
    d2->setDataType(MTL::DataType::DataTypePointer);
    d2->setAccess(MTL::BindingAccess::ArgumentAccessReadOnly);
    MTL::ArgumentDescriptor* d3 = MTL::ArgumentDescriptor::argumentDescriptor();
    d3->setIndex(NS::UInteger(3));
    d3->setDataType(MTL::DataType::DataTypePointer);
    d3->setAccess(MTL::BindingAccess::ArgumentAccessReadOnly);
    CFTypeRef descs[] = { (CFTypeRef)d0, (CFTypeRef)d1, (CFTypeRef)d2, (CFTypeRef)d3 };
    NS::Array* descArray = (NS::Array*)CFArrayCreate(kCFAllocatorDefault, descs, 4, &kCFTypeArrayCallBacks);
    _pInstanceArgElementEncoder = _pDevice->newArgumentEncoder(descArray);
    descArray->release();
    if (_pInstanceArgElementEncoder) {
      size_t elementLength = _pInstanceArgElementEncoder->encodedLength();
      size_t elementAlignment = _pInstanceArgElementEncoder->alignment();
      if (elementAlignment > 0) {
        size_t remainder = elementLength % elementAlignment;
        if (remainder != 0)
          elementLength += elementAlignment - remainder;
      }
      _instanceArgStride = elementLength;
    }
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

  // Store full primitive list and initialize per-instance data
  _allPrimitives = _pScene->getPrimitives();
  initializeInstances();
  buildBuffers();
  preloadInitialResidency();
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
  const size_t uniformsDataSize = sizeof(UniformsData);

  // Uniforms
  if (_pUniformsBuffer)
    _pUniformsBuffer->release();
  if (!ensureBudget(uniformsDataSize))
    return;
  _pUniformsBuffer =
      _pDevice->newBuffer(uniformsDataSize, MTL::ResourceStorageModeManaged);
  _pUniformsBuffer->didModifyRange(NS::Range::Make(0, uniformsDataSize));

  if (_pInstanceMetaBuffer) {
    _pInstanceMetaBuffer->release();
    _pInstanceMetaBuffer = nullptr;
  }
  size_t instanceCount = _instances.size();
  size_t metaCount = instanceCount > 0 ? instanceCount : 1;
  size_t metaSize = metaCount * sizeof(InstanceMetadataCPU);
  if (!ensureBudget(metaSize))
    return;
  _pInstanceMetaBuffer =
      _pDevice->newBuffer(metaSize, MTL::ResourceStorageModeManaged);
  std::memset(_pInstanceMetaBuffer->contents(), 0, metaSize);
  _pInstanceMetaBuffer->didModifyRange(NS::Range::Make(0, metaSize));

  if (_pInstanceArgBuffer) {
    _pInstanceArgBuffer->release();
    _pInstanceArgBuffer = nullptr;
  }
  if (_pInstanceArgEncoder) {
    size_t argLength = _pInstanceArgEncoder->encodedLength();
    if (!ensureBudget(argLength))
      return;
    _pInstanceArgBuffer =
        _pDevice->newBuffer(argLength, MTL::ResourceStorageModeShared);
    std::memset(_pInstanceArgBuffer->contents(), 0, argLength);
    _pInstanceArgEncoder->setArgumentBuffer(_pInstanceArgBuffer, 0);
  }

  rebuildTLAS();
}

void Renderer::preloadInitialResidency() {
  if (_instances.empty()) {
    _minInstanceFootprint = 0;
    _recentVisibleFootprint = 0;
    return;
  }

  struct Candidate {
    size_t index;
    float distance;
    size_t footprint;
  };

  std::vector<Candidate> candidates;
  candidates.reserve(_instances.size());

  float closestDistance = std::numeric_limits<float>::max();
  size_t fallbackIndex = std::numeric_limits<size_t>::max();
  size_t totalRequired = 0;
  size_t minFootprint = std::numeric_limits<size_t>::max();

  for (size_t i = 0; i < _instances.size(); ++i) {
    const InstanceRecord &inst = _instances[i];
    if (inst.cpuPrimitiveCount == 0)
      continue;

    size_t footprint = instanceFootprintBytes(inst) + kTLASNodeFootprintBytes;
    if (footprint == 0)
      continue;

    float dist = simd::length(inst.bounds.center - Camera::position) -
                 inst.bounds.radius;
    dist = std::max(dist, 0.0f);
    candidates.push_back({i, dist, footprint});
    totalRequired += footprint;
    minFootprint = std::min(minFootprint, footprint);
    if (dist < closestDistance) {
      closestDistance = dist;
      fallbackIndex = i;
    }
  }

  if (candidates.empty()) {
    _minInstanceFootprint = 0;
    _recentVisibleFootprint = 0;
    return;
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) {
              if (a.distance == b.distance)
                return a.index < b.index;
              return a.distance < b.distance;
            });

  bool hasBudget = _gpuMemoryBudget != std::numeric_limits<size_t>::max();
  if (!_manualBudget && hasBudget && totalRequired > 0 &&
      totalRequired > _gpuMemoryBudget) {
    size_t previousBudget = _gpuMemoryBudget;
    _gpuMemoryBudget = totalRequired;
    _recommendedBudget = std::max(_recommendedBudget, _gpuMemoryBudget);
    double prevMB =
        static_cast<double>(previousBudget) / (1024.0 * 1024.0);
    double newMB =
        static_cast<double>(_gpuMemoryBudget) / (1024.0 * 1024.0);
    printf("Expanded GPU memory budget from %.2f MB to %.2f MB for initial "
           "residency (%zu instances).\n",
           prevMB, newMB, candidates.size());
    hasBudget = _gpuMemoryBudget != std::numeric_limits<size_t>::max();
  }

  size_t budgetLimit = hasBudget ? _gpuMemoryBudget
                                 : std::numeric_limits<size_t>::max();
  size_t preloadedBytes = 0;
  size_t streamedCount = 0;

  for (const Candidate &candidate : candidates) {
    if (hasBudget &&
        (candidate.footprint > budgetLimit ||
         preloadedBytes + candidate.footprint > budgetLimit))
      continue;

    if (streamInInstance(candidate.index)) {
      preloadedBytes += candidate.footprint;
      streamedCount++;
    }
  }

  if (streamedCount == 0 && fallbackIndex != std::numeric_limits<size_t>::max()) {
    size_t footprint =
        instanceFootprintBytes(_instances[fallbackIndex]) +
        kTLASNodeFootprintBytes;
    if (footprint > 0 && streamInInstance(fallbackIndex)) {
      preloadedBytes += footprint;
      streamedCount = 1;
      minFootprint = std::min(minFootprint, footprint);
    }
  }

  if (minFootprint == std::numeric_limits<size_t>::max())
    _minInstanceFootprint = 0;
  else
    _minInstanceFootprint = minFootprint;

  _recentVisibleFootprint = totalRequired;

  if (streamedCount > 0) {
    double mb = static_cast<double>(preloadedBytes) / (1024.0 * 1024.0);
    printf("Preloaded %zu/%zu instances (%.2f MB) before first frame.\n",
           streamedCount, candidates.size(), mb);
  }

  if (_residentInstanceCount > 0) {
    _blasNodeCount = _currentBlasNodeCount;
    refreshActiveNodeCount();
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
  if (!_pUniformsBuffer)
    return;

  if (_lightTableDirty)
    rebuildLightTable();

  UniformsData &u = *((UniformsData *)_pUniformsBuffer->contents());

  bool cameraChanged = updateCamera();
  if (cameraChanged || _pendingAccumulationReset) {
    u.frameCount = 0;
    u.randomSeed = {randomFloat(), randomFloat(), randomFloat()};
    _pendingAccumulationReset = false;
  } else {
    u.frameCount++;
  }

  u.primitiveCount = _pScene->getPrimitiveCount();
  u.triangleCount = _pScene->getTriangleCount();
  u.totalPrimitiveCount = _allPrimitives.size();
  u.tlasNodeCount = _tlasNodeCount;
  u.blasNodeCount = _currentBlasNodeCount;
  u.maxRayDepth = _pScene->maxRayDepth;
  u.debugAS = InputSystem::debugAS;
  u.residentInstanceCount = static_cast<uint32_t>(_residentInstanceCount);
  u.totalInstanceCount = static_cast<uint32_t>(_instances.size());
  u.lightCount = static_cast<uint32_t>(_lightTable.size());

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

  pEnc->setFragmentBuffer(_pUniformsBuffer, 0, 0);
  pEnc->setFragmentBuffer(_pTLASBuffer, 0, 1);
  pEnc->setFragmentBuffer(_pInstanceMetaBuffer, 0, 2);
  pEnc->setFragmentBuffer(_pInstanceArgBuffer, 0, 3);
  pEnc->setFragmentBuffer(_pTLASInstanceIndexBuffer, 0, 4);
  pEnc->setFragmentBuffer(_pLightBuffer, 0, 5);

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
  const float FULL_DETAIL_DISTANCE = 250.0f;

  if (_instances.empty()) {
    _minInstanceFootprint = 0;
    return;
  }

  struct Candidate {
    size_t index;
    float distance;
    bool withinDistance;
  };

  std::vector<Candidate> candidates;
  candidates.reserve(_instances.size());

  for (size_t i = 0; i < _instances.size(); ++i) {
    const auto &bounds = _instances[i].bounds;
    float dist = simd::length(bounds.center - Camera::position) - bounds.radius;
    dist = std::max(dist, 0.0f);
    bool within = dist < FULL_DETAIL_DISTANCE;
    if (!within && _instances[i].state == ResidencyState::Resident) {
      // Keep instances that are already resident even if they fall outside of
      // the full-detail distance. Without this, meshes that were streamed in
      // during preload immediately become candidates for eviction on the next
      // frame and disappear even when there is enough budget to keep them
      // resident.
      within = true;
    }
    candidates.push_back({i, dist, within});
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) {
              if (a.distance == b.distance)
                return a.index < b.index;
              return a.distance < b.distance;
            });

  std::vector<bool> keep(_instances.size(), false);
  bool hasBudget = _gpuMemoryBudget != std::numeric_limits<size_t>::max();
  size_t closestIndex =
      candidates.empty() ? std::numeric_limits<size_t>::max()
                           : candidates.front().index;
  size_t minFootprint = std::numeric_limits<size_t>::max();
  size_t usedBudget = 0;
  size_t requiredBudget = 0;
  size_t visibleCount = 0;

  for (const Candidate &candidate : candidates) {
    size_t footprint =
        instanceFootprintBytes(_instances[candidate.index]) +
        kTLASNodeFootprintBytes;
    if (footprint > 0)
      minFootprint = std::min(minFootprint, footprint);
    if (!candidate.withinDistance)
      continue;

    requiredBudget += footprint;
    visibleCount++;
    if (_instances[candidate.index].state == ResidencyState::Resident) {
      keep[candidate.index] = true;
      usedBudget += footprint;
    }
  }

  _recentVisibleFootprint = requiredBudget;

  if (minFootprint == std::numeric_limits<size_t>::max())
    _minInstanceFootprint = 0;
  else
    _minInstanceFootprint = minFootprint;

  if (hasBudget && usedBudget > _gpuMemoryBudget) {
    for (auto it = candidates.rbegin();
         it != candidates.rend() && usedBudget > _gpuMemoryBudget; ++it) {
      size_t index = it->index;
      if (!keep[index])
        continue;
      size_t footprint =
          instanceFootprintBytes(_instances[index]) + kTLASNodeFootprintBytes;
      if (footprint == 0)
        continue;
      keep[index] = false;
      if (usedBudget >= footprint)
        usedBudget -= footprint;
    }
  }

  if (!_manualBudget && hasBudget && requiredBudget > 0 &&
      requiredBudget > _gpuMemoryBudget) {
    size_t previousBudget = _gpuMemoryBudget;
    _gpuMemoryBudget = requiredBudget;
    _recommendedBudget = std::max(_recommendedBudget, _gpuMemoryBudget);
    double prevMB = static_cast<double>(previousBudget) / (1024.0 * 1024.0);
    double newMB = static_cast<double>(_gpuMemoryBudget) / (1024.0 * 1024.0);
    printf("Expanded GPU memory budget from %.2f MB to %.2f MB to keep %zu instances resident.\n",
           prevMB, newMB, visibleCount);
    hasBudget = _gpuMemoryBudget != std::numeric_limits<size_t>::max();
  }

  for (const Candidate &candidate : candidates) {
    if (!candidate.withinDistance)
      continue;
    if (keep[candidate.index])
      continue;

    size_t footprint =
        instanceFootprintBytes(_instances[candidate.index]) +
        kTLASNodeFootprintBytes;
    if (footprint == 0)
      continue;

    if (hasBudget) {
      if (footprint > _gpuMemoryBudget)
        continue;
      if (usedBudget + footprint > _gpuMemoryBudget)
        continue;
    }

    keep[candidate.index] = true;
    usedBudget += footprint;
  }

  if (!candidates.empty() &&
      std::none_of(keep.begin(), keep.end(), [](bool value) { return value; })) {
    keep[closestIndex] = true;
  }

  bool residencyChanged = false;

  for (size_t i = 0; i < _instances.size(); ++i) {
    if (!keep[i] && _instances[i].state == ResidencyState::Resident) {
      streamOutInstance(i);
      residencyChanged = true;
    }
  }

  for (size_t i = 0; i < _instances.size(); ++i) {
    if (keep[i]) {
      if (streamInInstance(i))
        residencyChanged = true;
    }
  }

  if (_residentInstanceCount == 0 && !_instances.empty()) {
    size_t fallbackIndex = closestIndex == std::numeric_limits<size_t>::max()
                               ? 0
                               : closestIndex;
    if (streamInInstance(fallbackIndex))
      residencyChanged = true;
  }

  if (residencyChanged) {
    _blasNodeCount = _currentBlasNodeCount;
    refreshActiveNodeCount();
  }
}

void Renderer::initializeInstances() {
  for (size_t i = 0; i < _instances.size(); ++i)
    releaseInstanceResources(i);
  _instances.clear();
  _instancesPendingRelease.clear();
  _residentInstanceCount = 0;
  _currentBlasNodeCount = 0;
  _tlasNodeCount = 0;
  _activeNodeCount = 0;
  _blasNodeCount = 0;
  _cpuBlasNodeCount = 0;
  _totalNodeCount = 0;
  _minInstanceFootprint = 0;
  _recentVisibleFootprint = 0;
  _lastOffloadedNodeCount = 0;
  _lastOffloadedInstanceCount = 0;
  markLightTableDirty();

  size_t primitiveCount = _allPrimitives.size();
  if (primitiveCount > kMaxInstanceCapacity) {
    printf(
        "Warning: Scene has %zu primitives but only %zu instances are supported; "
        "excess primitives will be ignored.\n",
        primitiveCount, kMaxInstanceCapacity);
    primitiveCount = kMaxInstanceCapacity;
  }

  _instances.reserve(primitiveCount);

  auto finalizeInstance = [](InstanceRecord &inst) {
    if (inst.cpuPrimitiveCount == 0)
      return;

    inst.bounds.center = (inst.aabbMin + inst.aabbMax) * 0.5f;
    inst.bounds.radius =
        simd::length(inst.aabbMax - inst.bounds.center);

    struct BuildPrim {
      int index;
      simd::float3 min;
      simd::float3 max;
      simd::float3 centroid;
    };

    struct BlasNodeCPU {
      simd::float3 min;
      simd::float3 max;
      int leftFirst = 0;
      int second = 0;
    };

    std::vector<BuildPrim> refs;
    refs.reserve(inst.cpuPrimitiveCount);
    for (uint32_t i = 0; i < inst.cpuPrimitiveCount; ++i) {
      BuildPrim ref;
      ref.index = inst.primitiveIndices[i];
      ref.min = inst.primitiveBoundsMin[i];
      ref.max = inst.primitiveBoundsMax[i];
      ref.centroid = (ref.min + ref.max) * 0.5f;
      refs.push_back(ref);
    }

    std::vector<BlasNodeCPU> nodes;
    nodes.reserve(inst.cpuPrimitiveCount * 2);
    std::vector<int> ordered;
    ordered.reserve(inst.cpuPrimitiveCount);

    auto buildNode = [&](auto &&self, size_t begin, size_t end) -> int {
      size_t count = end - begin;
      if (count == 0)
        return -1;

      simd::float3 bmin = refs[begin].min;
      simd::float3 bmax = refs[begin].max;
      simd::float3 cmin = refs[begin].centroid;
      simd::float3 cmax = refs[begin].centroid;
      for (size_t i = begin + 1; i < end; ++i) {
        bmin = simd::min(bmin, refs[i].min);
        bmax = simd::max(bmax, refs[i].max);
        cmin = simd::min(cmin, refs[i].centroid);
        cmax = simd::max(cmax, refs[i].centroid);
      }

      int nodeIndex = static_cast<int>(nodes.size());
      nodes.push_back({});
      nodes[nodeIndex].min = bmin;
      nodes[nodeIndex].max = bmax;

      constexpr size_t kLeafSize = 4;
      if (count <= kLeafSize) {
        int start = static_cast<int>(ordered.size());
        for (size_t i = begin; i < end; ++i)
          ordered.push_back(refs[i].index);
        nodes[nodeIndex].leftFirst = start;
        nodes[nodeIndex].second = static_cast<int>(count);
      } else {
        simd::float3 extent = cmax - cmin;
        int axis = 0;
        if (extent.y > extent.x && extent.y >= extent.z)
          axis = 1;
        else if (extent.z > extent.x && extent.z > extent.y)
          axis = 2;

        float split = (cmin[axis] + cmax[axis]) * 0.5f;
        auto midIter = std::partition(refs.begin() + begin, refs.begin() + end,
                                      [axis, split](const BuildPrim &ref) {
                                        return ref.centroid[axis] < split;
                                      });
        size_t mid = static_cast<size_t>(midIter - refs.begin());
        if (mid == begin || mid == end)
          mid = begin + count / 2;

        int left = self(self, begin, mid);
        int right = self(self, mid, end);
        nodes[nodeIndex].leftFirst = left;
        nodes[nodeIndex].second = -right;
      }

      return nodeIndex;
    };

    if (!refs.empty())
      buildNode(buildNode, 0, refs.size());

    inst.cpuBlasNodeCount = static_cast<uint32_t>(nodes.size());
    inst.blasNodes.resize(nodes.size() * 2);
    for (size_t i = 0; i < nodes.size(); ++i) {
      float leftBits = 0.0f;
      float secondBits = 0.0f;
      std::memcpy(&leftBits, &nodes[i].leftFirst, sizeof(float));
      std::memcpy(&secondBits, &nodes[i].second, sizeof(float));
      inst.blasNodes[2 * i] = simd::make_float4(nodes[i].min, leftBits);
      inst.blasNodes[2 * i + 1] = simd::make_float4(nodes[i].max, secondBits);
    }

    inst.primitiveIndices = std::move(ordered);

    inst.state = ResidencyState::NotResident;
    inst.gpu = InstanceGPU{};
  };

  auto appendPrimitive = [&](InstanceRecord &inst, const Primitive &p) {
    simd::float3 pMin(0.0f);
    simd::float3 pMax(0.0f);

    switch (p.type) {
    case PrimitiveType::Sphere: {
      const float radius = p.sphere.radius;
      const float radiusSq = radius * radius;
      simd::float3 center = p.sphere.center;
      simd::float3 radiusVec = simd::make_float3(radius, radius, radius);
      pMin = center - radiusVec;
      pMax = center + radiusVec;
      inst.primitiveData.push_back(
          simd::make_float4(center, static_cast<float>(p.type)));
      inst.primitiveData.push_back(
          simd::make_float4(radius, radiusSq, 0.0f, 0.0f));
      inst.primitiveData.push_back(simd::make_float4(simd::float3(0.0f), 0.0f));
      inst.primitiveData.push_back(simd::make_float4(simd::float3(0.0f), 0.0f));
      break;
    }
    case PrimitiveType::Triangle: {
      const auto &tri = p.triangle;
      pMin = simd::min(tri.v0, simd::min(tri.v1, tri.v2));
      pMax = simd::max(tri.v0, simd::max(tri.v1, tri.v2));
      simd::float3 edge1 = tri.v1 - tri.v0;
      simd::float3 edge2 = tri.v2 - tri.v0;
      simd::float3 normal = simd::cross(edge1, edge2);
      float lenSq = simd::dot(normal, normal);
      if (lenSq > 1e-12f)
        normal *= 1.0f / std::sqrt(lenSq);
      else
        normal = simd::make_float3(0.0f, 0.0f, 0.0f);
      inst.primitiveData.push_back(
          simd::make_float4(tri.v0, static_cast<float>(p.type)));
      inst.primitiveData.push_back(simd::make_float4(edge1, 0.0f));
      inst.primitiveData.push_back(simd::make_float4(edge2, 0.0f));
      inst.primitiveData.push_back(simd::make_float4(normal, 0.0f));
      break;
    }
    case PrimitiveType::Rectangle: {
      const auto &rect = p.rectangle;
      simd::float3 corners[4] = {rect.center + rect.u + rect.v,
                                 rect.center + rect.u - rect.v,
                                 rect.center - rect.u + rect.v,
                                 rect.center - rect.u - rect.v};
      pMin = corners[0];
      pMax = corners[0];
      for (int k = 1; k < 4; ++k) {
        pMin = simd::min(pMin, corners[k]);
        pMax = simd::max(pMax, corners[k]);
      }
      simd::float3 normal = simd::cross(rect.u, rect.v);
      float lenSq = simd::dot(normal, normal);
      if (lenSq > 1e-12f)
        normal *= 1.0f / std::sqrt(lenSq);
      else
        normal = simd::make_float3(0.0f, 0.0f, 0.0f);
      float invDotU = 0.0f;
      float invDotV = 0.0f;
      float dotU = simd::dot(rect.u, rect.u);
      float dotV = simd::dot(rect.v, rect.v);
      if (dotU > 1e-12f)
        invDotU = 1.0f / dotU;
      if (dotV > 1e-12f)
        invDotV = 1.0f / dotV;
      inst.primitiveData.push_back(
          simd::make_float4(rect.center, static_cast<float>(p.type)));
      inst.primitiveData.push_back(simd::make_float4(rect.u, invDotU));
      inst.primitiveData.push_back(simd::make_float4(rect.v, invDotV));
      inst.primitiveData.push_back(simd::make_float4(normal, 0.0f));
      break;
    }
    }

    if (inst.cpuPrimitiveCount == 0) {
      inst.aabbMin = pMin;
      inst.aabbMax = pMax;
    } else {
      inst.aabbMin = simd::min(inst.aabbMin, pMin);
      inst.aabbMax = simd::max(inst.aabbMax, pMax);
    }

    inst.primitiveBoundsMin.push_back(pMin);
    inst.primitiveBoundsMax.push_back(pMax);
    inst.materialData.push_back(
        simd::make_float4(p.material.albedo, p.material.materialType));
    inst.materialData.push_back(
        simd::make_float4(p.material.emissionColor,
                           p.material.emissionPower));
    inst.primitiveIndices.push_back(static_cast<int>(inst.cpuPrimitiveCount));
    inst.cpuPrimitiveCount++;
  };

  std::unordered_map<uint32_t, std::vector<size_t>> meshPrimitiveMap;
  meshPrimitiveMap.reserve(primitiveCount);

  size_t fallbackTriangleCount = 0;
  size_t nonTriangleCount = 0;

  for (size_t i = 0; i < primitiveCount; ++i) {
    const Primitive &p = _allPrimitives[i];
    if (p.type == PrimitiveType::Triangle) {
      if (p.meshId != kInvalidMeshId) {
        auto &indices = meshPrimitiveMap[p.meshId];
        indices.push_back(i);
      } else {
        fallbackTriangleCount++;
      }
    } else {
      nonTriangleCount++;
    }
  }

  size_t potentialInstanceCount = meshPrimitiveMap.size() + fallbackTriangleCount +
                                  nonTriangleCount;
  if (potentialInstanceCount > kMaxInstanceCapacity) {
    printf(
        "Warning: Scene has %zu potential instances but only %zu are supported; "
        "excess instances will be ignored.\n",
        potentialInstanceCount, kMaxInstanceCapacity);
  }

  _instances.reserve(std::min(potentialInstanceCount, kMaxInstanceCapacity));

  auto buildInstanceFromIndices = [&](const std::vector<size_t> &indices,
                                      uint32_t meshId) {
    if (indices.empty() || _instances.size() >= kMaxInstanceCapacity)
      return;
    InstanceRecord inst;
    inst.meshId = meshId;
    inst.primitiveIndex = indices.front();
    for (size_t idx : indices) {
      appendPrimitive(inst, _allPrimitives[idx]);
    }
    finalizeInstance(inst);
    _instances.push_back(std::move(inst));
  };

  std::unordered_set<uint32_t> processedMeshes;
  processedMeshes.reserve(meshPrimitiveMap.size());

  for (size_t i = 0; i < primitiveCount; ++i) {
    const Primitive &p = _allPrimitives[i];
    if (_instances.size() >= kMaxInstanceCapacity)
      break;

    if (p.type == PrimitiveType::Triangle) {
      if (p.meshId != kInvalidMeshId) {
        if (processedMeshes.insert(p.meshId).second) {
          auto it = meshPrimitiveMap.find(p.meshId);
          if (it != meshPrimitiveMap.end())
            buildInstanceFromIndices(it->second, p.meshId);
        }
      } else {
        InstanceRecord inst;
        inst.meshId = kInvalidMeshId;
        inst.primitiveIndex = i;
        appendPrimitive(inst, p);
        finalizeInstance(inst);
        _instances.push_back(std::move(inst));
      }
    } else {
      InstanceRecord inst;
      inst.meshId = kInvalidMeshId;
      inst.primitiveIndex = i;
      appendPrimitive(inst, p);
      finalizeInstance(inst);
      _instances.push_back(std::move(inst));
    }
  }

  size_t totalBlas = 0;
  for (const auto &inst : _instances)
    totalBlas += inst.cpuBlasNodeCount;
  _cpuBlasNodeCount = totalBlas;
  _totalNodeCount = _tlasNodeCount + _cpuBlasNodeCount;
  _pendingAccumulationReset = true;
}

size_t Renderer::instanceFootprintBytes(const InstanceRecord &inst) const {
  size_t bytes = inst.primitiveData.size() * sizeof(simd::float4);
  bytes += inst.materialData.size() * sizeof(simd::float4);
  bytes += inst.primitiveIndices.size() * sizeof(int);
  bytes += inst.blasNodes.size() * sizeof(simd::float4);
  return bytes;
}

bool Renderer::streamInInstance(size_t index) {
  if (index >= _instances.size())
    return false;

  InstanceRecord &inst = _instances[index];
  if (inst.state == ResidencyState::Resident)
    return false;

  if (inst.state == ResidencyState::StreamingOut) {
    auto it = std::find(_instancesPendingRelease.begin(),
                        _instancesPendingRelease.end(), index);
    if (it != _instancesPendingRelease.end())
      _instancesPendingRelease.erase(it);
    inst.state = ResidencyState::Resident;
    inst.gpu.primitiveCount = inst.cpuPrimitiveCount;
    inst.gpu.blasNodeCount = inst.cpuBlasNodeCount;
    inst.gpu.rootNodeIndex = 0;
    _residentInstanceCount++;
    _currentBlasNodeCount += inst.gpu.blasNodeCount;
    updateInstanceArgument(index);
    updateInstanceMetadata(index);
    _pendingAccumulationReset = true;
    markLightTableDirty();
    return true;
  }

  size_t gpuFootprint = instanceFootprintBytes(inst);
  if (gpuFootprint > 0 && !ensureBudget(gpuFootprint)) {
    inst.state = ResidencyState::NotResident;
    return false;
  }

  inst.state = ResidencyState::StreamingIn;

  std::vector<MTL::Buffer *> stagingBuffers;
  stagingBuffers.reserve(4);

  auto cleanupStaging = [&]() {
    for (MTL::Buffer *buffer : stagingBuffers)
      buffer->release();
    stagingBuffers.clear();
  };

  MTL::CommandBuffer *cmd = _pCommandQueue->commandBuffer();
  if (!cmd) {
    inst.state = ResidencyState::NotResident;
    return false;
  }

  MTL::BlitCommandEncoder *blit = cmd->blitCommandEncoder();
  if (!blit) {
    inst.state = ResidencyState::NotResident;
    return false;
  }

  auto enqueueCopy = [&](const void *src, size_t size, MTL::Buffer **dst) {
    if (*dst) {
      (*dst)->release();
      *dst = nullptr;
    }
    if (size == 0)
      return true;
    MTL::Buffer *staging =
        _pDevice->newBuffer(size, MTL::ResourceStorageModeShared);
    std::memcpy(staging->contents(), src, size);
    MTL::Buffer *gpu =
        _pDevice->newBuffer(size, MTL::ResourceStorageModePrivate);
    blit->copyFromBuffer(staging, 0, gpu, 0, size);
    stagingBuffers.push_back(staging);
    *dst = gpu;
    return true;
  };

  bool success = enqueueCopy(inst.primitiveData.data(),
                             inst.primitiveData.size() * sizeof(simd::float4),
                             &inst.gpu.primitives);
  success = success &&
            enqueueCopy(inst.materialData.data(),
                        inst.materialData.size() * sizeof(simd::float4),
                        &inst.gpu.materials);
  success = success && enqueueCopy(inst.primitiveIndices.data(),
                                   inst.primitiveIndices.size() * sizeof(int),
                                   &inst.gpu.primitiveIndices);
  success = success &&
            enqueueCopy(inst.blasNodes.data(),
                        inst.blasNodes.size() * sizeof(simd::float4),
                        &inst.gpu.blasNodes);

  blit->endEncoding();

  if (!success) {
    cleanupStaging();
    releaseInstanceResources(index);
    inst.state = ResidencyState::NotResident;
    return false;
  }

  cmd->commit();
  cmd->waitUntilCompleted();

  cleanupStaging();

  inst.gpu.primitiveCount = inst.cpuPrimitiveCount;
  inst.gpu.blasNodeCount = inst.cpuBlasNodeCount;
  inst.gpu.rootNodeIndex = 0;
  inst.state = ResidencyState::Resident;
  _residentInstanceCount++;
  _currentBlasNodeCount += inst.gpu.blasNodeCount;

  updateInstanceArgument(index);
  updateInstanceMetadata(index);
  _pendingAccumulationReset = true;
  markLightTableDirty();
  return true;
}

void Renderer::streamOutInstance(size_t index) {
  if (index >= _instances.size())
    return;

  InstanceRecord &inst = _instances[index];
  if (inst.state != ResidencyState::Resident)
    return;

  inst.state = ResidencyState::StreamingOut;
  if (_residentInstanceCount > 0)
    _residentInstanceCount--;
  if (_currentBlasNodeCount >= inst.gpu.blasNodeCount)
    _currentBlasNodeCount -= inst.gpu.blasNodeCount;
  updateInstanceArgument(index);
  updateInstanceMetadata(index);
  if (std::find(_instancesPendingRelease.begin(), _instancesPendingRelease.end(),
                index) == _instancesPendingRelease.end())
    _instancesPendingRelease.push_back(index);
  _pendingAccumulationReset = true;
  markLightTableDirty();
}

void Renderer::releaseInstanceResources(size_t index) {
  if (index >= _instances.size())
    return;

  InstanceRecord &inst = _instances[index];
  if (inst.gpu.blasNodes) {
    inst.gpu.blasNodes->release();
    inst.gpu.blasNodes = nullptr;
  }
  if (inst.gpu.primitives) {
    inst.gpu.primitives->release();
    inst.gpu.primitives = nullptr;
  }
  if (inst.gpu.materials) {
    inst.gpu.materials->release();
    inst.gpu.materials = nullptr;
  }
  if (inst.gpu.primitiveIndices) {
    inst.gpu.primitiveIndices->release();
    inst.gpu.primitiveIndices = nullptr;
  }
  inst.gpu.primitiveCount = 0;
  inst.gpu.blasNodeCount = 0;
  inst.gpu.rootNodeIndex = 0;
  inst.state = ResidencyState::NotResident;
  markLightTableDirty();
}

void Renderer::updateInstanceArgument(size_t index) {
  if (!_pInstanceArgEncoder || !_pInstanceArgElementEncoder ||
      !_pInstanceArgBuffer || _instanceArgStride == 0)
    return;
  if (index >= kMaxInstanceCapacity)
    return;

  size_t offset = index * _instanceArgStride;
  if (offset + _instanceArgStride > _pInstanceArgEncoder->encodedLength())
    return;

  _pInstanceArgElementEncoder->setArgumentBuffer(_pInstanceArgBuffer,
                                                 NS::UInteger(offset));
  const InstanceRecord &inst = _instances[index];
  if (inst.state == ResidencyState::Resident) {
    _pInstanceArgElementEncoder->setBuffer(inst.gpu.blasNodes, 0,
                                           NS::UInteger(0));
    _pInstanceArgElementEncoder->setBuffer(inst.gpu.primitives, 0,
                                           NS::UInteger(1));
    _pInstanceArgElementEncoder->setBuffer(inst.gpu.materials, 0,
                                           NS::UInteger(2));
    _pInstanceArgElementEncoder->setBuffer(inst.gpu.primitiveIndices, 0,
                                           NS::UInteger(3));
  } else {
    _pInstanceArgElementEncoder->setBuffer(nullptr, 0, NS::UInteger(0));
    _pInstanceArgElementEncoder->setBuffer(nullptr, 0, NS::UInteger(1));
    _pInstanceArgElementEncoder->setBuffer(nullptr, 0, NS::UInteger(2));
    _pInstanceArgElementEncoder->setBuffer(nullptr, 0, NS::UInteger(3));
  }
}

void Renderer::updateInstanceMetadata(size_t index) {
  if (!_pInstanceMetaBuffer)
    return;
  if (index >= _instances.size())
    return;

  InstanceMetadataCPU *meta =
      reinterpret_cast<InstanceMetadataCPU *>(_pInstanceMetaBuffer->contents());
  InstanceMetadataCPU data{};
  const InstanceRecord &inst = _instances[index];
  if (inst.state == ResidencyState::Resident) {
    data.primitiveCount = inst.gpu.primitiveCount;
    data.blasNodeCount = inst.gpu.blasNodeCount;
    data.rootNodeIndex = inst.gpu.rootNodeIndex;
  }
  meta[index] = data;
  _pInstanceMetaBuffer->didModifyRange(
      NS::Range::Make(index * sizeof(InstanceMetadataCPU),
                      sizeof(InstanceMetadataCPU)));
}

void Renderer::markLightTableDirty() {
  _lightTableDirty = true;
  _pendingAccumulationReset = true;
}

static float encodeUInt32(uint32_t value) {
  float bits = 0.0f;
  std::memcpy(&bits, &value, sizeof(uint32_t));
  return bits;
}

void Renderer::rebuildLightTable() {
  if (!_lightTableDirty)
    return;

  if (_pLightBuffer) {
    _pLightBuffer->release();
    _pLightBuffer = nullptr;
  }

  _lightTable.clear();
  std::vector<float> weights;
  weights.reserve(_instances.size());

  constexpr float kPi = 3.14159265358979323846f;

  auto primitiveArea = [&](const InstanceRecord &inst,
                           size_t primIndex) -> float {
    size_t base = primIndex * kPrimitiveStride;
    if (base + 3 >= inst.primitiveData.size())
      return 0.0f;
    const simd::float4 &p0 = inst.primitiveData[base + 0];
    const simd::float4 &p1 = inst.primitiveData[base + 1];
    const simd::float4 &p2 = inst.primitiveData[base + 2];
    int type = static_cast<int>(p0.w);
    switch (type) {
    case 0: {
      float radius = p1.x;
      return 4.0f * kPi * radius * radius;
    }
    case 1: {
      simd::float3 edge1 = p1.xyz;
      simd::float3 edge2 = p2.xyz;
      simd::float3 crossProd = simd::cross(edge1, edge2);
      float len = simd::length(crossProd);
      return 0.5f * len;
    }
    case 2: {
      simd::float3 u = p1.xyz;
      simd::float3 v = p2.xyz;
      simd::float3 crossProd = simd::cross(u, v);
      float len = simd::length(crossProd);
      return 4.0f * len;
    }
    default:
      break;
    }
    return 0.0f;
  };

  double totalWeight = 0.0;

  for (size_t instIndex = 0; instIndex < _instances.size(); ++instIndex) {
    const InstanceRecord &inst = _instances[instIndex];
    if (inst.state != ResidencyState::Resident)
      continue;
    if (inst.cpuPrimitiveCount == 0)
      continue;

    for (uint32_t prim = 0; prim < inst.cpuPrimitiveCount; ++prim) {
      size_t matBase = static_cast<size_t>(prim) * 2;
      if (matBase + 1 >= inst.materialData.size())
        continue;
      const simd::float4 &m1 = inst.materialData[matBase + 1];
      simd::float3 emissionColor = {m1.x, m1.y, m1.z};
      float emissionPower = m1.w;
      if (emissionPower <= 0.0f)
        continue;

      simd::float3 radiance = emissionColor * emissionPower;
      float luminance = 0.2126f * radiance.x + 0.7152f * radiance.y +
                        0.0722f * radiance.z;
      if (luminance <= 0.0f)
        continue;

      float area = primitiveArea(inst, prim);
      if (area <= 0.0f)
        continue;

      GPULightData light{};
      light.meta = simd::make_float4(encodeUInt32(static_cast<uint32_t>(instIndex)),
                                     encodeUInt32(static_cast<uint32_t>(prim)),
                                     area, 0.0f);
      light.emission = simd::make_float4(emissionColor, emissionPower);
      light.cdf = simd::make_float4(0.0f, 0.0f, 0.0f, 0.0f);

      _lightTable.push_back(light);
      float weight = luminance * area;
      weights.push_back(weight);
      totalWeight += static_cast<double>(weight);
    }
  }

  if (_lightTable.empty() || totalWeight <= 0.0) {
    _lightTable.clear();
    _lightTableDirty = false;

    // Ensure the shader always has a valid buffer bound even when no emissive
    // primitives are present. Metal validation requires that all buffer slots
    // used by the shader are populated, so create a small dummy buffer in this
    // case instead of leaving it null.
    GPULightData emptyLight{};
    size_t bufferSize = sizeof(GPULightData);
    if (!_pLightBuffer) {
      _pLightBuffer =
          _pDevice->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    }

    if (_pLightBuffer && _pLightBuffer->length() >= bufferSize) {
      std::memcpy(_pLightBuffer->contents(), &emptyLight, bufferSize);
      _pLightBuffer->didModifyRange(NS::Range::Make(0, bufferSize));
    }

    return;
  }

  float cumulative = 0.0f;
  for (size_t i = 0; i < _lightTable.size(); ++i) {
    float weight = weights[i];
    float pdf = weight > 0.0f ? static_cast<float>(weight / totalWeight) : 0.0f;
    cumulative += pdf;
    _lightTable[i].meta.w = pdf;
    _lightTable[i].cdf.x = cumulative;
  }
  _lightTable.back().cdf.x = 1.0f;

  size_t bufferSize = _lightTable.size() * sizeof(GPULightData);
  if (!ensureBudget(bufferSize)) {
    _lightTable.clear();
    _lightTableDirty = false;
    return;
  }

  _pLightBuffer =
      _pDevice->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
  std::memcpy(_pLightBuffer->contents(), _lightTable.data(), bufferSize);
  _pLightBuffer->didModifyRange(NS::Range::Make(0, bufferSize));

  _lightTableDirty = false;
}

std::pair<size_t, size_t> Renderer::calculateOffloadedResidency() const {
  size_t offloadedNodes = 0;
  size_t offloadedInstances = 0;
  for (const auto &inst : _instances) {
    if (inst.state != ResidencyState::Resident) {
      offloadedNodes += inst.cpuBlasNodeCount;
      offloadedInstances++;
    }
  }
  return {offloadedNodes, offloadedInstances};
}

void Renderer::refreshActiveNodeCount() {
  size_t previousActive = _activeNodeCount;
  size_t previousOffloadedNodes = _lastOffloadedNodeCount;
  size_t previousOffloadedInstances = _lastOffloadedInstanceCount;
  _activeNodeCount = _tlasNodeCount + _currentBlasNodeCount;
  size_t total = _tlasNodeCount + _cpuBlasNodeCount;
  if (_totalNodeCount != total)
    _totalNodeCount = total;
  auto [offloadedNodes, offloadedInstances] = calculateOffloadedResidency();
  _lastOffloadedNodeCount = offloadedNodes;
  _lastOffloadedInstanceCount = offloadedInstances;
  if (_activeNodeCount != previousActive ||
      offloadedNodes != previousOffloadedNodes ||
      offloadedInstances != previousOffloadedInstances) {
    printf("Active nodes: %zu (offloaded: %zu across %zu instances)\n",
           _activeNodeCount, offloadedNodes, offloadedInstances);
  }
}

void Renderer::rebuildTLAS() {
  struct TLASNodeCPU {
    simd::float3 min;
    simd::float3 max;
    int leftFirst = 0;
    int second = 0;
  };

  struct BuildRef {
    size_t instanceIndex;
    simd::float3 min;
    simd::float3 max;
    simd::float3 centroid;
  };

  std::vector<TLASNodeCPU> nodes;
  std::vector<int> instanceIndices;

  if (!_instances.empty()) {
    std::vector<BuildRef> refs;
    refs.reserve(_instances.size());
    for (size_t index = 0; index < _instances.size(); ++index) {
      const InstanceRecord &inst = _instances[index];
      if (inst.cpuPrimitiveCount == 0)
        continue;
      BuildRef ref;
      ref.instanceIndex = index;
      ref.min = inst.aabbMin;
      ref.max = inst.aabbMax;
      ref.centroid = (inst.aabbMin + inst.aabbMax) * 0.5f;
      refs.push_back(ref);
    }

    nodes.reserve(refs.size() * 2);
    instanceIndices.reserve(refs.size());

    auto buildNode = [&](auto &&self, size_t begin, size_t end) -> int {
      size_t count = end - begin;
      if (count == 0)
        return -1;

      simd::float3 bmin = refs[begin].min;
      simd::float3 bmax = refs[begin].max;
      simd::float3 cmin = refs[begin].centroid;
      simd::float3 cmax = refs[begin].centroid;
      for (size_t i = begin + 1; i < end; ++i) {
        bmin = simd::min(bmin, refs[i].min);
        bmax = simd::max(bmax, refs[i].max);
        cmin = simd::min(cmin, refs[i].centroid);
        cmax = simd::max(cmax, refs[i].centroid);
      }

      int nodeIndex = static_cast<int>(nodes.size());
      nodes.push_back({});
      nodes[nodeIndex].min = bmin;
      nodes[nodeIndex].max = bmax;

      if (count <= static_cast<size_t>(kMaxTLASLeafInstances)) {
        int start = static_cast<int>(instanceIndices.size());
        for (size_t i = begin; i < end; ++i) {
          instanceIndices.push_back(static_cast<int>(refs[i].instanceIndex));
        }
        nodes[nodeIndex].leftFirst = start;
        nodes[nodeIndex].second = static_cast<int>(count);
      } else {
        simd::float3 extent = cmax - cmin;
        int axis = 0;
        if (extent.y > extent.x && extent.y >= extent.z)
          axis = 1;
        else if (extent.z > extent.x && extent.z > extent.y)
          axis = 2;

        size_t mid = begin + count / 2;
        auto comparator = [axis](const BuildRef &a, const BuildRef &b) {
          return a.centroid[axis] < b.centroid[axis];
        };
        std::nth_element(refs.begin() + begin, refs.begin() + mid,
                         refs.begin() + end, comparator);

        int leftChild = self(self, begin, mid);
        int rightChild = self(self, mid, end);
        nodes[nodeIndex].leftFirst = leftChild;
        nodes[nodeIndex].second = -rightChild;
      }

      return nodeIndex;
    };

    if (!refs.empty())
      buildNode(buildNode, 0, refs.size());
  }

  std::vector<simd::float4> tlasData;
  tlasData.reserve(nodes.size() * 2);
  for (size_t i = 0; i < nodes.size(); ++i) {
    float leftBits = 0.0f;
    float secondBits = 0.0f;
    std::memcpy(&leftBits, &nodes[i].leftFirst, sizeof(int));
    std::memcpy(&secondBits, &nodes[i].second, sizeof(int));
    tlasData.push_back(simd::make_float4(nodes[i].min, leftBits));
    tlasData.push_back(simd::make_float4(nodes[i].max, secondBits));
  }

  _tlasNodeCount = nodes.size();
  _totalNodeCount = _tlasNodeCount + _cpuBlasNodeCount;

  size_t nodeByteCount =
      !tlasData.empty() ? tlasData.size() * sizeof(simd::float4)
                        : sizeof(simd::float4) * 2;
  size_t indexByteCount =
      !instanceIndices.empty() ? instanceIndices.size() * sizeof(int)
                               : sizeof(int);

  if (_pTLASBuffer) {
    _pTLASBuffer->release();
    _pTLASBuffer = nullptr;
  }
  if (_pTLASInstanceIndexBuffer) {
    _pTLASInstanceIndexBuffer->release();
    _pTLASInstanceIndexBuffer = nullptr;
  }

  if (!ensureBudget(nodeByteCount + indexByteCount)) {
    size_t previousActive = _activeNodeCount;
    _tlasNodeCount = 0;
    _activeNodeCount = 0;
    _totalNodeCount = _cpuBlasNodeCount;
    if (_activeNodeCount != previousActive) {
      size_t offloaded = _totalNodeCount;
      printf("Active nodes: %zu (offloaded: %zu)\n", _activeNodeCount,
             offloaded);
    }
    return;
  }

  _pTLASBuffer =
      _pDevice->newBuffer(nodeByteCount, MTL::ResourceStorageModeManaged);
  simd::float4 *nodeDst =
      reinterpret_cast<simd::float4 *>(_pTLASBuffer->contents());
  if (!tlasData.empty())
    std::memcpy(nodeDst, tlasData.data(), tlasData.size() * sizeof(simd::float4));
  else
    std::memset(nodeDst, 0, nodeByteCount);
  _pTLASBuffer->didModifyRange(NS::Range::Make(0, nodeByteCount));

  _pTLASInstanceIndexBuffer =
      _pDevice->newBuffer(indexByteCount, MTL::ResourceStorageModeManaged);
  int *indexDst = reinterpret_cast<int *>(_pTLASInstanceIndexBuffer->contents());
  if (!instanceIndices.empty())
    std::memcpy(indexDst, instanceIndices.data(),
                instanceIndices.size() * sizeof(int));
  else
    std::memset(indexDst, 0, indexByteCount);
  _pTLASInstanceIndexBuffer->didModifyRange(
      NS::Range::Make(0, indexByteCount));

  refreshActiveNodeCount();
}

void Renderer::processPendingReleases() {
  for (size_t index : _instancesPendingRelease)
    releaseInstanceResources(index);
  _instancesPendingRelease.clear();
}

bool Renderer::createPrivateBuffer(const void *data, size_t size,
                                   MTL::Buffer **outBuffer) {
  if (*outBuffer) {
    (*outBuffer)->release();
    *outBuffer = nullptr;
  }
  if (size == 0)
    return true;
  if (!ensureBudget(size))
    return false;

  MTL::Buffer *staging =
      _pDevice->newBuffer(size, MTL::ResourceStorageModeShared);
  std::memcpy(staging->contents(), data, size);
  MTL::Buffer *gpu =
      _pDevice->newBuffer(size, MTL::ResourceStorageModePrivate);
  MTL::CommandBuffer *cmd = _pCommandQueue->commandBuffer();
  MTL::BlitCommandEncoder *blit = cmd->blitCommandEncoder();
  blit->copyFromBuffer(staging, 0, gpu, 0, size);
  blit->endEncoding();
  cmd->commit();
  cmd->waitUntilCompleted();

  staging->release();
  *outBuffer = gpu;
  return true;
}

bool Renderer::createPrivateBuffer(const std::vector<simd::float4> &data,
                                   MTL::Buffer **outBuffer) {
  const void *ptr = data.empty() ? nullptr : data.data();
  return createPrivateBuffer(ptr, data.size() * sizeof(simd::float4),
                             outBuffer);
}

bool Renderer::createPrivateBuffer(const std::vector<int> &data,
                                   MTL::Buffer **outBuffer) {
  const void *ptr = data.empty() ? nullptr : data.data();
  return createPrivateBuffer(ptr, data.size() * sizeof(int), outBuffer);
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

void Renderer::dumpAccelerationStructure(const std::string &path) {
  std::filesystem::create_directories(
      std::filesystem::path(path).parent_path());
  std::ofstream out(path);
  if (!out.is_open())
    return;

  out << "{\n";

  out << "  \"tlas\": [\n";
  bool wroteAny = false;
  for (size_t i = 0; i < _instances.size(); ++i) {
    const auto &inst = _instances[i];
    if (inst.state != ResidencyState::Resident)
      continue;
    if (wroteAny)
      out << ",\n";
    out << "    {\"instance\":" << i << ",\"min\":[" << inst.aabbMin.x << ","
        << inst.aabbMin.y << "," << inst.aabbMin.z << "],\"max\":["
        << inst.aabbMax.x << "," << inst.aabbMax.y << "," << inst.aabbMax.z
        << "]}";
    wroteAny = true;
  }
  if (wroteAny)
    out << "\n";
  out << "  ],\n";

  out << "  \"instances\": [\n";
  for (size_t i = 0; i < _instances.size(); ++i) {
    const auto &inst = _instances[i];
    const char *stateStr = "notResident";
    switch (inst.state) {
    case ResidencyState::NotResident:
      stateStr = "notResident";
      break;
    case ResidencyState::StreamingIn:
      stateStr = "streamingIn";
      break;
    case ResidencyState::Resident:
      stateStr = "resident";
      break;
    case ResidencyState::StreamingOut:
      stateStr = "streamingOut";
      break;
    }
    out << "    {\"index\":" << i << ",\"state\":\"" << stateStr
        << "\",\"cpuPrimitiveCount\":" << inst.cpuPrimitiveCount
        << ",\"bounds\":[" << inst.bounds.center.x << ","
        << inst.bounds.center.y << "," << inst.bounds.center.z << ","
        << inst.bounds.radius << "],\"resident\":"
        << (inst.state == ResidencyState::Resident ? "true" : "false")
        << "}";
    if (i + 1 < _instances.size())
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

bool Renderer::popCompletedFrameMetrics(FrameMetrics &outMetrics) {
  std::lock_guard<std::mutex> lock(_metricsMutex);
  if (!_hasPendingMetrics)
    return false;
  outMetrics = _pendingFrameMetrics;
  _hasPendingMetrics = false;
  return true;
}

void Renderer::setGPUMemoryBudgetMB(double mb) {
  _manualBudget = true;
  if (mb <= 0.0) {
    _gpuMemoryBudget = std::numeric_limits<size_t>::max();
    _recommendedBudget = _gpuMemoryBudget;
  } else {
    _gpuMemoryBudget = static_cast<size_t>(mb * 1024.0 * 1024.0);
    _recommendedBudget = _gpuMemoryBudget;
  }
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

void Renderer::adjustBudgetForPerformance() {
  if (_manualBudget)
    return;
  if (_gpuMemoryBudget == std::numeric_limits<size_t>::max())
    return;
  if (_minInstanceFootprint == 0 && _recentVisibleFootprint == 0)
    return;
  if (_lastGPUTime <= 0.0)
    return;

  size_t minBudget = std::max(_minInstanceFootprint, size_t(16 * 1024 * 1024));
  if (_recentVisibleFootprint > 0)
    minBudget = std::max(minBudget, _recentVisibleFootprint);
  size_t maxBudget = _recommendedBudget != std::numeric_limits<size_t>::max()
                         ? std::max(_recommendedBudget, minBudget)
                         : std::max(_gpuMemoryBudget, minBudget);

  size_t newBudget = _gpuMemoryBudget;

  if (_lastGPUTime > _targetFrameTime * kBudgetDecreaseThreshold &&
      _gpuMemoryBudget > minBudget) {
    size_t scaled = static_cast<size_t>(
        static_cast<double>(_gpuMemoryBudget) * kBudgetDecreaseFactor);
    if (scaled < minBudget)
      scaled = minBudget;
    if (scaled < newBudget)
      newBudget = scaled;
  } else if (_lastGPUTime < _targetFrameTime * kBudgetIncreaseThreshold &&
             _gpuMemoryBudget < maxBudget) {
    size_t scaled = static_cast<size_t>(
        static_cast<double>(_gpuMemoryBudget) * kBudgetIncreaseFactor);
    if (scaled < _gpuMemoryBudget + _minInstanceFootprint)
      scaled = _gpuMemoryBudget + _minInstanceFootprint;
    if (scaled > maxBudget)
      scaled = maxBudget;
    if (scaled > newBudget)
      newBudget = scaled;
  }

  if (newBudget != _gpuMemoryBudget) {
    _gpuMemoryBudget = newBudget;
    double mb = static_cast<double>(_gpuMemoryBudget) / (1024.0 * 1024.0);
    printf("Dynamic GPU budget adjusted to %.2f MB (GPU %.2f ms)\n", mb,
           _lastGPUTime * 1000.0);
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
  processPendingReleases();
  adjustBudgetForPerformance();
  auto [offloadedNodes, offloadedInstances] = calculateOffloadedResidency();
  double gpuMemoryMB = currentGPUMemoryMB();

  {
    std::lock_guard<std::mutex> lock(_metricsMutex);
    _pendingFrameMetrics.cpuTime = _lastCPUTime;
    _pendingFrameMetrics.gpuTime = _lastGPUTime;
    _pendingFrameMetrics.raysPerSecond = _lastRaysPerSecond;
    _pendingFrameMetrics.activeNodes = _activeNodeCount;
    _pendingFrameMetrics.offloadedNodes = offloadedNodes;
    _pendingFrameMetrics.offloadedInstances = offloadedInstances;
    _pendingFrameMetrics.gpuMemoryMB = gpuMemoryMB;
    _hasPendingMetrics = true;
  }

  printf(
      "Nodes active: %zu offloaded: %zu (instances: %zu) CPU: %.3f ms GPU: %.3f ms Rays/s: %.2f\n",
      _activeNodeCount, offloadedNodes, offloadedInstances,
      _lastCPUTime * 1000.0, _lastGPUTime * 1000.0, _lastRaysPerSecond);
}

double Renderer::lastCPUTime() const { return _lastCPUTime; }
double Renderer::lastGPUTime() const { return _lastGPUTime; }
double Renderer::lastRaysPerSecond() const { return _lastRaysPerSecond; }
size_t Renderer::activeNodeCount() const { return _activeNodeCount; }
size_t Renderer::totalNodeCount() const { return _totalNodeCount; }
size_t Renderer::offloadedNodeCount() const {
  return calculateOffloadedResidency().first;
}

size_t Renderer::offloadedInstanceCount() const {
  return calculateOffloadedResidency().second;
}

