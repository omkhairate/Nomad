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
#include <CoreFoundation/CoreFoundation.h>

using namespace MetalCppPathTracer;

static constexpr size_t kMaxInstanceCapacity = 16384;

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
  if (_pInstanceMetaBuffer)
    _pInstanceMetaBuffer->release();
  if (_pInstanceArgBuffer)
    _pInstanceArgBuffer->release();
  if (_pInstanceArgElementEncoder)
    _pInstanceArgElementEncoder->release();
  if (_pInstanceArgEncoder)
    _pInstanceArgEncoder->release();

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
  bool tlasDirty = false;

  for (size_t i = 0; i < _instances.size(); ++i) {
    const auto &bounds = _instances[i].bounds;
    float dist = simd::length(bounds.center - Camera::position) - bounds.radius;
    dist = std::max(dist, 0.0f);
    bool shouldBeResident = dist < FULL_DETAIL_DISTANCE;

    if (shouldBeResident) {
      if (streamInInstance(i))
        tlasDirty = true;
    } else {
      if (_instances[i].state == ResidencyState::Resident) {
        streamOutInstance(i);
        tlasDirty = true;
      }
    }
  }

  if (_residentInstanceCount == 0 && !_instances.empty()) {
    if (streamInInstance(0))
      tlasDirty = true;
  }

  if (tlasDirty)
    rebuildTLAS();
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
  _totalNodeCount = 0;

  size_t primitiveCount = _allPrimitives.size();
  if (primitiveCount > kMaxInstanceCapacity) {
    printf(
        "Warning: Scene has %zu primitives but only %zu instances are supported; "
        "excess primitives will be ignored.\n",
        primitiveCount, kMaxInstanceCapacity);
    primitiveCount = kMaxInstanceCapacity;
  }

  _instances.reserve(primitiveCount);

  for (size_t i = 0; i < primitiveCount; ++i) {
    const Primitive &p = _allPrimitives[i];
    InstanceRecord inst;
    inst.primitiveIndex = i;

    if (p.type == PrimitiveType::Sphere) {
      inst.bounds = {p.sphere.center, p.sphere.radius};
      simd::float3 radius =
          simd::make_float3(p.sphere.radius, p.sphere.radius, p.sphere.radius);
      inst.aabbMin = p.sphere.center - radius;
      inst.aabbMax = p.sphere.center + radius;
    } else if (p.type == PrimitiveType::Triangle) {
      simd::float3 v0 = p.triangle.v0;
      simd::float3 v1 = p.triangle.v1;
      simd::float3 v2 = p.triangle.v2;
      inst.bounds.center = (v0 + v1 + v2) / 3.0f;
      float r = simd::length(v0 - inst.bounds.center);
      r = std::max(r, (float)simd::length(v1 - inst.bounds.center));
      r = std::max(r, (float)simd::length(v2 - inst.bounds.center));
      inst.bounds.radius = r;
      inst.aabbMin = {
          std::min({v0.x, v1.x, v2.x}),
          std::min({v0.y, v1.y, v2.y}),
          std::min({v0.z, v1.z, v2.z}),
      };
      inst.aabbMax = {
          std::max({v0.x, v1.x, v2.x}),
          std::max({v0.y, v1.y, v2.y}),
          std::max({v0.z, v1.z, v2.z}),
      };
    } else {
      simd::float3 c = p.rectangle.center;
      simd::float3 u = p.rectangle.u;
      simd::float3 v = p.rectangle.v;
      inst.bounds.center = c;
      inst.bounds.radius = simd::length(u) + simd::length(v);
      simd::float3 corners[4] = {c + u + v, c + u - v, c - u + v, c - u - v};
      inst.aabbMin = corners[0];
      inst.aabbMax = corners[0];
      for (int k = 1; k < 4; ++k) {
        inst.aabbMin = {
            std::min(inst.aabbMin.x, corners[k].x),
            std::min(inst.aabbMin.y, corners[k].y),
            std::min(inst.aabbMin.z, corners[k].z),
        };
        inst.aabbMax = {
            std::max(inst.aabbMax.x, corners[k].x),
            std::max(inst.aabbMax.y, corners[k].y),
            std::max(inst.aabbMax.z, corners[k].z),
        };
      }
    }

    inst.primitiveData.resize(3);
    if (p.type == PrimitiveType::Sphere) {
      inst.primitiveData[0] =
          simd::make_float4(p.sphere.center, static_cast<float>(p.type));
      inst.primitiveData[1] =
          simd::make_float4(simd::make_float3(p.sphere.radius, 0, 0), 0);
      inst.primitiveData[2] = simd::make_float4(simd::float3(0), 0);
    } else if (p.type == PrimitiveType::Rectangle) {
      inst.primitiveData[0] =
          simd::make_float4(p.rectangle.center, static_cast<float>(p.type));
      inst.primitiveData[1] = simd::make_float4(p.rectangle.u, 0);
      inst.primitiveData[2] = simd::make_float4(p.rectangle.v, 0);
    } else {
      inst.primitiveData[0] =
          simd::make_float4(p.triangle.v0, static_cast<float>(p.type));
      inst.primitiveData[1] = simd::make_float4(p.triangle.v1, 0);
      inst.primitiveData[2] = simd::make_float4(p.triangle.v2, 0);
    }

    inst.materialData.resize(2);
    inst.materialData[0] =
        simd::make_float4(p.material.albedo, p.material.materialType);
    inst.materialData[1] =
        simd::make_float4(p.material.emissionColor, p.material.emissionPower);

    inst.primitiveIndices = {0};
    inst.cpuPrimitiveCount = 1;

    inst.blasNodes.resize(2);
    int leftFirst = 0;
    int count = 1;
    float leftBits = 0.0f;
    float countBits = 0.0f;
    std::memcpy(&leftBits, &leftFirst, sizeof(float));
    std::memcpy(&countBits, &count, sizeof(float));
    inst.blasNodes[0] = simd::make_float4(inst.aabbMin, leftBits);
    inst.blasNodes[1] = simd::make_float4(inst.aabbMax, countBits);
    inst.cpuBlasNodeCount = 1;
    inst.state = ResidencyState::NotResident;
    inst.gpu = InstanceGPU{};

    _instances.push_back(std::move(inst));
  }

  size_t totalBlas = 0;
  for (const auto &inst : _instances)
    totalBlas += inst.cpuBlasNodeCount;
  _totalNodeCount = _instances.size() + totalBlas;
  _pendingAccumulationReset = true;
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
    return true;
  }

  inst.state = ResidencyState::StreamingIn;

  if (!createPrivateBuffer(inst.primitiveData, &inst.gpu.primitives))
    return false;
  if (!createPrivateBuffer(inst.materialData, &inst.gpu.materials)) {
    releaseInstanceResources(index);
    return false;
  }
  if (!createPrivateBuffer(inst.primitiveIndices, &inst.gpu.primitiveIndices)) {
    releaseInstanceResources(index);
    return false;
  }
  if (!createPrivateBuffer(inst.blasNodes, &inst.gpu.blasNodes)) {
    releaseInstanceResources(index);
    return false;
  }

  inst.gpu.primitiveCount = inst.cpuPrimitiveCount;
  inst.gpu.blasNodeCount = inst.cpuBlasNodeCount;
  inst.gpu.rootNodeIndex = 0;
  inst.state = ResidencyState::Resident;
  _residentInstanceCount++;
  _currentBlasNodeCount += inst.gpu.blasNodeCount;

  updateInstanceArgument(index);
  updateInstanceMetadata(index);
  _pendingAccumulationReset = true;
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

void Renderer::rebuildTLAS() {
  std::vector<simd::float4> tlasEntries;
  tlasEntries.reserve(_instances.size() * 2);

  size_t previousActive = _activeNodeCount;
  _currentBlasNodeCount = 0;

  for (size_t i = 0; i < _instances.size(); ++i) {
    const InstanceRecord &inst = _instances[i];
    if (inst.state != ResidencyState::Resident)
      continue;
    float instanceBits = 0.0f;
    int instanceId = static_cast<int>(i);
    std::memcpy(&instanceBits, &instanceId, sizeof(float));
    tlasEntries.push_back(simd::make_float4(inst.aabbMin, instanceBits));
    tlasEntries.push_back(simd::make_float4(inst.aabbMax, 0.0f));
    _currentBlasNodeCount += inst.gpu.blasNodeCount;
  }

  _residentInstanceCount = tlasEntries.size() / 2;
  _blasNodeCount = _currentBlasNodeCount;
  _tlasNodeCount = _residentInstanceCount;
  _activeNodeCount = _tlasNodeCount + _currentBlasNodeCount;

  size_t totalBlas = 0;
  for (const auto &inst : _instances)
    totalBlas += inst.cpuBlasNodeCount;
  _totalNodeCount = _instances.size() + totalBlas;

  size_t requiredCount = tlasEntries.size();
  size_t byteCount =
      requiredCount > 0 ? requiredCount * sizeof(simd::float4)
                        : sizeof(simd::float4) * 2;

  if (_pTLASBuffer) {
    _pTLASBuffer->release();
    _pTLASBuffer = nullptr;
  }
  if (!ensureBudget(byteCount))
    return;
  _pTLASBuffer = _pDevice->newBuffer(byteCount, MTL::ResourceStorageModeManaged);
  simd::float4 *dst =
      reinterpret_cast<simd::float4 *>(_pTLASBuffer->contents());
  if (requiredCount > 0)
    std::memcpy(dst, tlasEntries.data(), byteCount);
  else
    std::memset(dst, 0, byteCount);
  _pTLASBuffer->didModifyRange(NS::Range::Make(0, byteCount));

  if (_activeNodeCount != previousActive)
    printf("Active nodes: %zu\n", _activeNodeCount);
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

  if (!ensureBudget(size)) {
    staging->release();
    return false;
  }
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
  processPendingReleases();
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

