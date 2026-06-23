#include "Scene.h"
#include "SceneLoader.h"
#include "../Renderer/ParallelFor.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace NomadPathTracer {

void ResidencyParameters::normalizeEnvironmentDepthSettings() {
  size_t pairCount =
      std::min(environmentDepthRadii.size(), environmentDepthWeights.size());
  if (pairCount == 0) {
    environmentDepthRadii.clear();
    environmentDepthWeights.clear();
    return;
  }

  std::vector<std::pair<float, float>> paired;
  paired.reserve(pairCount);
  for (size_t i = 0; i < pairCount; ++i) {
    paired.emplace_back(environmentDepthRadii[i], environmentDepthWeights[i]);
  }

  std::sort(paired.begin(), paired.end(),
            [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

  environmentDepthRadii.resize(pairCount);
  environmentDepthWeights.resize(pairCount);
  for (size_t i = 0; i < pairCount; ++i) {
    environmentDepthRadii[i] = paired[i].first;
    environmentDepthWeights[i] = paired[i].second;
  }
}

Scene::Scene() { clear(); }

void Scene::markTriangleCacheDirty() { triangleCacheDirty = true; }

void Scene::BVHScratchBuffers::resetStatistics() {
  allocationEvents = 0;
  maxScratchSize = 0;
  primitiveMins.clear();
  primitiveMaxs.clear();
  primitiveCentroids.clear();
}

void Scene::BVHScratchBuffers::ensureSize(size_t size) {
  if (size > maxScratchSize)
    maxScratchSize = size;

  if (primitiveMins.capacity() < size) {
    primitiveMins.reserve(size);
    primitiveMaxs.reserve(size);
    primitiveCentroids.reserve(size);
    ++allocationEvents;
  }

  primitiveMins.resize(size);
  primitiveMaxs.resize(size);
  primitiveCentroids.resize(size);
}

void Scene::clear() {
  // Reset any cached mesh data owned by the loader when the scene is cleared.
  SceneLoader::ClearCache();
  primitives.clear();
  bvhNodes.clear();
  primitiveIndices.clear();
  objects.clear();
  objectIndices.clear();
  tlasNodes.clear();
  textures.clear();
  texturePaths.clear();
  textureLookup.clear();
  environment = EnvironmentSettings{};
  environment.brightness = 1.0f;
  cameraPath.clear();
  screenSize = {1280.f, 720.f};
  maxRayDepth = 32;
  residencyStrategy = ResidencyStrategy::DistanceLOD;
  residencyParams = ResidencyParameters{};
  startCompacted = false;
  textureResidencyMemoryCapMB = 2048.0;
  geometryResidencyMemoryCapMB = 2048.0;
  totalGpuMemoryCapMB = 4096.0;
  maxTileSampleWorkPerCommand = kDefaultMaxTileSampleWorkPerCommand;
  maxTileSampleWorkPerCommandSet = false;
  restirEnabled = false;
  observerCameraValid = false;
  observerCamera = ObserverCamera{};
  triangleVerticesCache.clear();
  triangleIndicesCache.clear();
  markTriangleCacheDirty();
}

size_t Scene::addPrimitive(const Primitive &p) {
  return addObjectInternal(&p, 1, true, -1);
}

size_t Scene::addObject(const std::vector<Primitive> &prims,
                        int meshGroupId) {
  if (prims.empty())
    return primitives.size();
  return addObjectInternal(prims.data(), prims.size(), true, meshGroupId);
}

size_t Scene::addObjectSilent(const std::vector<Primitive> &prims,
                              int meshGroupId) {
  if (prims.empty())
    return primitives.size();
  return addObjectInternal(prims.data(), prims.size(), false, meshGroupId);
}

size_t Scene::addObjectInternal(const Primitive *prims, size_t count,
                                bool logPrimitives, int meshGroupId) {
  if (count == 0)
    return primitives.size();

  bool hasTriangles = false;
  for (size_t i = 0; i < count; ++i) {
    if (prims[i].type == PrimitiveType::Triangle) {
      hasTriangles = true;
      break;
    }
  }

  size_t start = primitives.size();
  primitives.insert(primitives.end(), prims, prims + count);

  for (size_t i = 0; i < count; ++i) {
    Primitive &stored = primitives[start + i];
    if (stored.type == PrimitiveType::Triangle) {
      stored.triangle.computeFrame();
    } else if (stored.type == PrimitiveType::Rectangle) {
      stored.rectangle.computeFrame();
    } else if (stored.type == PrimitiveType::Sphere) {
      stored.sphere.supportsNormalMap = 0;
    }
  }

  SceneObject object;
  object.firstPrimitive = start;
  object.primitiveCount = count;
  object.meshGroupId = meshGroupId;
  objects.push_back(object);

  if (logPrimitives) {
    for (size_t i = 0; i < count; ++i) {
      const Primitive &p = primitives[start + i];
      if (p.type == PrimitiveType::Sphere) {
        const auto &s = p.sphere;
        printf("Sphere -> Position: (%.2f, %.2f, %.2f), Radius: %.2f\n",
               s.center.x, s.center.y, s.center.z, s.radius);
      }
    }
  }

  if (hasTriangles)
    markTriangleCacheDirty();

  return primitives.size() - 1;
}

size_t Scene::getPrimitiveCount() const { return primitives.size(); }

size_t Scene::getSphereCount() const {
  size_t count = 0;
  for (const auto &p : primitives)
    if (p.type == PrimitiveType::Sphere)
      count++;
  return count;
}

size_t Scene::getTriangleCount() const {
  size_t count = 0;
  for (const auto &p : primitives)
    if (p.type == PrimitiveType::Triangle)
      count++;
  return count;
}

size_t Scene::getRectangleCount() const {
  size_t count = 0;
  for (const auto &p : primitives)
    if (p.type == PrimitiveType::Rectangle)
      count++;
  return count;
}

const std::vector<Primitive> &Scene::getPrimitives() const {
  return primitives;
}

const std::vector<size_t> &Scene::getPrimitiveIndices() const {
  return primitiveIndices;
}

const std::vector<SceneObject> &Scene::getObjects() const { return objects; }

const std::vector<Texture> &Scene::getTextures() const { return textures; }

const std::vector<std::string> &Scene::getTexturePaths() const {
  return texturePaths;
}

int Scene::registerTexture(const std::string &cacheKey,
                           const std::string &displayPath, Texture texture) {
  auto it = textureLookup.find(cacheKey);
  if (it != textureLookup.end())
    return it->second;

  int index = static_cast<int>(textures.size());
  textures.push_back(std::move(texture));
  texturePaths.push_back(displayPath);
  textureLookup.emplace(cacheKey, index);
  return index;
}

const EnvironmentSettings &Scene::getEnvironment() const { return environment; }

const std::string &Scene::getEnvironmentTexturePath() const {
  return environment.texturePath;
}

float Scene::getEnvironmentBrightness() const { return environment.brightness; }

void Scene::setEnvironmentTexturePath(const std::string &path) {
  environment.texturePath = path;
}

void Scene::setEnvironmentBrightness(float brightness) {
  environment.brightness = std::max(brightness, 0.0f);
}

bool Scene::hasEnvironmentTexture() const {
  return !environment.texturePath.empty();
}

ResidencyStrategy Scene::getResidencyStrategy() const {
  return residencyStrategy;
}

void Scene::setResidencyStrategy(ResidencyStrategy strategy) {
  residencyStrategy = strategy;
}

const ResidencyParameters &Scene::getResidencyParameters() const {
  return residencyParams;
}

void Scene::setResidencyParameters(const ResidencyParameters &params) {
  residencyParams = params;
  residencyParams.normalizeEnvironmentDepthSettings();
}

size_t Scene::getMaxTileSampleWorkPerCommand() const {
  return maxTileSampleWorkPerCommand;
}

bool Scene::hasCustomMaxTileSampleWorkPerCommand() const {
  return maxTileSampleWorkPerCommandSet;
}

void Scene::setMaxTileSampleWorkPerCommand(size_t work) {
  maxTileSampleWorkPerCommand = std::max<size_t>(work, 1);
  maxTileSampleWorkPerCommandSet = true;
}

bool Scene::getStartCompacted() const { return startCompacted; }

void Scene::setStartCompacted(bool start) { startCompacted = start; }

double Scene::getTextureResidencyMemoryCapMB() const {
  return textureResidencyMemoryCapMB;
}

void Scene::setTextureResidencyMemoryCapMB(double capMB) {
  textureResidencyMemoryCapMB = capMB;
}

double Scene::getGeometryResidencyMemoryCapMB() const {
  return geometryResidencyMemoryCapMB;
}

void Scene::setGeometryResidencyMemoryCapMB(double capMB) {
  geometryResidencyMemoryCapMB = capMB;
}

double Scene::getTotalGpuMemoryCapMB() const { return totalGpuMemoryCapMB; }

void Scene::setTotalGpuMemoryCapMB(double capMB) {
  totalGpuMemoryCapMB = capMB;
}

bool Scene::getRestirEnabled() const { return restirEnabled; }

void Scene::setRestirEnabled(bool enabled) { restirEnabled = enabled; }

void Scene::setObserverCamera(const ObserverCamera &camera) {
  observerCamera = camera;
  observerCameraValid = true;
}

bool Scene::hasObserverCamera() const { return observerCameraValid; }

const ObserverCamera &Scene::getObserverCamera() const {
  return observerCamera;
}


} // namespace NomadPathTracer
