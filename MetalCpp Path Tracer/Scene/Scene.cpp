#include "Scene.h"
#include "SceneLoader.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace MetalCppPathTracer {

Scene::Scene() { clear(); }

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
  cameraPath.clear();
  screenSize = {1280.f, 720.f};
  maxRayDepth = 32;
  residencyStrategy = ResidencyStrategy::DistanceLOD;
  residencyParams = ResidencyParameters{};
  startCompacted = false;
  textureResidencyMemoryCapMB = 2048.0;
  observerCameraValid = false;
  observerCamera = ObserverCamera{};
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

  size_t start = primitives.size();
  primitives.insert(primitives.end(), prims, prims + count);

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

int Scene::registerTexture(const std::string &path, Texture texture) {
  auto it = textureLookup.find(path);
  if (it != textureLookup.end())
    return it->second;

  int index = static_cast<int>(textures.size());
  textures.push_back(std::move(texture));
  texturePaths.push_back(path);
  textureLookup.emplace(path, index);
  return index;
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
}

bool Scene::getStartCompacted() const { return startCompacted; }

void Scene::setStartCompacted(bool start) { startCompacted = start; }

double Scene::getTextureResidencyMemoryCapMB() const {
  return textureResidencyMemoryCapMB;
}

void Scene::setTextureResidencyMemoryCapMB(double capMB) {
  textureResidencyMemoryCapMB = capMB;
}

void Scene::setObserverCamera(const ObserverCamera &camera) {
  observerCamera = camera;
  observerCameraValid = true;
}

bool Scene::hasObserverCamera() const { return observerCameraValid; }

const ObserverCamera &Scene::getObserverCamera() const {
  return observerCamera;
}

void Scene::buildBVH() {
  primitiveIndices.resize(primitives.size());
  std::iota(primitiveIndices.begin(), primitiveIndices.end(), 0);

  bvhNodes.clear();
  tlasNodes.clear();
  objectIndices.clear();

  if (primitives.empty() || objects.empty())
    return;

  bvhNodes.reserve(primitives.size() * 2);

#ifdef METALCPPPATHTRACER_LOG_BVH_ALLOCATIONS
  size_t totalScratchAllocations = 0;
  size_t peakScratchSize = 0;
#endif

  for (size_t i = 0; i < objects.size(); ++i) {
    auto &obj = objects[i];
    if (obj.primitiveCount == 0)
      continue;

    obj.cachedBlasNodes.clear();
    obj.cachedPrimitiveIndices.clear();
    obj.cachedBlasRootIndex = -1;

    size_t nodeStart = bvhNodes.size();
    BVHScratchBuffers scratch;
    scratch.resetStatistics();
    int root = buildBVHRecursive(obj.firstPrimitive,
                                 obj.firstPrimitive + obj.primitiveCount,
                                 scratch);
#ifdef METALCPPPATHTRACER_LOG_BVH_ALLOCATIONS
    totalScratchAllocations += scratch.allocationEvents;
    peakScratchSize = std::max(peakScratchSize, scratch.maxScratchSize);
#endif
    size_t nodeEnd = bvhNodes.size();
    obj.blasRootIndex = root;
    if (root >= 0) {
      const BVHNode &rootNode = bvhNodes[root];
      obj.boundsMin = rootNode.boundsMin;
      obj.boundsMax = rootNode.boundsMax;
      objectIndices.push_back(i);

      if (nodeEnd > nodeStart) {
        obj.cachedBlasNodes.reserve(nodeEnd - nodeStart);
        for (size_t nodeIdx = nodeStart; nodeIdx < nodeEnd; ++nodeIdx) {
          BVHNode node = bvhNodes[nodeIdx];
          if (node.count > 0) {
            node.leftFirst -= static_cast<int>(obj.firstPrimitive);
          } else {
            int leftChild = node.leftFirst - static_cast<int>(nodeStart);
            int rightChild = -node.count - static_cast<int>(nodeStart);
            node.leftFirst = leftChild;
            node.count = -rightChild;
          }
          obj.cachedBlasNodes.push_back(node);
        }
        obj.cachedBlasRootIndex = 0;

        obj.cachedPrimitiveIndices.resize(obj.primitiveCount);
        for (size_t local = 0; local < obj.primitiveCount; ++local) {
          size_t globalIndex =
              primitiveIndices[obj.firstPrimitive + local];
          obj.cachedPrimitiveIndices[local] = globalIndex;
        }
      }
    }
  }

#ifdef METALCPPPATHTRACER_LOG_BVH_ALLOCATIONS
  printf("BVH build scratch allocations: %zu events (peak %zu primitives)\n",
         totalScratchAllocations, peakScratchSize);
#endif

  if (!objectIndices.empty())
    buildTLASRecursive(0, objectIndices.size());
}

size_t Scene::getBVHNodeCount() const { return bvhNodes.size(); }

const std::vector<BVHNode> &Scene::getBVHNodes() const { return bvhNodes; }

simd::float4 *Scene::createTransformsBuffer() const {
  simd::float4 *buffer =
      new simd::float4[kPrimitiveFloat4Count * primitives.size()];
  for (size_t i = 0; i < primitives.size(); ++i) {
    const auto &p = primitives[i];
    size_t base = kPrimitiveFloat4Count * i;
    if (p.type == PrimitiveType::Sphere) {
      buffer[base + 0] =
          simd::make_float4(p.sphere.center, static_cast<float>(p.type));
      buffer[base + 1] =
          simd::make_float4(simd::make_float3(p.sphere.radius, 0, 0), 0);
      buffer[base + 2] = simd::make_float4(simd::float3(0), 0);
      buffer[base + 3] = simd::make_float4(simd::float3(0), 0);
    } else if (p.type == PrimitiveType::Rectangle) {
      buffer[base + 0] =
          simd::make_float4(p.rectangle.center, static_cast<float>(p.type));
      buffer[base + 1] = simd::make_float4(p.rectangle.u, 0);
      buffer[base + 2] = simd::make_float4(p.rectangle.v, 0);
      buffer[base + 3] = simd::make_float4(simd::float3(0), 0);
    } else {
      buffer[base + 0] =
          simd::make_float4(p.triangle.v0, static_cast<float>(p.type));
      buffer[base + 1] = simd::make_float4(p.triangle.v1, p.triangle.uv0.x);
      buffer[base + 2] = simd::make_float4(p.triangle.v2, p.triangle.uv0.y);
      buffer[base + 3] = simd::make_float4(p.triangle.uv1.x, p.triangle.uv1.y,
                                           p.triangle.uv2.x, p.triangle.uv2.y);
    }
  }
  return buffer;
}

simd::float4 *Scene::createMaterialsBuffer() const {
  simd::float4 *buffer =
      new simd::float4[kMaterialFloat4Count * primitives.size()];
  for (size_t i = 0; i < primitives.size(); ++i) {
    const auto &m = primitives[i].material;
    auto packed = encodeMaterial(m);
    for (size_t j = 0; j < kMaterialFloat4Count; ++j) {
      buffer[kMaterialFloat4Count * i + j] = packed[j];
    }
  }
  return buffer;
}

simd::float4 *Scene::createSphereBuffer() {
  size_t sphereCount = getSphereCount();
  simd::float4 *buffer = new simd::float4[sphereCount];

  size_t index = 0;
  for (const auto &p : primitives) {
    if (p.type == PrimitiveType::Sphere) {
      buffer[index++] = simd::make_float4(p.sphere.center, p.sphere.radius);
    }
  }

  return buffer;
}

simd::float4 *Scene::createSphereMaterialsBuffer() {
  size_t sphereCount = getSphereCount();
  simd::float4 *buffer = new simd::float4[kMaterialFloat4Count * sphereCount];

  size_t index = 0;
  for (const auto &p : primitives) {
    if (p.type == PrimitiveType::Sphere) {
      auto packed = encodeMaterial(p.material);
      for (size_t j = 0; j < kMaterialFloat4Count; ++j) {
        buffer[kMaterialFloat4Count * index + j] = packed[j];
      }
      index++;
    }
  }

  return buffer;
}

namespace {

void subsetPrimitiveBounds(const Primitive &p, simd::float3 &pMin,
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

float subsetPrimitiveAxisValue(const Primitive &p, int axis) {
  switch (p.type) {
  case PrimitiveType::Sphere:
    return p.sphere.center[axis];
  case PrimitiveType::Rectangle:
    return p.rectangle.center[axis];
  case PrimitiveType::Triangle:
    return (p.triangle.v0[axis] + p.triangle.v1[axis] +
            p.triangle.v2[axis]) /
           3.0f;
  }
  return 0.0f;
}

float subsetSurfaceArea(const simd::float3 &bmin, const simd::float3 &bmax) {
  simd::float3 d = bmax - bmin;
  return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

int buildSubsetBVHRecursive(const std::vector<Primitive> &primitives,
                            std::vector<int> &primitiveIndices,
                            std::vector<BVHNode> &nodes, size_t start,
                            size_t end) {
  BVHNode node;
  simd::float3 bMin(std::numeric_limits<float>::max());
  simd::float3 bMax(-std::numeric_limits<float>::max());

  for (size_t i = start; i < end; ++i) {
    const Primitive &p = primitives[primitiveIndices[i]];
    simd::float3 pMin, pMax;
    subsetPrimitiveBounds(p, pMin, pMax);
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

  const float parentArea = subsetSurfaceArea(bMin, bMax);
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
                return subsetPrimitiveAxisValue(primitives[a], axis) <
                       subsetPrimitiveAxisValue(primitives[b], axis);
              });

    simd::float3 currMin(std::numeric_limits<float>::max());
    simd::float3 currMax(-std::numeric_limits<float>::max());
    for (size_t i = start; i < end; ++i) {
      const Primitive &p = primitives[primitiveIndices[i]];
      simd::float3 pMin, pMax;
      subsetPrimitiveBounds(p, pMin, pMax);
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
      subsetPrimitiveBounds(p, pMin, pMax);
      currMin = simd::min(currMin, pMin);
      currMax = simd::max(currMax, pMax);
      rightMin[i - start] = currMin;
      rightMax[i - start] = currMax;
    }

    for (size_t i = 1; i < range; ++i) {
      float saLeft = subsetSurfaceArea(leftMin[i - 1], leftMax[i - 1]);
      float saRight = subsetSurfaceArea(rightMin[i], rightMax[i]);

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
              return subsetPrimitiveAxisValue(primitives[a], bestAxis) <
                     subsetPrimitiveAxisValue(primitives[b], bestAxis);
            });

  int leftChild = buildSubsetBVHRecursive(primitives, primitiveIndices, nodes,
                                          start, bestSplit);
  int rightChild = buildSubsetBVHRecursive(primitives, primitiveIndices, nodes,
                                           bestSplit, end);

  nodes[nodeIndex].leftFirst = leftChild;
  nodes[nodeIndex].count = -rightChild;
  return nodeIndex;
}

} // namespace

simd::float4 *Scene::createBVHBuffer() const {
  simd::float4 *buffer = new simd::float4[bvhNodes.size() * 2];
  for (size_t i = 0; i < bvhNodes.size(); ++i) {
    const auto &n = bvhNodes[i];
    buffer[2 * i + 0] = simd::make_float4(n.boundsMin, *(float *)&n.leftFirst);
    buffer[2 * i + 1] = simd::make_float4(n.boundsMax, *(float *)&n.count);
  }
  return buffer;
}

simd::float4 *Scene::createBVHBuffer(const std::vector<Primitive> &subset,
                                     std::vector<int> &primitiveIndices,
                                     size_t &outCount,
                                     std::vector<BVHNode> &outNodes) const {
  outCount = 0;
  outNodes.clear();
  if (subset.empty())
    return nullptr;

  if (primitiveIndices.size() != subset.size()) {
    primitiveIndices.resize(subset.size());
    std::iota(primitiveIndices.begin(), primitiveIndices.end(), 0);
  }

  outNodes.reserve(subset.size() * 2);
  buildSubsetBVHRecursive(subset, primitiveIndices, outNodes, 0,
                          primitiveIndices.size());
  outCount = outNodes.size();

  if (outCount == 0)
    return nullptr;

  simd::float4 *buffer = new simd::float4[outCount * 2];
  for (size_t i = 0; i < outNodes.size(); ++i) {
    const BVHNode &node = outNodes[i];
    float leftFirstBits = 0.0f;
    float countBits = 0.0f;
    std::memcpy(&leftFirstBits, &node.leftFirst, sizeof(int));
    std::memcpy(&countBits, &node.count, sizeof(int));
    buffer[2 * i] = simd::make_float4(node.boundsMin, leftFirstBits);
    buffer[2 * i + 1] = simd::make_float4(node.boundsMax, countBits);
  }

  return buffer;
}

simd::float4 *Scene::createTLASBuffer(size_t &outCount) const {
  outCount = 0;
  if (tlasNodes.empty())
    return nullptr;

  outCount = tlasNodes.size();
  simd::float4 *buffer = new simd::float4[outCount * 2];

  for (size_t i = 0; i < tlasNodes.size(); ++i) {
    const TLASNode &node = tlasNodes[i];
    float leftBits = 0.0f;
    float rightBits = 0.0f;
    std::memcpy(&leftBits, &node.leftChild, sizeof(int));
    std::memcpy(&rightBits, &node.rightChild, sizeof(int));
    buffer[2 * i] = simd::make_float4(node.boundsMin, leftBits);
    buffer[2 * i + 1] = simd::make_float4(node.boundsMax, rightBits);
  }

  return buffer;
}

simd::float4 *Scene::createTLASBuffer(size_t &outCount,
                                      const std::vector<Primitive> &subset,
                                      const std::vector<BVHNode> &blasNodes) const {
  outCount = 0;
  if (subset.empty() || blasNodes.empty())
    return nullptr;

  outCount = 1;
  simd::float4 *buffer = new simd::float4[outCount * 2];

  const BVHNode &root = blasNodes.front();
  int encodedLeaf = -1;
  int instanceId = 0;
  float leftBits = 0.0f;
  float rightBits = 0.0f;
  std::memcpy(&leftBits, &encodedLeaf, sizeof(int));
  std::memcpy(&rightBits, &instanceId, sizeof(int));

  buffer[0] = simd::make_float4(root.boundsMin, leftBits);
  buffer[1] = simd::make_float4(root.boundsMax, rightBits);

  return buffer;
}

int *Scene::createPrimitiveIndexBuffer() const {
  int *buffer = new int[primitiveIndices.size()];
  for (size_t i = 0; i < primitiveIndices.size(); ++i) {
    buffer[i] = static_cast<int>(primitiveIndices[i]);
  }
  return buffer;
}

void Scene::createTriangleBuffers(std::vector<simd::float3> &outVertices,
                                  std::vector<simd::uint3> &outIndices) const {
  outVertices.clear();
  outIndices.clear();
  uint32_t baseVertex = 0;

  for (const auto &p : primitives) {
    if (p.type != PrimitiveType::Triangle)
      continue;

    outVertices.push_back(p.triangle.v0);
    outVertices.push_back(p.triangle.v1);
    outVertices.push_back(p.triangle.v2);

    outIndices.push_back(
        simd::make_uint3(baseVertex, baseVertex + 1, baseVertex + 2));
    baseVertex += 3;
  }
}

int Scene::buildBVHRecursive(size_t start, size_t end,
                             BVHScratchBuffers &scratch) {
  BVHNode node;
  simd::float3 bMin(std::numeric_limits<float>::max());
  simd::float3 bMax(-std::numeric_limits<float>::max());

  for (size_t i = start; i < end; ++i) {
    const auto &p = primitives[primitiveIndices[i]];
    simd::float3 pMin, pMax;
    primitiveBounds(p, pMin, pMax);
    bMin = simd::min(bMin, pMin);
    bMax = simd::max(bMax, pMax);
  }

  node.boundsMin = bMin;
  node.boundsMax = bMax;
  node.leftFirst = static_cast<int>(start);
  size_t range = end - start;
  node.count = static_cast<int>(range);

  int nodeIndex = static_cast<int>(bvhNodes.size());
  bvhNodes.push_back(node);

  if (node.count <= 8)
    return nodeIndex;

  const float parentArea = surfaceArea(bMin, bMax);
  if (parentArea <= 0.0f)
    return nodeIndex;

  scratch.ensureSize(range);
  auto &primitiveMins = scratch.primitiveMins;
  auto &primitiveMaxs = scratch.primitiveMaxs;
  auto &primitiveCentroids = scratch.primitiveCentroids;

  for (size_t i = start; i < end; ++i) {
    const auto &p = primitives[primitiveIndices[i]];
    simd::float3 pMin, pMax;
    primitiveBounds(p, pMin, pMax);
    primitiveMins[i - start] = pMin;
    primitiveMaxs[i - start] = pMax;

    simd::float3 centroid = primitiveCentroid(p);
    primitiveCentroids[i - start] = centroid;
  }

  auto sahResult =
      evaluateSAHSplit(primitiveMins.data(), primitiveMaxs.data(),
                       primitiveCentroids.data(), range, parentArea);

  int bestAxis = sahResult.axis;
  size_t bestLeftCount =
      bestAxis == -1 ? range / 2 : std::min(sahResult.leftCount, range - 1);
  size_t bestSplit = start + bestLeftCount;

  if (bestAxis == -1 || bestLeftCount == 0 || bestLeftCount >= range)
    return nodeIndex;

  std::stable_sort(primitiveIndices.begin() + start,
                   primitiveIndices.begin() + end, [&](size_t a, size_t b) {
                     return primitiveAxisValue(primitives[a], bestAxis) <
                            primitiveAxisValue(primitives[b], bestAxis);
                   });

  bestSplit = start + bestLeftCount;

  int leftChild = buildBVHRecursive(start, bestSplit, scratch);
  int rightChild = buildBVHRecursive(bestSplit, end, scratch);

  bvhNodes[nodeIndex].leftFirst = leftChild;
  bvhNodes[nodeIndex].count = -rightChild;

  return nodeIndex;
}

int Scene::buildTLASRecursive(size_t start, size_t end) {
  TLASNode node;
  simd::float3 bMin(std::numeric_limits<float>::max());
  simd::float3 bMax(-std::numeric_limits<float>::max());

  for (size_t i = start; i < end; ++i) {
    const SceneObject &obj = objects[objectIndices[i]];
    bMin = simd::min(bMin, obj.boundsMin);
    bMax = simd::max(bMax, obj.boundsMax);
  }

  node.boundsMin = bMin;
  node.boundsMax = bMax;
  node.leftChild = -1;
  node.rightChild = -1;

  int nodeIndex = static_cast<int>(tlasNodes.size());
  tlasNodes.push_back(node);

  const size_t range = end - start;
  if (range == 1) {
    size_t objectIndex = objectIndices[start];
    tlasNodes[nodeIndex].leftChild =
        -static_cast<int>(objectIndex) - 1; // encode object index
    tlasNodes[nodeIndex].rightChild = static_cast<int>(objectIndex);
    return nodeIndex;
  }

  const float parentArea = surfaceArea(bMin, bMax);
  if (parentArea <= 0.0f) {
    size_t mid = start + range / 2;
    int leftChild = buildTLASRecursive(start, mid);
    int rightChild = buildTLASRecursive(mid, end);
    tlasNodes[nodeIndex].leftChild = leftChild;
    tlasNodes[nodeIndex].rightChild = rightChild;
    return nodeIndex;
  }

  std::vector<simd::float3> objectMins(range);
  std::vector<simd::float3> objectMaxs(range);
  std::vector<simd::float3> objectCentroids(range);
  for (size_t i = 0; i < range; ++i) {
    const SceneObject &obj = objects[objectIndices[start + i]];
    objectMins[i] = obj.boundsMin;
    objectMaxs[i] = obj.boundsMax;
    objectCentroids[i] = 0.5f * (obj.boundsMin + obj.boundsMax);
  }

  auto sahResult = evaluateSAHSplit(objectMins.data(), objectMaxs.data(),
                                    objectCentroids.data(), range, parentArea);
  int bestAxis = sahResult.axis;
  size_t bestLeftCount =
      bestAxis == -1 ? range / 2 : std::min(sahResult.leftCount, range - 1);

  if (bestAxis == -1 || bestLeftCount == 0 || bestLeftCount >= range) {
    size_t mid = start + range / 2;
    int leftChild = buildTLASRecursive(start, mid);
    int rightChild = buildTLASRecursive(mid, end);
    tlasNodes[nodeIndex].leftChild = leftChild;
    tlasNodes[nodeIndex].rightChild = rightChild;
    return nodeIndex;
  }

  size_t bestSplit = start + bestLeftCount;

  std::sort(objectIndices.begin() + start, objectIndices.begin() + end,
            [&](size_t a, size_t b) {
              return objectAxisValue(a, bestAxis) <
                     objectAxisValue(b, bestAxis);
            });

  int leftChild = buildTLASRecursive(start, bestSplit);
  int rightChild = buildTLASRecursive(bestSplit, end);
  tlasNodes[nodeIndex].leftChild = leftChild;
  tlasNodes[nodeIndex].rightChild = rightChild;

  return nodeIndex;
}

Scene::SAHSplitResult Scene::evaluateSAHSplit(
    const simd::float3 *boundsMin, const simd::float3 *boundsMax,
    const simd::float3 *centroids, size_t count, float parentArea) const {
  SAHSplitResult result;
  if (count <= 1 || parentArea <= 0.0f)
    return result;

  constexpr int axisCount = 3;
  constexpr int kBinCount = 12;

  std::array<float, axisCount> centroidMin = {
      std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
      std::numeric_limits<float>::max()};
  std::array<float, axisCount> centroidMax = {
      -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::max()};

  for (size_t i = 0; i < count; ++i) {
    const simd::float3 &centroid = centroids[i];
    centroidMin[0] = std::min(centroidMin[0], centroid.x);
    centroidMin[1] = std::min(centroidMin[1], centroid.y);
    centroidMin[2] = std::min(centroidMin[2], centroid.z);
    centroidMax[0] = std::max(centroidMax[0], centroid.x);
    centroidMax[1] = std::max(centroidMax[1], centroid.y);
    centroidMax[2] = std::max(centroidMax[2], centroid.z);
  }

  std::array<int, kBinCount> binCount{};
  std::array<simd::float3, kBinCount> binMin;
  std::array<simd::float3, kBinCount> binMax;
  std::array<simd::float3, kBinCount> prefixMin;
  std::array<simd::float3, kBinCount> prefixMax;
  std::array<int, kBinCount> prefixCount{};
  std::array<simd::float3, kBinCount> suffixMin;
  std::array<simd::float3, kBinCount> suffixMax;
  std::array<int, kBinCount> suffixCount{};

  for (int axis = 0; axis < axisCount; ++axis) {
    float axisMin = centroidMin[axis];
    float axisMax = centroidMax[axis];
    float axisRange = axisMax - axisMin;
    if (axisRange <= 1e-6f)
      continue;

    for (int i = 0; i < kBinCount; ++i) {
      binCount[i] = 0;
      binMin[i] =
          simd::float3(std::numeric_limits<float>::max());
      binMax[i] =
          simd::float3(-std::numeric_limits<float>::max());
    }

    float invRange = 1.0f / axisRange;
    for (size_t i = 0; i < count; ++i) {
      float centroid = centroids[i][axis];
      int bin = static_cast<int>((centroid - axisMin) * invRange * kBinCount);
      bin = std::max(0, std::min(kBinCount - 1, bin));
      ++binCount[bin];
      binMin[bin] = simd::min(binMin[bin], boundsMin[i]);
      binMax[bin] = simd::max(binMax[bin], boundsMax[i]);
    }

    simd::float3 runningMin(std::numeric_limits<float>::max());
    simd::float3 runningMax(-std::numeric_limits<float>::max());
    int runningCount = 0;
    for (int i = 0; i < kBinCount; ++i) {
      if (binCount[i] > 0) {
        runningMin = simd::min(runningMin, binMin[i]);
        runningMax = simd::max(runningMax, binMax[i]);
      }
      runningCount += binCount[i];
      prefixMin[i] = runningMin;
      prefixMax[i] = runningMax;
      prefixCount[i] = runningCount;
    }

    runningMin = simd::float3(std::numeric_limits<float>::max());
    runningMax = simd::float3(-std::numeric_limits<float>::max());
    runningCount = 0;
    for (int i = kBinCount - 1; i >= 0; --i) {
      if (binCount[i] > 0) {
        runningMin = simd::min(runningMin, binMin[i]);
        runningMax = simd::max(runningMax, binMax[i]);
      }
      runningCount += binCount[i];
      suffixMin[i] = runningMin;
      suffixMax[i] = runningMax;
      suffixCount[i] = runningCount;
    }

    for (int i = 0; i < kBinCount - 1; ++i) {
      int leftCount = prefixCount[i];
      int rightCount = suffixCount[i + 1];
      if (leftCount == 0 || rightCount == 0)
        continue;

      float saLeft = surfaceArea(prefixMin[i], prefixMax[i]);
      float saRight = surfaceArea(suffixMin[i + 1], suffixMax[i + 1]);

      float cost = 0.125f + (saLeft / parentArea) * leftCount +
                   (saRight / parentArea) * rightCount;

      if (cost < result.cost) {
        result.cost = cost;
        result.axis = axis;
        result.leftCount = static_cast<size_t>(leftCount);
      }
    }
  }

  return result;
}

float Scene::surfaceArea(const simd::float3 &bmin, const simd::float3 &bmax) {
  simd::float3 d = bmax - bmin;
  return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

float Scene::primitiveAxisValue(const Primitive &p, int axis) const {
  if (p.type == PrimitiveType::Sphere)
    return p.sphere.center[axis];
  else if (p.type == PrimitiveType::Rectangle)
    return p.rectangle.center[axis];
  else
    return (p.triangle.v0[axis] + p.triangle.v1[axis] + p.triangle.v2[axis]) /
           3.0f;
}

simd::float3 Scene::primitiveCentroid(const Primitive &p) const {
  return simd::make_float3(primitiveAxisValue(p, 0),
                           primitiveAxisValue(p, 1),
                           primitiveAxisValue(p, 2));
}

float Scene::objectAxisValue(size_t objectIndex, int axis) const {
  const SceneObject &obj = objects[objectIndex];
  return 0.5f * (obj.boundsMin[axis] + obj.boundsMax[axis]);
}

void Scene::primitiveBounds(const Primitive &p, simd::float3 &pMin,
                            simd::float3 &pMax) const {
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

} // namespace MetalCppPathTracer
