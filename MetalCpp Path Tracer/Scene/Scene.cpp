#include "Scene.h"
#include "SceneLoader.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace MetalCppPathTracer {

Scene::Scene() { clear(); }

void Scene::clear() {
  // Reset any cached mesh data owned by the loader when the scene is cleared.
  SceneLoader::ClearCache();
  primitives.clear();
  bvhNodes.clear();
  primitiveIndices.clear();
  objects.clear();
  objectIndices.clear();
  tlasNodes.clear();
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

  for (size_t i = 0; i < objects.size(); ++i) {
    auto &obj = objects[i];
    if (obj.primitiveCount == 0)
      continue;

    obj.cachedBlasNodes.clear();
    obj.cachedPrimitiveIndices.clear();
    obj.cachedBlasRootIndex = -1;

    size_t nodeStart = bvhNodes.size();
    int root = buildBVHRecursive(obj.firstPrimitive,
                                 obj.firstPrimitive + obj.primitiveCount);
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

  if (!objectIndices.empty())
    buildTLASRecursive(0, objectIndices.size());
}

size_t Scene::getBVHNodeCount() const { return bvhNodes.size(); }

const std::vector<BVHNode> &Scene::getBVHNodes() const { return bvhNodes; }

simd::float4 *Scene::createTransformsBuffer() const {
  simd::float4 *buffer = new simd::float4[primitives.size() * 3];
  for (size_t i = 0; i < primitives.size(); ++i) {
    const auto &p = primitives[i];
    if (p.type == PrimitiveType::Sphere) {
      buffer[3 * i + 0] =
          simd::make_float4(p.sphere.center, static_cast<float>(p.type));
      buffer[3 * i + 1] =
          simd::make_float4(simd::make_float3(p.sphere.radius, 0, 0), 0);
      buffer[3 * i + 2] = simd::make_float4(simd::float3(0), 0);
    } else if (p.type == PrimitiveType::Rectangle) {
      buffer[3 * i + 0] =
          simd::make_float4(p.rectangle.center, static_cast<float>(p.type));
      buffer[3 * i + 1] = simd::make_float4(p.rectangle.u, 0);
      buffer[3 * i + 2] = simd::make_float4(p.rectangle.v, 0);
    } else {
      buffer[3 * i + 0] =
          simd::make_float4(p.triangle.v0, static_cast<float>(p.type));
      buffer[3 * i + 1] = simd::make_float4(p.triangle.v1, 0);
      buffer[3 * i + 2] = simd::make_float4(p.triangle.v2, 0);
    }
  }
  return buffer;
}

simd::float4 *Scene::createMaterialsBuffer() const {
  simd::float4 *buffer = new simd::float4[2 * primitives.size()];
  for (size_t i = 0; i < primitives.size(); ++i) {
    const auto &m = primitives[i].material;
    buffer[2 * i + 0] = simd::make_float4(m.albedo, m.materialType);
    buffer[2 * i + 1] = simd::make_float4(m.emissionColor, m.emissionPower);
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
  simd::float4 *buffer = new simd::float4[2 * sphereCount];

  size_t index = 0;
  for (const auto &p : primitives) {
    if (p.type == PrimitiveType::Sphere) {
      const auto &m = p.material;
      buffer[2 * index + 0] = simd::make_float4(m.albedo, m.materialType);
      buffer[2 * index + 1] =
          simd::make_float4(m.emissionColor, m.emissionPower);
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

int Scene::buildBVHRecursive(size_t start, size_t end) {
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
  node.count = static_cast<int>(end - start);

  int nodeIndex = static_cast<int>(bvhNodes.size());
  bvhNodes.push_back(node);

  if (node.count <= 8)
    return nodeIndex;

  const int axisCount = 3;
  float bestCost = std::numeric_limits<float>::max();
  int bestAxis = -1;
  size_t bestSplit = start + (end - start) / 2;

  const float parentArea = surfaceArea(bMin, bMax);
  if (parentArea <= 0.0f)
    return nodeIndex;

  for (int axis = 0; axis < axisCount; ++axis) {
    std::sort(primitiveIndices.begin() + start, primitiveIndices.begin() + end,
              [&](size_t a, size_t b) {
                return primitiveAxisValue(primitives[a], axis) <
                       primitiveAxisValue(primitives[b], axis);
              });

    std::vector<simd::float3> leftMin(end - start);
    std::vector<simd::float3> leftMax(end - start);
    std::vector<simd::float3> rightMin(end - start);
    std::vector<simd::float3> rightMax(end - start);

    simd::float3 currMin(std::numeric_limits<float>::max());
    simd::float3 currMax(-std::numeric_limits<float>::max());
    for (size_t i = start; i < end; ++i) {
      const auto &p = primitives[primitiveIndices[i]];
      simd::float3 pMin, pMax;
      primitiveBounds(p, pMin, pMax);
      currMin = simd::min(currMin, pMin);
      currMax = simd::max(currMax, pMax);
      leftMin[i - start] = currMin;
      leftMax[i - start] = currMax;
    }

    currMin = simd::float3(std::numeric_limits<float>::max());
    currMax = simd::float3(-std::numeric_limits<float>::max());
    for (size_t i = end; i-- > start;) {
      const auto &p = primitives[primitiveIndices[i]];
      simd::float3 pMin, pMax;
      primitiveBounds(p, pMin, pMax);
      currMin = simd::min(currMin, pMin);
      currMax = simd::max(currMax, pMax);
      rightMin[i - start] = currMin;
      rightMax[i - start] = currMax;
    }

    for (size_t i = 1; i < (end - start); ++i) {
      float saLeft = surfaceArea(leftMin[i - 1], leftMax[i - 1]);
      float saRight = surfaceArea(rightMin[i], rightMax[i]);

      size_t leftCount = i;
      size_t rightCount = (end - start) - i;

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
            [&](size_t a, size_t b) {
              return primitiveAxisValue(primitives[a], bestAxis) <
                     primitiveAxisValue(primitives[b], bestAxis);
            });

  int leftChild = buildBVHRecursive(start, bestSplit);
  int rightChild = buildBVHRecursive(bestSplit, end);

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

  size_t count = end - start;
  if (count == 1) {
    size_t objectIndex = objectIndices[start];
    tlasNodes[nodeIndex].leftChild =
        -static_cast<int>(objectIndex) - 1; // encode object index
    tlasNodes[nodeIndex].rightChild = static_cast<int>(objectIndex);
    return nodeIndex;
  }

  const float parentArea = surfaceArea(bMin, bMax);
  if (parentArea <= 0.0f) {
    size_t mid = start + count / 2;
    int leftChild = buildTLASRecursive(start, mid);
    int rightChild = buildTLASRecursive(mid, end);
    tlasNodes[nodeIndex].leftChild = leftChild;
    tlasNodes[nodeIndex].rightChild = rightChild;
    return nodeIndex;
  }

  const size_t range = end - start;
  float bestCost = std::numeric_limits<float>::max();
  int bestAxis = -1;
  size_t bestSplit = start + range / 2;

  for (int axis = 0; axis < 3; ++axis) {
    std::sort(objectIndices.begin() + start, objectIndices.begin() + end,
              [&](size_t a, size_t b) {
                return objectAxisValue(a, axis) < objectAxisValue(b, axis);
              });

    std::vector<simd::float3> leftMin(range);
    std::vector<simd::float3> leftMax(range);
    std::vector<simd::float3> rightMin(range);
    std::vector<simd::float3> rightMax(range);

    simd::float3 currMin(std::numeric_limits<float>::max());
    simd::float3 currMax(-std::numeric_limits<float>::max());
    for (size_t i = start; i < end; ++i) {
      const SceneObject &obj = objects[objectIndices[i]];
      currMin = simd::min(currMin, obj.boundsMin);
      currMax = simd::max(currMax, obj.boundsMax);
      leftMin[i - start] = currMin;
      leftMax[i - start] = currMax;
    }

    currMin = simd::float3(std::numeric_limits<float>::max());
    currMax =
        simd::float3(-std::numeric_limits<float>::max());
    for (size_t i = end; i-- > start;) {
      const SceneObject &obj = objects[objectIndices[i]];
      currMin = simd::min(currMin, obj.boundsMin);
      currMax = simd::max(currMax, obj.boundsMax);
      rightMin[i - start] = currMin;
      rightMax[i - start] = currMax;
    }

    for (size_t i = 1; i < range; ++i) {
      float saLeft = surfaceArea(leftMin[i - 1], leftMax[i - 1]);
      float saRight = surfaceArea(rightMin[i], rightMax[i]);

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

  if (bestAxis == -1) {
    size_t mid = start + range / 2;
    int leftChild = buildTLASRecursive(start, mid);
    int rightChild = buildTLASRecursive(mid, end);
    tlasNodes[nodeIndex].leftChild = leftChild;
    tlasNodes[nodeIndex].rightChild = rightChild;
    return nodeIndex;
  }

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
