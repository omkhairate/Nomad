#include "Scene.h"
#include <algorithm>
#include <cstdio>
#include <limits>

namespace MetalCppPathTracer {

Scene::Scene() { clear(); }

void Scene::clear() {
  primitives.clear();
  bvhNodes.clear();
  primitiveIndices.clear();
  cameraPath.clear();
  screenSize = {1280.f, 720.f};
  maxRayDepth = 32;
}

size_t Scene::addPrimitive(const Primitive &p) {
  primitives.push_back(p);
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

void Scene::buildBVH() {
  std::stable_sort(primitives.begin(), primitives.end(),
                   [](const Primitive &a, const Primitive &b) {
                     return static_cast<int>(a.type) < static_cast<int>(b.type);
                   });

  primitiveIndices.resize(primitives.size());
  for (size_t i = 0; i < primitives.size(); ++i)
    primitiveIndices[i] = i;

  bvhNodes.clear();
  buildBVHRecursive(0, primitives.size());

  for (const auto &p : primitives) {
    if (p.type == PrimitiveType::Sphere) {
      const auto &s = p.sphere;
      printf("Sphere -> Position: (%.2f, %.2f, %.2f), Radius: %.2f\n",
             s.center.x, s.center.y, s.center.z, s.radius);
    }
  }
}

size_t Scene::getBVHNodeCount() const { return bvhNodes.size(); }

const std::vector<BVHNode> &Scene::getBVHNodes() const { return bvhNodes; }

simd::float4 *Scene::createTransformsBuffer() {
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

simd::float4 *Scene::createMaterialsBuffer() {
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

simd::float4 *Scene::createBVHBuffer() {
  simd::float4 *buffer = new simd::float4[bvhNodes.size() * 2];
  for (size_t i = 0; i < bvhNodes.size(); ++i) {
    const auto &n = bvhNodes[i];
    buffer[2 * i + 0] = simd::make_float4(n.boundsMin, *(float *)&n.leftFirst);
    buffer[2 * i + 1] = simd::make_float4(n.boundsMax, *(float *)&n.count);
  }
  return buffer;
}

simd::float4 *Scene::createTLASBuffer(size_t &outCount) {
  outCount = 0;
  if (bvhNodes.empty()) {
    return nullptr;
  }

  const BVHNode &root = bvhNodes[0];
  if (root.count > 0) {
    outCount = 1;
    simd::float4 *buffer = new simd::float4[2];
    int rootIndex = 0;
    buffer[0] = simd::make_float4(root.boundsMin, *(float *)&rootIndex);
    buffer[1] = simd::make_float4(root.boundsMax, 0.0f);
    return buffer;
  }

  int leftChild = root.leftFirst;
  int rightChild = -root.count;

  outCount = 2;
  simd::float4 *buffer = new simd::float4[outCount * 2];

  const BVHNode &left = bvhNodes[leftChild];
  buffer[0] = simd::make_float4(left.boundsMin, *(float *)&leftChild);
  buffer[1] = simd::make_float4(left.boundsMax, 0.0f);

  const BVHNode &right = bvhNodes[rightChild];
  buffer[2] = simd::make_float4(right.boundsMin, *(float *)&rightChild);
  buffer[3] = simd::make_float4(right.boundsMax, 0.0f);

  return buffer;
}

int *Scene::createPrimitiveIndexBuffer() {
  int *buffer = new int[primitiveIndices.size()];
  for (size_t i = 0; i < primitiveIndices.size(); ++i) {
    buffer[i] = static_cast<int>(primitiveIndices[i]);
  }
  return buffer;
}

void Scene::createTriangleBuffers(std::vector<simd::float3> &outVertices,
                                  std::vector<simd::uint3> &outIndices) {
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
