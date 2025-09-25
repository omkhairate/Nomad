#ifndef SCENE_H
#define SCENE_H

#include "BVHNode.h"
#include "Primitive.h"
#include <cstdint>
#include <simd/simd.h>
#include <vector>

namespace MetalCppPathTracer {

enum class ResidencyStrategy {
  DistanceLOD = 0,
  EnergyImportance = 1,
};

struct SceneObject {
  size_t firstPrimitive = 0;
  size_t primitiveCount = 0;
  simd::float3 boundsMin = simd::make_float3(0.0f, 0.0f, 0.0f);
  simd::float3 boundsMax = simd::make_float3(0.0f, 0.0f, 0.0f);
  int blasRootIndex = -1;
};

struct TLASNode {
  simd::float3 boundsMin = simd::make_float3(0.0f, 0.0f, 0.0f);
  simd::float3 boundsMax = simd::make_float3(0.0f, 0.0f, 0.0f);
  int leftChild = -1;
  int rightChild = -1;
};

struct CameraKeyframe {
  uint32_t frame;
  simd::float3 position;
  simd::float3 lookAt;
};

class Scene {
public:
  Scene();

  void clear();

  size_t addPrimitive(const Primitive &p);
  size_t addObject(const std::vector<Primitive> &prims);
  size_t addObjectSilent(const std::vector<Primitive> &prims);
  size_t getPrimitiveCount() const;
  size_t getSphereCount() const;
  size_t getTriangleCount() const;
  size_t getRectangleCount() const;

  const std::vector<Primitive> &getPrimitives() const;
  const std::vector<size_t> &getPrimitiveIndices() const;
  const std::vector<SceneObject> &getObjects() const;

  ResidencyStrategy getResidencyStrategy() const;
  void setResidencyStrategy(ResidencyStrategy strategy);

  void buildBVH();
  size_t getBVHNodeCount() const;
  const std::vector<BVHNode> &getBVHNodes() const;

  simd::float2 screenSize;
  uint32_t maxRayDepth;

  simd::float4 *createTransformsBuffer() const;
  simd::float4 *createMaterialsBuffer() const;
  simd::float4 *createSphereBuffer();
  simd::float4 *createSphereMaterialsBuffer();
  simd::float4 *createBVHBuffer() const;
  simd::float4 *createTLASBuffer(size_t &outCount) const;
  int *createPrimitiveIndexBuffer() const;
  void createTriangleBuffers(std::vector<simd::float3> &outVertices,
                             std::vector<simd::uint3> &outIndices) const;

  std::vector<CameraKeyframe> cameraPath;

private:
  std::vector<Primitive> primitives;
  std::vector<size_t> primitiveIndices;
  std::vector<BVHNode> bvhNodes;
  std::vector<SceneObject> objects;
  std::vector<size_t> objectIndices;
  std::vector<TLASNode> tlasNodes;
  ResidencyStrategy residencyStrategy;

  size_t addObjectInternal(const Primitive *prims, size_t count,
                          bool logPrimitives);
  int buildBVHRecursive(size_t start, size_t end);
  int buildTLASRecursive(size_t start, size_t end);
  float surfaceArea(const simd::float3 &bmin, const simd::float3 &bmax);
  float primitiveAxisValue(const Primitive &p, int axis) const;
  float objectAxisValue(size_t objectIndex, int axis) const;
  void primitiveBounds(const Primitive &p, simd::float3 &pMin,
                       simd::float3 &pMax) const;
};

} // namespace MetalCppPathTracer

#endif
