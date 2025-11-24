#ifndef SCENE_H
#define SCENE_H

#include "BVHNode.h"
#include "Primitive.h"
#include <cstdint>
#include <limits>
#include <simd/simd.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace MetalCppPathTracer {

enum class ResidencyStrategy {
  DistanceLOD = 0,
  EnergyImportance = 1,
  RayHitBudget = 2,
  ScreenSpaceFootprint = 3,
  Probabilistic = 4,
  AlwaysResident = 5,
  EnvironmentHit = 6,
  UnifiedScore = 7,
};

struct SceneObject {
  size_t firstPrimitive = 0;
  size_t primitiveCount = 0;
  simd::float3 boundsMin = simd::make_float3(0.0f, 0.0f, 0.0f);
  simd::float3 boundsMax = simd::make_float3(0.0f, 0.0f, 0.0f);
  int meshGroupId = -1;
  int blasRootIndex = -1;
  int cachedBlasRootIndex = -1;
  std::vector<BVHNode> cachedBlasNodes;
  std::vector<size_t> cachedPrimitiveIndices;
};

struct ResidencyParameters {
  float lodEnterDistance = 225.0f;
  float lodExitDistance = 275.0f;
  float lodEnterViewDegrees = 5.0f;
  float lodExitViewDegrees = 12.0f;
  uint32_t stateCooldownFrames = 5;
  size_t lodMaxTogglesPerFrame = 24;

  float energyTargetFraction = 0.9f;
  size_t energyMinActivePrimitives = 16;
  size_t energyMaxTogglesPerFrame = 32;
  float energyVisibilityBoost = 1.75f;
  float energyImportanceSmoothing = 0.85f;

  float unifiedEnergyWeight = 1.0f;
  float unifiedHitWeight = 1.0f;
  float unifiedCoverageWeight = 1.0f;
  float unifiedDistanceWeight = 1.0f;

  float rayHitDecay = 0.85f;
  float rayHitTargetFraction = 0.6f;
  size_t rayHitMinActivePrimitives = 16;
  size_t rayHitMaxTogglesPerFrame = 12;
  uint32_t rayHitRebuildCooldownFrames = 6;

  float probabilityDecay = 0.9f;
  float probabilityThreshold = 0.5f;
  float probabilityTargetFraction = 0.6f;
  size_t probabilityMinActivePrimitives = 16;
  size_t probabilityMaxTogglesPerFrame = 16;
  float probabilityUncertaintyBoost = 0.25f;
  float probabilityEvidenceWindow = 64.0f;
  float probabilityDesiredHysteresis = 0.05f;
  uint32_t probabilityRecentPromotionFrames = 4;
  uint32_t probabilityIdleCooldownFrames = 3;
  float probabilityIdleDecay = 0.98f;
  float probabilityVisibleFloor = 0.0f;

  float screenFootprintTargetFraction = 0.65f;
  float screenFootprintMinPixelCoverage = 32.0f;
  size_t screenFootprintMinActivePrimitives = 16;
  size_t screenFootprintMaxTogglesPerFrame = 10;

  float environmentTargetActiveFraction = 0.0f;
  float environmentEscapeThreshold = 0.4f;
  size_t environmentMinActivePrimitives = 16;
  size_t environmentMaxTogglesPerFrame = 16;
  std::vector<float> environmentDepthWeights;
  std::vector<float> environmentDepthRadii;

  // Allows resident buffers to shrink when most primitives remain inactive.
  bool enableBufferShrink = true;
  float bufferShrinkActiveRatio = 0.3f;

  // Controls whether per-object BLAS caches should be precomputed during BVH
  // construction. Disabling this reduces build time at the cost of on-demand
  // rebuilds when residency streaming requires the data.
  bool buildCachedBlas = true;
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

struct ObserverCamera {
  simd::float3 position = simd::make_float3(0.0f, 0.0f, 0.0f);
  simd::float3 lookAt = simd::make_float3(0.0f, 0.0f, -1.0f);
  float verticalFov = 60.0f;
};

struct Texture {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<float> pixels; // RGBA
};

struct EnvironmentSettings {
  std::string texturePath;
  float brightness = 1.0f;
};

class Scene {
  struct SAHSplitResult {
    int axis = -1;
    size_t leftCount = 0;
    float cost = std::numeric_limits<float>::max();
  };

public:
  Scene();

  void clear();

  size_t addPrimitive(const Primitive &p);
  size_t addObject(const std::vector<Primitive> &prims,
                   int meshGroupId = -1);
  size_t addObjectSilent(const std::vector<Primitive> &prims,
                         int meshGroupId = -1);
  size_t getPrimitiveCount() const;
  size_t getSphereCount() const;
  size_t getTriangleCount() const;
  size_t getRectangleCount() const;

  const std::vector<Primitive> &getPrimitives() const;
  const std::vector<size_t> &getPrimitiveIndices() const;
  const std::vector<SceneObject> &getObjects() const;

  const std::vector<Texture> &getTextures() const;
  const std::vector<std::string> &getTexturePaths() const;
  int registerTexture(const std::string &cacheKey, const std::string &displayPath,
                      Texture texture);

  const EnvironmentSettings &getEnvironment() const;
  const std::string &getEnvironmentTexturePath() const;
  float getEnvironmentBrightness() const;
  void setEnvironmentTexturePath(const std::string &path);
  void setEnvironmentBrightness(float brightness);
  bool hasEnvironmentTexture() const;

  ResidencyStrategy getResidencyStrategy() const;
  void setResidencyStrategy(ResidencyStrategy strategy);

  const ResidencyParameters &getResidencyParameters() const;
  void setResidencyParameters(const ResidencyParameters &params);

  bool getStartCompacted() const;
  void setStartCompacted(bool start);

  double getTextureResidencyMemoryCapMB() const;
  void setTextureResidencyMemoryCapMB(double capMB);

  void buildBVH();
  size_t getBVHNodeCount() const;
  const std::vector<BVHNode> &getBVHNodes() const;

  SAHSplitResult evaluateSAHSplit(const simd::float3 *boundsMin,
                                  const simd::float3 *boundsMax,
                                  const simd::float3 *centroids,
                                  size_t count, float parentArea) const;

  simd::float2 screenSize;
  uint32_t maxRayDepth;

  simd::float4 *createTransformsBuffer() const;
  simd::float4 *createMaterialsBuffer() const;
  simd::float4 *createSphereBuffer();
  simd::float4 *createSphereMaterialsBuffer();
  simd::float4 *createBVHBuffer() const;
  simd::float4 *createBVHBuffer(const std::vector<Primitive> &subset,
                                std::vector<int> &primitiveIndices,
                                size_t &outCount,
                                std::vector<BVHNode> &outNodes) const;
  simd::float4 *createTLASBuffer(size_t &outCount) const;
  simd::float4 *createTLASBuffer(size_t &outCount,
                                 const std::vector<Primitive> &subset,
                                 const std::vector<BVHNode> &blasNodes) const;
  int *createPrimitiveIndexBuffer() const;
  void createTriangleBuffers(std::vector<simd::float3> &outVertices,
                             std::vector<simd::uint3> &outIndices) const;

  std::vector<CameraKeyframe> cameraPath;

  void setObserverCamera(const ObserverCamera &camera);
  bool hasObserverCamera() const;
  const ObserverCamera &getObserverCamera() const;

private:
  struct BVHScratchBuffers {
    std::vector<simd::float3> primitiveMins;
    std::vector<simd::float3> primitiveMaxs;
    std::vector<simd::float3> primitiveCentroids;
    size_t allocationEvents = 0;
    size_t maxScratchSize = 0;

    void resetStatistics();
    void ensureSize(size_t size);
  };

  std::vector<Primitive> primitives;
  std::vector<size_t> primitiveIndices;
  std::vector<BVHNode> bvhNodes;
  std::vector<SceneObject> objects;
  std::vector<size_t> objectIndices;
  std::vector<TLASNode> tlasNodes;
  std::vector<Texture> textures;
  std::vector<std::string> texturePaths;
  std::unordered_map<std::string, int> textureLookup;
  EnvironmentSettings environment;
  ResidencyStrategy residencyStrategy;
  ResidencyParameters residencyParams;
  bool startCompacted;
  double textureResidencyMemoryCapMB;
  bool observerCameraValid;
  ObserverCamera observerCamera;
  mutable std::vector<simd::float3> triangleVerticesCache;
  mutable std::vector<simd::uint3> triangleIndicesCache;
  mutable bool triangleCacheDirty = true;

  size_t addObjectInternal(const Primitive *prims, size_t count,
                          bool logPrimitives, int meshGroupId);
  int buildBVHRecursive(size_t start, size_t end, BVHScratchBuffers &scratch);
  int buildTLASRecursive(size_t start, size_t end);
  float surfaceArea(const simd::float3 &bmin, const simd::float3 &bmax) const;
  float primitiveAxisValue(const Primitive &p, int axis) const;
  simd::float3 primitiveCentroid(const Primitive &p) const;
  float objectAxisValue(size_t objectIndex, int axis) const;
  void primitiveBounds(const Primitive &p, simd::float3 &pMin,
                       simd::float3 &pMax) const;
  void markTriangleCacheDirty();
};

} // namespace MetalCppPathTracer

#endif
