#ifndef PATH_TRACING_H
#define PATH_TRACING_H

#include <metal_raytracing>
#include <metal_stdlib>

#define M_PI 3.14159265358979323846
#define PRIMITIVE_STRIDE 4

#include "Intersect.h"
#include "Random.h"
#include "Scatter.h"
#include "Structs.h"

using namespace metal;

template <typename T> inline void swap(thread T &a, thread T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}

inline float3 randomUnitVector(thread uint32_t &seed) {
  float z = 2.0 * randomFloat(seed) - 1.0;
  float t = 2.0 * M_PI * randomFloat(seed);
  float r = sqrt(1.0 - z * z);
  return float3(r * cos(t), r * sin(t), z);
}

// Helper: Random point in unit sphere
inline float3 randomInUnitSphere(thread uint32_t &seed) {
  while (true) {
    float3 p =
        2.0 * float3(randomFloat(seed), randomFloat(seed), randomFloat(seed)) -
        float3(1.0);

    if (length_squared(p) < 1.0)
      return p;
  }
}

// BVH intersection helper
inline bool intersectAABB(thread const ray &r, float3 bmin, float3 bmax,
                          float tMin, float tMax) {
  for (int i = 0; i < 3; ++i) {
    float invD = 1.0 / r.direction[i];
    float t0 = (bmin[i] - r.origin[i]) * invD;
    float t1 = (bmax[i] - r.origin[i]) * invD;
    if (invD < 0.0)
      swap(t0, t1);
    tMin = max(tMin, t0);
    tMax = min(tMax, t1);
    // Use a strict comparison to allow zero-width slabs (e.g., thin rectangles)
    if (tMax < tMin)
      return false;
  }
  return true;
}

// BVH traversal version of firstHit
inline intersection firstHitBVH(thread const ray &r,
                                device const float4 *bvhNodes,
                                device const float4 *primitives,
                                device const int *primitiveIndices,
                                uint primitiveCount,
                                int startNode,
                                int instanceId,
                                float tMax = INFINITY) {
  intersection in;
  in.t = tMax;
  in.primitiveId = -1;
  in.isTriangle = 0;
  in.nodeIndex = -1;
  in.instanceId = instanceId;

  bool foundHit = false;

  constexpr int stackSize = 64;
  int stack[stackSize];
  int stackPtr = 0;
  stack[stackPtr++] = startNode;

  while (stackPtr > 0) {
    int nodeIdx = stack[--stackPtr];

    float3 bmin = bvhNodes[2 * nodeIdx + 0].xyz;
    float3 bmax = bvhNodes[2 * nodeIdx + 1].xyz;
    int leftFirst = as_type<int>(bvhNodes[2 * nodeIdx + 0].w);
    int second = as_type<int>(bvhNodes[2 * nodeIdx + 1].w);

    float currentMax = foundHit ? in.t : tMax;
    if (!intersectAABB(r, bmin, bmax, 0.0001, currentMax))
      continue;

    if (second > 0) {
      int count = second;
      for (int i = 0; i < count; ++i) {
        int primIdx = primitiveIndices[leftFirst + i];
        if (primIdx >= int(primitiveCount))
          continue;
        int base = primIdx * PRIMITIVE_STRIDE;
        float4 p0 = primitives[base + 0];
        float4 p1 = primitives[base + 1];
        float4 p2 = primitives[base + 2];
        float4 p3 = primitives[base + 3];

        int primitiveType = int(p0.w);
        float tHit = INFINITY;
        float3 n = float3(0);
        float3 hit = float3(0);
        bool hitThis = false;

        if (primitiveType == 0) {
          float3 center = p0.xyz;
          float radius = p1.x;
          float radiusSq = p1.y > 0.0 ? p1.y : radius * radius;
          float3 oc = r.origin - center;
          float a = dot(r.direction, r.direction);
          float b = dot(oc, r.direction);
          float c = dot(oc, oc) - radiusSq;
          float discriminant = b * b - a * c;

          if (discriminant > 0.0) {
            float sqrtD = sqrt(discriminant);
            float temp = (-b - sqrtD) / a;
            if (temp < in.t && temp > 0.0001) {
              tHit = temp;
              hit = r.origin + tHit * r.direction;
              n = normalize(hit - center);
              hitThis = true;
            }
          }
        } else if (primitiveType == 1) {
          float3 v0 = p0.xyz;
          float3 edge1 = p1.xyz;
          float3 edge2 = p2.xyz;
          float3 storedNormal = p3.xyz;
          float3 h = cross(r.direction, edge2);
          float a = dot(edge1, h);
          if (abs(a) > 1e-5) {
            float f = 1.0 / a;
            float3 s = r.origin - v0;
            float u = f * dot(s, h);
            if (u >= 0.0 && u <= 1.0) {
              float3 q = cross(s, edge1);
              float v = f * dot(r.direction, q);
              if (v >= 0.0 && u + v <= 1.0) {
                float tt = f * dot(edge2, q);
                if (tt > 0.0001 && tt < in.t) {
                  tHit = tt;
                  hit = r.origin + tHit * r.direction;
                  float normalLenSq = dot(storedNormal, storedNormal);
                  if (normalLenSq > 1e-12)
                    n = storedNormal;
                  else
                    n = normalize(cross(edge1, edge2));
                  hitThis = true;
                  in.isTriangle = 1;
                }
              }
            }
          }
        } else if (primitiveType == 2) {
          float3 center = p0.xyz;
          float3 e1 = p1.xyz;
          float3 e2 = p2.xyz;
          float invDotU = p1.w;
          float invDotV = p2.w;
          float3 normal = p3.xyz;
          float normalLenSq = dot(normal, normal);
          if (normalLenSq < 1e-12)
            normal = normalize(cross(e1, e2));
          float denom = dot(normal, r.direction);
          if (fabs(denom) > 1e-5) {
            float tt = dot(center - r.origin, normal) / denom;
            if (tt > 0.0001 && tt < in.t) {
              float3 hitPoint = r.origin + tt * r.direction;
              float3 rel = hitPoint - center;
              float u = invDotU > 0.0 ? dot(rel, e1) * invDotU : 0.0;
              float v = invDotV > 0.0 ? dot(rel, e2) * invDotV : 0.0;
              if (fabs(u) <= 1.0 && fabs(v) <= 1.0) {
                tHit = tt;
                hit = hitPoint;
                n = normal;
                hitThis = true;
              }
            }
          }
        }

        if (hitThis && tHit < in.t && tHit < tMax) {
          in.t = tHit;
          in.primitiveId = primIdx;
          in.normal = n;
          in.point = hit;
          in.isTriangle = primitiveType;
          in.nodeIndex = nodeIdx;
          foundHit = true;
        }
      }
    } else {
      int rightChild = -second;
      stack[stackPtr++] = leftFirst;
      stack[stackPtr++] = rightChild;
    }
  }

  if (!foundHit) {
    in.t = INFINITY;
  }

  if (in.primitiveId != -1) {
    in.frontFace = dot(in.normal, r.direction) < 0.0;
    if (!in.frontFace)
      in.normal = -in.normal;
  }

  return in;
}

inline bool isOccluded(thread const ray &shadowRay, float maxDistance,
                       device const float4 *tlasNodes,
                       device const int *tlasInstanceIndices,
                       uint tlasNodeCount,
                       device InstanceArgumentBuffer &instanceArgs,
                       constant InstanceMetadata *instanceMetadata,
                       uint targetInstance,
                       int targetPrimitive) {
  if (tlasNodeCount == 0)
    return false;

  constexpr int stackSize = 128;
  int stack[stackSize];
  int stackPtr = 0;
  stack[stackPtr++] = 0;

  while (stackPtr > 0) {
    int nodeIdx = stack[--stackPtr];
    float3 bmin = tlasNodes[2 * nodeIdx + 0].xyz;
    float3 bmax = tlasNodes[2 * nodeIdx + 1].xyz;
    int leftFirst = as_type<int>(tlasNodes[2 * nodeIdx + 0].w);
    int second = as_type<int>(tlasNodes[2 * nodeIdx + 1].w);

    if (!intersectAABB(shadowRay, bmin, bmax, 0.0001, maxDistance))
      continue;

    if (second > 0) {
      for (int i = 0; i < second; ++i) {
        int instanceId = tlasInstanceIndices[leftFirst + i];
        if (instanceId < 0 || instanceId >= MAX_INSTANCE_COUNT)
          continue;
        const constant InstanceMetadata &meta = instanceMetadata[instanceId];
        if (meta.primitiveCount == 0)
          continue;
        const device InstanceResources &resources =
            instanceArgs.instances[instanceId];

        intersection hit = firstHitBVH(shadowRay, resources.blasNodes,
                                       resources.primitives,
                                       resources.primitiveIndices,
                                       meta.primitiveCount,
                                       int(meta.rootNodeIndex), instanceId,
                                       maxDistance);
        if (hit.primitiveId != -1) {
          if (instanceId == int(targetInstance) &&
              targetPrimitive >= 0 &&
              hit.primitiveId == targetPrimitive) {
            // Ignore the sampled light itself when it is exactly at the
            // maximum distance.
            if (hit.t >= maxDistance * 0.999f)
              continue;
          }
          return true;
        }
      }
    } else {
      int rightChild = -second;
      stack[stackPtr++] = leftFirst;
      stack[stackPtr++] = rightChild;
    }
  }

  return false;
}

inline bool sampleLightPrimitive(int primitiveType,
                                 device const float4 *primitives,
                                 int baseIndex,
                                 thread uint32_t &seed,
                                 thread float3 &position,
                                 thread float3 &normal) {
  float4 p0 = primitives[baseIndex + 0];
  float4 p1 = primitives[baseIndex + 1];
  float4 p2 = primitives[baseIndex + 2];
  float4 p3 = primitives[baseIndex + 3];

  if (primitiveType == 0) {
    float3 center = p0.xyz;
    float radius = p1.x;
    float u1 = randomFloat(seed);
    seed = random(seed);
    float u2 = randomFloat(seed);
    seed = random(seed);
    float z = 1.0 - 2.0 * u1;
    float r = sqrt(fmax(0.0, 1.0 - z * z));
    float phi = 2.0 * M_PI * u2;
    float x = r * cos(phi);
    float y = r * sin(phi);
    normal = normalize(float3(x, y, z));
    position = center + radius * normal;
    return true;
  } else if (primitiveType == 1) {
    float3 v0 = p0.xyz;
    float3 edge1 = p1.xyz;
    float3 edge2 = p2.xyz;
    float u1 = randomFloat(seed);
    seed = random(seed);
    float u2 = randomFloat(seed);
    seed = random(seed);
    float su1 = sqrt(u1);
    float b1 = su1 * (1.0 - u2);
    float b2 = su1 * u2;
    position = v0 + b1 * edge1 + b2 * edge2;
    normal = p3.xyz;
    float lenSq = dot(normal, normal);
    if (lenSq < 1e-12)
      normal = normalize(cross(edge1, edge2));
    else
      normal = normalize(normal);
    return true;
  } else if (primitiveType == 2) {
    float3 center = p0.xyz;
    float3 u = p1.xyz;
    float3 v = p2.xyz;
    float u1 = randomFloat(seed);
    seed = random(seed);
    float u2 = randomFloat(seed);
    seed = random(seed);
    float sx = 2.0 * u1 - 1.0;
    float sy = 2.0 * u2 - 1.0;
    position = center + sx * u + sy * v;
    normal = p3.xyz;
    float lenSq = dot(normal, normal);
    if (lenSq < 1e-12)
      normal = normalize(cross(u, v));
    else
      normal = normalize(normal);
    return true;
  }

  return false;
}

inline float4 rayColor(ray r, float3 rayDx, float3 rayDy,
                       device const float4 *tlasNodes,
                       device const int *tlasInstanceIndices,
                       uint tlasNodeCount,
                       device InstanceArgumentBuffer &instanceArgs,
                       constant InstanceMetadata *instanceMetadata,
                       device const Light *lights,
                       uint lightCount,
                       thread uint32_t &seed, uint maxRayDepth,
                       uint debugAS, uint blasNodeCount) {
  float footprint = length(cross(rayDx, rayDy));
  float lodAtten = 1.0 / (1.0 + footprint);
  if (debugAS == 1) {
    if (tlasNodeCount == 0)
      return float4(0.0, 0.0, 0.0, 1.0);
    constexpr int stackSize = 128;
    int stack[stackSize];
    int stackPtr = 0;
    if (stackPtr < stackSize)
      stack[stackPtr++] = 0;
    while (stackPtr > 0) {
      int nodeIdx = stack[--stackPtr];
      float3 bmin = tlasNodes[2 * nodeIdx + 0].xyz;
      float3 bmax = tlasNodes[2 * nodeIdx + 1].xyz;
      int leftFirst = as_type<int>(tlasNodes[2 * nodeIdx + 0].w);
      int second = as_type<int>(tlasNodes[2 * nodeIdx + 1].w);
      if (!intersectAABB(r, bmin, bmax, 0.0001, INFINITY))
        continue;
      if (second > 0) {
        float t = (tlasNodeCount > 1) ? float(nodeIdx) / float(tlasNodeCount - 1)
                                      : 0.0;
        return float4(t, 1.0 - t, 0.0, 1.0);
      }
      int rightChild = -second;
      if (stackPtr + 2 <= stackSize) {
        stack[stackPtr++] = leftFirst;
        stack[stackPtr++] = rightChild;
      }
    }
    return float4(0.0, 0.0, 0.0, 1.0);
  } else if (debugAS == 2) {
    intersection bestHit;
    bestHit.t = INFINITY;
    bestHit.primitiveId = -1;
    bestHit.nodeIndex = -1;
    bestHit.instanceId = -1;
    if (tlasNodeCount == 0)
      return float4(0.0, 0.0, 0.0, 1.0);
    constexpr int stackSize = 128;
    int stack[stackSize];
    int stackPtr = 0;
    if (stackPtr < stackSize)
      stack[stackPtr++] = 0;
    while (stackPtr > 0) {
      int nodeIdx = stack[--stackPtr];
      float3 bmin = tlasNodes[2 * nodeIdx + 0].xyz;
      float3 bmax = tlasNodes[2 * nodeIdx + 1].xyz;
      int leftFirst = as_type<int>(tlasNodes[2 * nodeIdx + 0].w);
      int second = as_type<int>(tlasNodes[2 * nodeIdx + 1].w);
      if (!intersectAABB(r, bmin, bmax, 0.0001, bestHit.t))
        continue;
      if (second > 0) {
        for (int i = 0; i < second; ++i) {
          int instanceId = tlasInstanceIndices[leftFirst + i];
          if (instanceId < 0 || instanceId >= MAX_INSTANCE_COUNT)
            continue;
          const constant InstanceMetadata &meta = instanceMetadata[instanceId];
          if (meta.primitiveCount == 0)
            continue;
          const device InstanceResources &resources =
              instanceArgs.instances[instanceId];
          intersection hit = firstHitBVH(r, resources.blasNodes,
                                         resources.primitives,
                                         resources.primitiveIndices,
                                         meta.primitiveCount,
                                         int(meta.rootNodeIndex), instanceId);
          if (hit.primitiveId != -1 && hit.t < bestHit.t)
            bestHit = hit;
        }
      } else {
        int rightChild = -second;
        if (stackPtr + 2 <= stackSize) {
          stack[stackPtr++] = leftFirst;
          stack[stackPtr++] = rightChild;
        }
      }
    }
    if (bestHit.primitiveId != -1) {
      float t = (blasNodeCount > 1)
                    ? float(bestHit.nodeIndex) / float(blasNodeCount - 1)
                    : 0.0;
      return float4(t, 0.0, 1.0 - t, 1.0);
    }
    return float4(0.0, 0.0, 0.0, 1.0);
  }

  float3 throughput = float3(1.0);
  float3 radiance = float3(0.0);

  for (uint depth = 0; depth < maxRayDepth; ++depth) {
    intersection bestHit;
    bestHit.t = INFINITY;
    bestHit.primitiveId = -1;
    bestHit.nodeIndex = -1;
    bestHit.instanceId = -1;

    if (tlasNodeCount > 0) {
      constexpr int stackSize = 128;
      int stack[stackSize];
      int stackPtr = 0;
      if (stackPtr < stackSize)
        stack[stackPtr++] = 0;

      while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        float3 bmin = tlasNodes[2 * nodeIdx + 0].xyz;
        float3 bmax = tlasNodes[2 * nodeIdx + 1].xyz;
        int leftFirst = as_type<int>(tlasNodes[2 * nodeIdx + 0].w);
        int second = as_type<int>(tlasNodes[2 * nodeIdx + 1].w);
        if (!intersectAABB(r, bmin, bmax, 0.0001, bestHit.t))
          continue;
        if (second > 0) {
          for (int i = 0; i < second; ++i) {
            int instanceId = tlasInstanceIndices[leftFirst + i];
            if (instanceId < 0 || instanceId >= MAX_INSTANCE_COUNT)
              continue;
            const constant InstanceMetadata &meta =
                instanceMetadata[instanceId];
            if (meta.primitiveCount == 0)
              continue;
            const device InstanceResources &resources =
                instanceArgs.instances[instanceId];

            intersection hit = firstHitBVH(r, resources.blasNodes,
                                           resources.primitives,
                                           resources.primitiveIndices,
                                           meta.primitiveCount,
                                           int(meta.rootNodeIndex),
                                           instanceId);
            if (hit.primitiveId != -1 && hit.t < bestHit.t)
              bestHit = hit;
          }
        } else {
          int rightChild = -second;
          if (stackPtr + 2 <= stackSize) {
            stack[stackPtr++] = leftFirst;
            stack[stackPtr++] = rightChild;
          }
        }
      }
    }

    if (bestHit.primitiveId == -1) {
      float3 unitDir = normalize(r.direction);
      float t = 0.5 * (unitDir.y + 1.0);
      float3 skyColor = mix(float3(1.0), float3(0.6, 0.7, 1.0), t);
      radiance += throughput * skyColor;
      break;
    }
    int matIndex = bestHit.primitiveId * 2;
    const constant InstanceMetadata &hitMeta =
        instanceMetadata[bestHit.instanceId];
    if (matIndex + 1 >= int(hitMeta.primitiveCount) * 2)
      break;
    const device InstanceResources &hitResources =
        instanceArgs.instances[bestHit.instanceId];
    float4 m0 = hitResources.materials[matIndex + 0];
    float4 m1 = hitResources.materials[matIndex + 1];

    float3 albedo = m0.xyz * lodAtten;
    float materialType = m0.w;
    float3 emissionColor = m1.xyz;
    float emissionPower = m1.w;

    bool isEmitter = (emissionPower > 0.0 || materialType == 2);
    if (isEmitter) {
      radiance += throughput * (emissionColor * emissionPower);
    }

    if (lightCount > 0 && lights != nullptr && materialType == 0.0 &&
        !isEmitter) {
      float selectXi = randomFloat(seed);
      seed = random(seed);

      uint chosen = 0;
      if (lightCount > 1) {
        uint left = 0;
        uint right = lightCount;
        while (left < right) {
          uint mid = (left + right) / 2;
          float cdf = lights[mid].cdf.x;
          if (selectXi < cdf)
            right = mid;
          else
            left = mid + 1;
        }
        chosen = min(left, lightCount - 1);
      }

      const device Light &lightEntry = lights[chosen];
      float lightPdf = lightEntry.meta.w;
      float area = lightEntry.meta.z;
      if (lightPdf > 0.0 && area > 0.0) {
        uint lightInstance = as_type<uint>(lightEntry.meta.x);
        uint lightPrimitive = as_type<uint>(lightEntry.meta.y);
        if (lightInstance < MAX_INSTANCE_COUNT) {
          const constant InstanceMetadata &lightMeta =
              instanceMetadata[lightInstance];
          if (lightPrimitive < lightMeta.primitiveCount) {
            const device InstanceResources &lightResources =
                instanceArgs.instances[lightInstance];
            if (lightResources.primitives != nullptr &&
                lightResources.materials != nullptr) {
              int baseIndex = int(lightPrimitive) * PRIMITIVE_STRIDE;
              int maxIndex = int(lightMeta.primitiveCount) * PRIMITIVE_STRIDE;
              if (baseIndex + 3 < maxIndex) {
                float3 lightPos;
                float3 lightNormal;
                bool sampled = sampleLightPrimitive(
                    int(lightResources.primitives[baseIndex].w),
                    lightResources.primitives, baseIndex, seed, lightPos,
                    lightNormal);
                if (sampled) {
                  float3 toLight = lightPos - bestHit.point;
                  float distSq = dot(toLight, toLight);
                  float dist = sqrt(distSq);
                  if (dist > 1e-4) {
                    float3 wi = toLight / dist;
                    float cosSurface = fmax(0.0, dot(bestHit.normal, wi));
                    float cosLight = fmax(0.0, dot(lightNormal, -wi));
                    if (cosSurface > 0.0 && cosLight > 0.0) {
                      float shadowMax = dist - 0.001;
                      bool blocked = false;
                      if (shadowMax > 0.0) {
                        ray shadowRay{bestHit.point + bestHit.normal * 0.0005f,
                                      wi};
                        shadowRay.min_distance = 0.0001f;
                        shadowRay.max_distance = shadowMax;
                        blocked = isOccluded(
                            shadowRay, shadowMax, tlasNodes, tlasInstanceIndices,
                            tlasNodeCount, instanceArgs, instanceMetadata,
                            lightInstance, int(lightPrimitive));
                      }
                      if (!blocked) {
                        float3 emission = lightEntry.emission.xyz *
                                          lightEntry.emission.w;
                        float pdfArea = lightPdf / area;
                        if (pdfArea > 0.0) {
                          float3 bsdfVal = evaluateBSDF(-r.direction, wi,
                                                        bestHit.normal,
                                                        materialType, albedo);
                          float geometry = cosLight / max(distSq, 1e-6f);
                          float3 direct = emission * bsdfVal * cosSurface *
                                          geometry / pdfArea;
                          radiance += throughput * direct;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    float3 offsetNormal = bestHit.frontFace ? bestHit.normal : -bestHit.normal;
    r.origin = bestHit.point + 0.0001 * offsetNormal;
    BSDFSample sample = sampleBSDF(r.direction, bestHit.normal,
                                   bestHit.frontFace, materialType, albedo,
                                   seed);
    if (sample.pdf <= 0.0 || all(sample.weight <= float3(0.0)))
      break;

    throughput *= sample.weight;
    r.direction = sample.direction;
    r.min_distance = 0.0001;
    r.max_distance = INFINITY;

    if (depth >= 5) {
      float p = max(throughput.x, max(throughput.y, throughput.z));
      float randVal = randomFloat(seed);
      seed = random(seed);
      if (randVal >= p)
        break;
      throughput /= p;
    }
  }

  return float4(clamp(radiance, 0.0, 1.0), 1.0);
}

#endif
