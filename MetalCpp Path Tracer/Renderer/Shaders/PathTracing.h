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
                                int instanceId) {
  intersection in;
  in.t = INFINITY;
  in.primitiveId = -1;
  in.isTriangle = 0;
  in.nodeIndex = -1;
  in.instanceId = instanceId;

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

    if (!intersectAABB(r, bmin, bmax, 0.0001, in.t))
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

        if (hitThis && tHit < in.t) {
          in.t = tHit;
          in.primitiveId = primIdx;
          in.normal = n;
          in.point = hit;
          in.isTriangle = primitiveType;
          in.nodeIndex = nodeIdx;
        }
      }
    } else {
      int rightChild = -second;
      stack[stackPtr++] = leftFirst;
      stack[stackPtr++] = rightChild;
    }
  }

  if (in.primitiveId != -1) {
    in.frontFace = dot(in.normal, r.direction) < 0.0;
    if (!in.frontFace)
      in.normal = -in.normal;
  }

  return in;
}

inline float4 rayColor(ray r, float3 rayDx, float3 rayDy,
                       device const float4 *tlasNodes,
                       device const int *tlasInstanceIndices,
                       uint tlasNodeCount,
                       device InstanceArgumentBuffer &instanceArgs,
                       constant InstanceMetadata *instanceMetadata,
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

  float4 absorption = float4(1.0);
  float4 light = float4(0.0);

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
      light += absorption * float4(skyColor, 1.0);
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

    if (emissionPower > 0.0 || materialType == 2) {
      light += absorption * float4(emissionColor, 1.0) * emissionPower;
    }

    float3 offsetNormal = bestHit.frontFace ? bestHit.normal : -bestHit.normal;
    r.origin = bestHit.point + 0.0001 * offsetNormal;
    scatter(r, bestHit, materialType, seed);
    seed = random(seed);
    absorption *= float4(albedo, 1.0);

    if (depth >= 5) {
      float p = max(absorption.x, max(absorption.y, absorption.z));
      float randVal = randomFloat(seed);
      seed = random(seed);
      if (randVal >= p)
        break;
      absorption /= p;
    }
  }

  return clamp(light, 0.0, 1.0);
}

#endif
