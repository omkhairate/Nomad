#ifndef INTERSECT_H
#define INTERSECT_H

#include <metal_stdlib>
using namespace metal;

#include "Structs.h"

// Triangle intersection (Möller–Trumbore)
inline bool triangleIntersection(
    thread const Ray &ray,
    float3 v0,
    float3 v1,
    float3 v2,
    thread float &tHit,
    thread float3 &barycentric)
{
    const float EPSILON = 1e-6;

    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;

    float3 h = cross(ray.direction, edge2);
    float a = dot(edge1, h);

    if (abs(a) < EPSILON)
        return false; // Ray parallel to triangle

    float f = 1.0 / a;
    float3 s = ray.origin - v0;
    float u = f * dot(s, h);

    if (u < 0.0 || u > 1.0)
        return false;

    float3 q = cross(s, edge1);
    float v = f * dot(ray.direction, q);

    if (v < 0.0 || u + v > 1.0)
        return false;

    float t = f * dot(edge2, q);

    if (t < ray.minDistance || t > ray.maxDistance)
        return false;

    // Valid intersection
    tHit = t;
    barycentric = float3(1.0 - u - v, u, v);
    return true;
}

#endif


