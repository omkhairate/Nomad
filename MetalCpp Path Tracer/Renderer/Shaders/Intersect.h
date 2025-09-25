#ifndef INTERSECT_H
#define INTERSECT_H

#include <metal_stdlib>
using namespace metal;

#include "Structs.h"

// Sphere intersection (your original function)
inline float sphereIntersection(thread const ray &ray, device const float4 &sphere)
{
    const float3 sphereCenter = sphere.xyz;
    const float r = sphere.w;

    float3 originToCenter = sphereCenter - ray.origin;
    float a = length_squared(ray.direction);
    float h = dot(ray.direction, originToCenter);
    float c = length_squared(originToCenter) - r*r;

    float discriminant = h*h - a*c;

    if (discriminant < 0) return INFINITY;

    float sqrtDiscriminant = sqrt(discriminant);

    float root = (h - sqrtDiscriminant) / a;

    if (ray.min_distance > root || ray.max_distance < root)
    {
        root = (h + sqrtDiscriminant) / a;
        if (ray.max_distance < root || ray.min_distance > root)
        {
            return INFINITY;
        }
    }

    return root;
}


// Triangle intersection (Möller–Trumbore)
inline bool triangleIntersection(
    thread const ray &ray,
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

    if (t < ray.min_distance || t > ray.max_distance)
        return false;

    // Valid intersection
    tHit = t;
    barycentric = float3(1.0 - u - v, u, v);
    return true;
}

// Rectangle intersection
inline bool rectangleIntersection(
    thread const ray &ray,
    float3 center,
    float3 e1,
    float3 e2,
    thread float &tHit,
    thread float3 &normal)
{
    normal = normalize(cross(e1, e2));
    float denom = dot(normal, ray.direction);
    if (fabs(denom) < 1e-6)
        return false;

    float t = dot(center - ray.origin, normal) / denom;
    if (t < ray.min_distance || t > ray.max_distance)
        return false;

    float3 pHit = ray.origin + t * ray.direction;
    float3 rel = pHit - center;
    float u = dot(rel, e1) / dot(e1, e1);
    float v = dot(rel, e2) / dot(e2, e2);
    if (fabs(u) > 1.0 || fabs(v) > 1.0)
        return false;

    tHit = t;
    return true;
}

#endif


