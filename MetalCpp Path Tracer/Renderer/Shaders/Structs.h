#ifndef STRUCTS_H
#define STRUCTS_H

#include <metal_stdlib>

struct v2f
{
    float4 position [[position]];
    float2 uv;
};

struct intersection
{
    float t = INFINITY;
    float3 point;
    float3 normal;
    bool frontFace;
    int primitiveId = -1;
    int isTriangle = 0;
    int nodeIndex = -1; // BLAS leaf node index
};


struct UniformsData
{
    int primitiveIndex;
    simd::float3 cameraPosition;
    simd::float2 screenSize;

    simd::float3 viewportU;
    simd::float3 viewportV;
    simd::float3 firstPixelPosition;
    simd::float3 rayDx;
    simd::float3 rayDy;

    simd::float3 randomSeed;

    uint64_t primitiveCount;
    uint64_t triangleCount;
    uint64_t frameCount = 0;
    uint64_t totalPrimitiveCount;
    uint64_t tlasNodeCount;
    uint64_t blasNodeCount;
    uint maxRayDepth;
    uint debugAS;
    uint lightCount;
    float lightTotalWeight;
    uint padding0;
    uint padding1;
};




#endif
