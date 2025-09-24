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
    int instanceId = -1;
};


struct InstanceMetadata
{
    uint primitiveCount;
    uint blasNodeCount;
    uint rootNodeIndex;
    uint padding;
};

#define MAX_INSTANCE_COUNT 16384

struct InstanceResources
{
    device const float4* blasNodes [[id(0)]];
    device const float4* primitives [[id(1)]];
    device const float4* materials [[id(2)]];
    device const int* primitiveIndices [[id(3)]];
};

struct InstanceArgumentBuffer
{
    array<InstanceResources, MAX_INSTANCE_COUNT> instances;
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
    uint residentInstanceCount;
    uint totalInstanceCount;
};




#endif
