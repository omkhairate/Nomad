#ifndef STRUCTS_H
#define STRUCTS_H

#include <metal_stdlib>

using namespace metal;

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

struct InstanceRecord
{
    int blasRootIndex;
    uint primitiveBase;
    uint primitiveCount;
    uint primitiveIndexBase;
    uint triangleBase;
    uint triangleCount;
};

struct GeometryHandle
{
    uint64_t vertexBufferAddress = 0;
    uint64_t indexBufferAddress = 0;
    uint vertexStride = 0;
    uint indexStride = 0;
    uint vertexCount = 0;
    uint indexCount = 0;
    uint triangleBase = 0;
    uint triangleCount = 0;
    uint instanceSlot = 0;
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
    uint sampleCountTextureIndex;
    uint sampleImportanceTextureIndex;
    uint minSamplesPerPixel;
    uint maxSamplesPerPixel;
};




#endif
