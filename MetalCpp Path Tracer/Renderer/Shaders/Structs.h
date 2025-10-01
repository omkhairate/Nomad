#ifndef STRUCTS_H
#define STRUCTS_H

#include <metal_stdlib>
#include <metal_raytracing>

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
    uint geometryIndex;
    uint geometryCount;
};


struct ObjectGeometry
{
    device float3* vertexBuffer [[id(0)]];
    device uint* indexBuffer [[id(1)]];
    metal::raytracing::acceleration_structure blas [[id(2)]];
};

struct ObjectGeometryTable
{
    array<ObjectGeometry, 1> entries;
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
