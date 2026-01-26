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
    float3 tangent = float3(0.0f);
    float3 bitangent = float3(0.0f);
    int supportsNormalMap = 0;
    bool frontFace;
    int primitiveId = -1;
    int isTriangle = 0;
    int nodeIndex = -1; // BLAS leaf node index
    float2 uv = float2(0.0f);
    float2 barycentric = float2(0.0f);
};

struct InstanceRecord
{
    int blasRootIndex;
    uint primitiveBase;
    uint primitiveCount;
    uint primitiveIndexBase;
};

struct GeometryHandle
{
    uint64_t vertexBufferAddress = 0;
    uint64_t indexBufferAddress = 0;
    uint vertexStride = 0;
    uint indexStride = 0;
    uint vertexCount = 0;
    uint indexCount = 0;
    uint instanceSlot = 0;
    uint padding = 0;
};


struct UniformsData
{
    int primitiveIndex;
    simd::float3 cameraPosition;
    simd::float2 screenSize;
    float aperture;
    float focusDistance;

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
    uint textureCount;
    uint environmentMapEnabled;
    float environmentMapIntensity;
    float environmentPadding0;
    float environmentPadding1;
    float cameraMotionMetric;
    simd::float4x4 prevViewProjection;
};

struct TileRegion
{
    uint2 origin;
    uint2 size;
};

struct PackedTexture
{
    uint offset;
    uint width;
    uint height;
    uint flags;
};




#endif
