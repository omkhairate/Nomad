#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct OverlayUniforms {
    simd::float4x4 viewProjection;
};

struct OverlayVertexOut {
    float4 position [[position]];
    float4 color;
};

struct OverlayLineVertex {
    float3 position;
    float4 color;
};

vertex OverlayVertexOut frustumDebugVertexMain(
    uint vertexId [[vertex_id]],
    device const OverlayLineVertex *vertices [[buffer(0)]],
    constant OverlayUniforms &uniforms [[buffer(1)]]) {
    OverlayVertexOut out;
    OverlayLineVertex lineVertex = vertices[vertexId];
    float3 worldPos = lineVertex.position;
    out.position = uniforms.viewProjection * float4(worldPos, 1.0);
    out.color = lineVertex.color;
    return out;
}
