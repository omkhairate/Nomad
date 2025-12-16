#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct OverlayUniforms {
    simd::float4x4 viewProjection;
};

struct OverlayVertexOut {
    float4 position [[position]];
};

vertex OverlayVertexOut frustumDebugVertexMain(
    uint vertexId [[vertex_id]],
    device const float3 *positions [[buffer(0)]],
    constant OverlayUniforms &uniforms [[buffer(1)]]) {
    OverlayVertexOut out;
    float3 worldPos = positions[vertexId];
    out.position = uniforms.viewProjection * float4(worldPos, 1.0);
    return out;
}
