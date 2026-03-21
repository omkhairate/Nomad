#include <metal_stdlib>

using namespace metal;

struct OverlayVertexOut {
    float4 position [[position]];
    float4 color;
};

fragment float4 frustumDebugFragmentMain(OverlayVertexOut in [[stage_in]]) {
    return in.color;
}
