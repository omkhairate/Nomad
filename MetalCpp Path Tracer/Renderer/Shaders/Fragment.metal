#include <metal_stdlib>

using namespace metal;

#include "Structs.h"

fragment float4 presentMain(
    v2f in [[stage_in]],
    texture2d<half, access::sample> accumulation [[texture(0)]],
    texture2d<half, access::sample> importance [[texture(1)]],
    constant UniformsData &uniforms [[buffer(0)]]) {
  constexpr sampler linearSampler(filter::linear, address::clamp_to_edge);
  if (uniforms.debugOutputMode != 0 && importance.get_width() > 0 &&
      importance.get_height() > 0) {
    half4 imp = importance.sample(linearSampler, in.uv);
    float scale = max(uniforms.importanceVisualizationScale, 0.0f);
    float detail = clamp(float(imp.y) * scale, 0.0f, 1.0f);
    float completion = clamp(float(imp.w), 0.0f, 1.0f);
    float variance = clamp(float(imp.z) * scale, 0.0f, 1.0f);
    float3 debugColor = float3(detail, completion, variance);
    return float4(debugColor, 1.0f);
  }
  return float4(accumulation.sample(linearSampler, in.uv));
}
