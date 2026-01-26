#include <metal_stdlib>

using namespace metal;

#include "Structs.h"

fragment float4 presentMain(v2f in [[stage_in]],
                             texture2d<float, access::sample> currentFrame [[texture(0)]]) {
  constexpr sampler linearSampler(filter::linear, address::clamp_to_edge);
  return float4(currentFrame.sample(linearSampler, in.uv));
}
