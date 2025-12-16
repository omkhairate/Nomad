#include <metal_stdlib>

using namespace metal;

#include "Structs.h"

fragment float4 presentMain(v2f in [[stage_in]],
                             texture2d<half, access::sample> accumulation [[texture(0)]]) {
  constexpr sampler linearSampler(filter::linear, address::clamp_to_edge);
  return float4(accumulation.sample(linearSampler, in.uv));
}
