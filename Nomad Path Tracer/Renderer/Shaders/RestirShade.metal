#include <metal_stdlib>

using namespace metal;

#include "PathTracing.h"

kernel void restirShadeMain(
    device const RestirReservoir *reservoirBuffer [[buffer(0)]],
    texture2d<float, access::write> currentFrame [[texture(0)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (!reservoirBuffer)
    return;

  uint width = currentFrame.get_width();
  uint height = currentFrame.get_height();
  if (gid.x >= width || gid.y >= height)
    return;

  uint index = gid.y * width + gid.x;
  RestirReservoir reservoir = reservoirBuffer[index];
  float3 reservoirEstimate = finalizeReservoir(reservoir);
  reservoirEstimate = clamp(reservoirEstimate, 0.0f, 1.0f);
  currentFrame.write(float4(reservoirEstimate, 1.0f), gid);
}
