#include <metal_stdlib>

using namespace metal;

#include "PathTracing.h"

inline uint hashUint(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352d;
  x ^= x >> 15;
  x *= 0x846ca68b;
  x ^= x >> 16;
  return x;
}

kernel void restirShadeMain(
    device const RestirReservoir *reservoirBuffer [[buffer(0)]],
    constant UniformsData &uniforms [[buffer(1)]],
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
  if (uniforms.restirDebugMode == 1u) {
    float denom = max(float(reservoir.m), 1.0f);
    float normalized = clamp(reservoir.wSum / denom, 0.0f, 1.0f);
    reservoirEstimate = float3(normalized);
  } else if (uniforms.restirDebugMode == 2u) {
    uint hashed = hashUint(reservoir.packedLightId ^ index);
    float3 color = float3(float(hashed & 0xffu),
                          float((hashed >> 8) & 0xffu),
                          float((hashed >> 16) & 0xffu)) /
                   255.0f;
    reservoirEstimate = color;
  }
  reservoirEstimate = clamp(reservoirEstimate, 0.0f, 1.0f);
  currentFrame.write(float4(reservoirEstimate, 1.0f), gid);
}
