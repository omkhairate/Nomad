#include <metal_stdlib>

using namespace metal;

struct AccumulationDecayUniforms {
    float factor;
    uint hasHistory;
    uint hasOutput;
    uint padding;
};

kernel void accumulationDecayMain(
    texture2d<half, access::read_write> accumulationA [[texture(0)]],
    texture2d<half, access::read_write> accumulationB [[texture(1)]],
    texture2d<half, access::read_write> sampleCount [[texture(2)]],
    texture2d<half, access::read_write> albedoAccum [[texture(3)]],
    texture2d<half, access::read_write> normalAccum [[texture(4)]],
    texture2d<half, access::read_write> sampleImportance [[texture(5)]],
    texture2d<half, access::read_write> denoisedHistory [[texture(6)]],
    texture2d<half, access::read_write> denoisedOutput [[texture(7)]],
    constant AccumulationDecayUniforms &uniforms [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]) {
  uint width = accumulationA.get_width();
  uint height = accumulationA.get_height();
  if (gid.x >= width || gid.y >= height)
    return;

  float factor = uniforms.factor;

  half4 valueA = accumulationA.read(gid);
  accumulationA.write(half4(float4(valueA) * factor), gid);

  half4 valueB = accumulationB.read(gid);
  accumulationB.write(half4(float4(valueB) * factor), gid);

  half4 samples = sampleCount.read(gid);
  samples.x = half(float(samples.x) * factor);
  sampleCount.write(samples, gid);

  half4 albedo = albedoAccum.read(gid);
  albedoAccum.write(half4(float4(albedo) * factor), gid);

  half4 normal = normalAccum.read(gid);
  normalAccum.write(half4(float4(normal) * factor), gid);

  half4 importance = sampleImportance.read(gid);
  sampleImportance.write(half4(float4(importance) * factor), gid);

  if (uniforms.hasHistory != 0u) {
    half4 history = denoisedHistory.read(gid);
    denoisedHistory.write(half4(float4(history) * factor), gid);
  }

  if (uniforms.hasOutput != 0u) {
    half4 output = denoisedOutput.read(gid);
    denoisedOutput.write(half4(float4(output) * factor), gid);
  }
}
