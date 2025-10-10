#ifndef OBSERVER_CAMERA_H
#define OBSERVER_CAMERA_H

#include <simd/simd.h>

namespace MetalCppPathTracer {

namespace ObserverCamera {

inline simd::float3 position = {0.0f, 0.0f, 0.0f};
inline simd::float3 forward = {0.0f, 0.0f, -1.0f};
inline simd::float3 up = {0.0f, 1.0f, 0.0f};
inline float verticalFov = 45.0f;
inline simd::float2 screenSize = {320.0f, 180.0f};

inline void reset() {
  position = {0.0f, 20.0f, 60.0f};
  forward = simd::normalize(simd::float3{0.0f, -0.3f, -1.0f});
  up = {0.0f, 1.0f, 0.0f};
  verticalFov = 45.0f;
}

inline void configureForScene(const simd::float3 &center, float radius) {
  float safeRadius = radius;
  if (!(safeRadius > 1e-3f)) {
    safeRadius = 25.0f;
  }
  simd::float3 offset = simd::normalize(simd::float3{1.0f, 1.0f, 1.0f});
  position = center + offset * (safeRadius * 2.5f);
  forward = simd::normalize(center - position);
  if (simd::length_squared(forward) < 1e-6f) {
    forward = simd::normalize(-position);
  }
  simd::float3 worldUp = {0.0f, 1.0f, 0.0f};
  simd::float3 right = simd::cross(forward, worldUp);
  if (simd::length_squared(right) < 1e-6f) {
    right = {1.0f, 0.0f, 0.0f};
  }
  up = simd::normalize(simd::cross(right, forward));
}

inline void lookAt(const simd::float3 &target) {
  simd::float3 dir = target - position;
  if (simd::length_squared(dir) < 1e-6f) {
    return;
  }
  forward = simd::normalize(dir);
  simd::float3 worldUp = {0.0f, 1.0f, 0.0f};
  simd::float3 right = simd::cross(forward, worldUp);
  if (simd::length_squared(right) < 1e-6f) {
    right = {1.0f, 0.0f, 0.0f};
  }
  up = simd::normalize(simd::cross(right, forward));
}

} // namespace ObserverCamera

} // namespace MetalCppPathTracer

#endif
