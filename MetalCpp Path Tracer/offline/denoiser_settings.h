#pragma once

// Configuration parameters for the offline EXR denoiser.
// The "sharpened" preset tightens the bilateral filter kernels and blends
// the filtered result with the original color instead of fully overwriting it.
// This keeps more high-frequency detail at the cost of retaining a little noise.
namespace MetalCppPathTracer::DenoiserSettings {

// Spatial Gaussian sigma controlling the bilateral filter footprint.
inline constexpr float kSharpenedSpatialSigma = 0.5f;

// Albedo Gaussian sigma used to reject neighbors with differing base colors.
inline constexpr float kSharpenedAlbedoSigma = 0.15f;

// Normal Gaussian sigma controlling how aggressively surface orientation
// differences suppress contributions from neighbors.
inline constexpr float kSharpenedNormalSigma = 0.05f;

// Blend factor between the noisy input color and the filtered color.
// 0 keeps the original, 1 fully replaces it with the filtered estimate.
inline constexpr float kSharpenedDenoiseStrength = 0.8f;

// Optional clamp to [0, 1] for LDR previews. Disabled by default so HDR
// highlights are preserved when writing EXR outputs.
inline constexpr bool kSharpenedClampOutput = false;

} // namespace MetalCppPathTracer::DenoiserSettings
