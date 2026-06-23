#ifndef INPUT_SYSTEM_H
#define INPUT_SYSTEM_H

#include <cstdint>
#include <simd/simd.h>

namespace NomadPathTracer {

namespace InputSystem
{

enum class DebugAccelerationMode : uint32_t {
    None = 0,
    Tlas = 1,
    Blas = 2,
    ShadowVisibility = 3,
};

inline constexpr bool isKnownDebugAccelerationMode(DebugAccelerationMode mode)
{
    return mode == DebugAccelerationMode::None ||
           mode == DebugAccelerationMode::Tlas ||
           mode == DebugAccelerationMode::Blas ||
           mode == DebugAccelerationMode::ShadowVisibility;
}

inline constexpr bool isExposedDebugAccelerationMode(DebugAccelerationMode mode)
{
    return mode == DebugAccelerationMode::None ||
           mode == DebugAccelerationMode::Tlas ||
           mode == DebugAccelerationMode::Blas;
}

inline simd::float3 movementInput {0};
inline simd::float2 rotationInput {0};
inline float zoomInput = 0;
inline bool resetInput = 0;
inline DebugAccelerationMode debugAccelerationMode = DebugAccelerationMode::None;
inline bool observerToggleRequest = false;

inline constexpr uint32_t rawDebugAccelerationMode(DebugAccelerationMode mode)
{
    return static_cast<uint32_t>(mode);
}

inline void setDebugAccelerationMode(DebugAccelerationMode mode)
{
    debugAccelerationMode =
        isKnownDebugAccelerationMode(mode) ? mode : DebugAccelerationMode::None;
}

inline void toggleDebugAccelerationMode(DebugAccelerationMode mode)
{
    if (!isExposedDebugAccelerationMode(mode))
        return;

    debugAccelerationMode =
        (debugAccelerationMode == mode) ? DebugAccelerationMode::None : mode;
}

inline void clearInputs()
{
    rotationInput = 0;
    zoomInput = 0;
    resetInput = 0;
}

};

};

#endif
