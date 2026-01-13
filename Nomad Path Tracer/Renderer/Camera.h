#ifndef CAMERA_H
#define CAMERA_H

#include <simd/simd.h>
#include "InputSystem.h"

namespace NomadPathTracer {

namespace Camera
{

struct State
{
    simd::float3 position {0, 0, 0};
    simd::float3 forward {0, 0, -1};
    simd::float3 up {0, 1, 0};
    float verticalFov = 60.0f;
    float focalLength = 1.0f;
    float aperture = 0.0f;
    float focusDistance = 1.0f;
};

inline simd::float3 position;
inline simd::float3 forward;
inline simd::float3 up;

inline float verticalFov;
inline float focalLength;
inline float aperture;
inline float focusDistance;
inline simd::float2 screenSize;
inline float deltaTime = 0.0f;

inline constexpr float movementSpeed = 0.1;
inline constexpr float rotationSpeed = 0.002;
inline constexpr float zoomSpeed = 0.1;

inline static void reset()
{
    position = {0, 20, 50};     // Behind the spheres, looking forward
    forward  = {0, 0, -1};      // Looking straight toward -Z
    up       = {0, 1, 0};       // Y is up

    verticalFov = 60.0;
    focalLength = 1.0;
    aperture = 0.0f;
    focusDistance = 1.0f;
    deltaTime = 0.0f;
}

inline State captureState()
{
    return {position, forward, up, verticalFov, focalLength, aperture, focusDistance};
}

inline void applyState(const State& state)
{
    position = state.position;
    forward = state.forward;
    up = state.up;
    verticalFov = state.verticalFov;
    focalLength = state.focalLength;
    aperture = state.aperture;
    focusDistance = state.focusDistance;
}


inline bool move(simd::float3 movementDirection)
{
    if(simd::length_squared(movementDirection) == 0 || deltaTime <= 0.0f) return false;
    
    const simd::float3 cameraRight = simd::normalize(simd::cross(forward, simd::float3{0, 1, 0}));
    
    const simd::float3 forwardMovementDirection = simd::cross(simd::float3{0, 1, 0}, cameraRight);
    
    const float frameMovementSpeed = movementSpeed * deltaTime;

    position += frameMovementSpeed * simd::normalize(cameraRight*InputSystem::movementInput.x +
                                                simd::float3{0, 1, 0}*InputSystem::movementInput.y +
                                                forwardMovementDirection*InputSystem::movementInput.z);
    return true;

}

inline bool rotate(simd::float2 angles)
{
    if(simd::length_squared(angles) == 0 || deltaTime <= 0.0f) return false;

    simd::float3 cameraRight = simd::cross(forward, simd::float3{0, 1, 0});
    const float frameRotationSpeed = rotationSpeed * deltaTime;

    simd::quatf rotationUp = simd::quatf(-InputSystem::rotationInput.y*frameRotationSpeed, cameraRight);
    Camera::forward = simd::normalize(simd_act(rotationUp, forward));

    cameraRight = simd::cross(Camera::forward, simd::float3{0, 1, 0});
    up = simd::normalize(simd::cross(cameraRight, forward));
    simd::quatf rotationRight = simd::quatf(-InputSystem::rotationInput.x*frameRotationSpeed, up);
    forward = simd::normalize(simd_act(rotationRight, forward));

    return true;
}

inline bool zoom(float zoomAmount)
{
    if (zoomAmount == 0 || deltaTime <= 0.0f) return false;

    const float frameZoomSpeed = zoomSpeed * deltaTime;

    verticalFov = simd::clamp(verticalFov + zoomAmount * frameZoomSpeed, 30.0f, 120.0f);
    
    return true;
}

inline bool transformWithInputs()
{
    if(InputSystem::resetInput) reset();
    
    const bool reseted = InputSystem::resetInput;
    const bool moved = move(InputSystem::movementInput);
    const bool rotated = rotate(InputSystem::rotationInput);
    const bool zoomed = zoom(InputSystem::zoomInput);
    
    InputSystem::clearInputs();
    
    const bool transformed =  (reseted || moved || rotated || zoomed);
    
    return transformed;
}
};

};

#endif
