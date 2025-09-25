#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include "Material.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Rectangle.h"
#include <cstdint>

namespace MetalCppPathTracer {

enum class PrimitiveType {
    Sphere = 0,
    Triangle = 1,
    Rectangle = 2
};

static constexpr uint32_t kInvalidMeshId = 0;

struct Primitive {
    PrimitiveType type;
    Material material;
    uint32_t meshId = kInvalidMeshId;
    union {
        Sphere sphere;
        Triangle triangle;
        Rectangle rectangle;
    };
};

} // namespace MetalCppPathTracer

#endif
