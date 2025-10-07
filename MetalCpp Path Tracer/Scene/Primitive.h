#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include "Material.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Rectangle.h"

namespace MetalCppPathTracer {

enum class PrimitiveType {
    Sphere = 0,
    Triangle = 1,
    Rectangle = 2
};

struct Primitive {
    PrimitiveType type;
    Material material;
    union {
        Sphere sphere;
        Triangle triangle;
        Rectangle rectangle;
    };
};

} // namespace MetalCppPathTracer

#endif
