#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include "Material.h"
#include "Triangle.h"

namespace MetalCppPathTracer {

enum class PrimitiveType {
    Triangle = 0
};

struct Primitive {
    PrimitiveType type{PrimitiveType::Triangle};
    Material material;
    Triangle triangle;
};

} // namespace MetalCppPathTracer

#endif
