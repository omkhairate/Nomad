#ifndef RENDERER_VIEW_CONTROLLER_HPP
#define RENDERER_VIEW_CONTROLLER_HPP

#include <AppKit/AppKit.hpp>
#include <Metal/Metal.hpp>

namespace NomadPathTracer {

class RendererViewController {
public:
  NS::ViewController *get(CGRect frame, MTL::Device *device);
};

} // namespace NomadPathTracer

#endif
