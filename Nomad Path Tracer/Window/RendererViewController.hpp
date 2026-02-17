#ifndef RENDERER_VIEW_CONTROLLER_HPP
#define RENDERER_VIEW_CONTROLLER_HPP

#include <AppKit/AppKit.hpp>
#include <Metal/Metal.hpp>

namespace NomadPathTracer {

class RendererViewController {
public:
  ~RendererViewController();
  NS::View *get(CGRect frame, MTL::Device *device);

private:
  class ViewDelegate *_viewDelegate = nullptr;
};

} // namespace NomadPathTracer

#endif
