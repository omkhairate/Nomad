#ifndef APPLICATION_DELEGATE_H
#define APPLICATION_DELEGATE_H

#include <AppKit/AppKit.hpp>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <MetalKit/MetalKit.hpp>

#include "RendererViewController.hpp"

namespace NomadPathTracer {

class ApplicationDelegate : public NS::ApplicationDelegate {
public:
  ~ApplicationDelegate();

  ApplicationDelegate() = default;
  virtual void
  applicationWillFinishLaunching(NS::Notification *pNotification) override;
  virtual void
  applicationDidFinishLaunching(NS::Notification *pNotification) override;
  virtual bool applicationShouldTerminateAfterLastWindowClosed(
      NS::Application *pSender) override;

private:
  NS::Window *_pWindow;
  MTL::Device *_pDevice;
  NS::ViewController *_pRendererViewController = nullptr;
  RendererViewController _rendererViewController;
  bool _initialized = false;
};

}; // namespace NomadPathTracer

#endif //  APPLICATION_DELEGATE_H
