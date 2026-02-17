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
  NS::Window *_pWindow = nullptr;
  MTL::Device *_pDevice = nullptr;
  NS::View *_pRendererView = nullptr;
  RendererViewController _rendererViewController;
  bool _initialized = false;
};

}; // namespace NomadPathTracer

#endif //  APPLICATION_DELEGATE_H
