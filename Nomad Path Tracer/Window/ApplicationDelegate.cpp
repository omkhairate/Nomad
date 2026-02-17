#include "ApplicationDelegate.h"

using namespace NomadPathTracer;

ApplicationDelegate::~ApplicationDelegate() {
  if (_pRendererView)
    _pRendererView->release();
  if (_pWindow)
    _pWindow->release();
  if (_pDevice)
    _pDevice->release();
}

void ApplicationDelegate::applicationWillFinishLaunching(
    NS::Notification *pNotification) {
  NS::Application *pApp =
      reinterpret_cast<NS::Application *>(pNotification->object());
  pApp->setActivationPolicy(NS::ActivationPolicy::ActivationPolicyRegular);
}

void ApplicationDelegate::applicationDidFinishLaunching(
    NS::Notification *pNotification) {
  if (_initialized)
    return;
  _initialized = true;

  constexpr float kDefaultWidth = 1280.0f;
  constexpr float kDefaultHeight = 720.0f;
  CGRect frame = {{100.0, 100.0}, {kDefaultWidth, kDefaultHeight}};

  _pWindow = NS::Window::alloc()->init(frame,
                                       NS::WindowStyleMaskClosable |
                                           NS::WindowStyleMaskTitled |
                                           NS::WindowStyleMaskResizable,
                                       NS::BackingStoreBuffered, false);

  _pDevice = MTL::CreateSystemDefaultDevice();

  _pRendererView = _rendererViewController.get(frame, _pDevice);
  _pWindow->setContentView(_pRendererView);
  // Window title updated to reflect the new application name.
  _pWindow->setTitle(
      NS::String::string("Nomad", NS::StringEncoding::UTF8StringEncoding));
  _pWindow->makeKeyAndOrderFront(nullptr);

  NS::Application *pApp =
      reinterpret_cast<NS::Application *>(pNotification->object());
  pApp->activateIgnoringOtherApps(true);
}

bool ApplicationDelegate::applicationShouldTerminateAfterLastWindowClosed(
    NS::Application * /*pSender*/) {
  return true;
}

// hehehehe
