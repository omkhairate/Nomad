#include "ApplicationDelegate.h"
#include "SceneLoader.h"
#include <filesystem>

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

  Scene tmpScene;
  bool loaded = SceneLoader::LoadSceneFromXML("scene.xml", &tmpScene);
  if (!loaded) {
    std::filesystem::path alt =
        std::filesystem::path(__FILE__).parent_path() / "../scene.xml";
    SceneLoader::LoadSceneFromXML(alt.string(), &tmpScene);
  }

  CGRect frame = {{100.0, 100.0}, {tmpScene.screenSize.x, tmpScene.screenSize.y}};

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
