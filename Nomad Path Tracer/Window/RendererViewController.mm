#import "RendererViewController.hpp"

#import <AppKit/AppKit.h>
#import <MetalKit/MetalKit.h>

#include "ControllerView.hpp"
#include "ViewDelegate.h"

@interface RendererViewControllerBridge : NSViewController
- (instancetype)initWithFrame:(CGRect)frame device:(MTL::Device *)device;
@end

@implementation RendererViewControllerBridge {
  MTKView *_mtkView;
}

- (instancetype)initWithFrame:(CGRect)frame device:(MTL::Device *)device {
  self = [super initWithNibName:nil bundle:nil];
  if (!self)
    return nil;

  NomadPathTracer::ControllerView viewFactory;
  _mtkView = (__bridge MTKView *)viewFactory.get(frame);
  MTK::View *cppView = (__bridge MTK::View *)_mtkView;
  cppView->setDevice(device);
  cppView->setColorPixelFormat(MTL::PixelFormat::PixelFormatRGBA16Float);
  cppView->setClearColor(MTL::ClearColor::Make(0.0, 0.0, 0.0, 1.0));
  cppView->setPreferredFramesPerSecond(60);
  cppView->setEnableSetNeedsDisplay(false);

  viewFactory.setViewDelegate(nullptr);
  cppView->setDelegate(nullptr);
  cppView->setPaused(false);
  return self;
}

- (void)dealloc {
  [_mtkView setDelegate:nil];
  [super dealloc];
}

- (void)loadView {
  self.view = _mtkView;
}

@end

NomadPathTracer::RendererViewController::~RendererViewController() {
  if (_viewDelegate) {
    delete _viewDelegate;
    _viewDelegate = nullptr;
  }
}

NS::View *NomadPathTracer::RendererViewController::get(
    CGRect frame, MTL::Device *device) {
  RendererViewControllerBridge *controller =
      [[RendererViewControllerBridge alloc] initWithFrame:frame device:device];

  MTKView *mtkView = [controller view];
  MTK::View *cppView = (__bridge MTK::View *)mtkView;

  if (_viewDelegate) {
    delete _viewDelegate;
  }
  _viewDelegate = new NomadPathTracer::ViewDelegate(device);

  NomadPathTracer::ControllerView viewFactory;
  viewFactory.setViewDelegate(_viewDelegate);
  cppView->setDelegate(_viewDelegate);

  NS::View *contentView = (__bridge NS::View *)mtkView;
  [mtkView retain];
  [controller release];
  return contentView;
}
