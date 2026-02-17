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
  NomadPathTracer::ViewDelegate *_viewDelegate;
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

  _viewDelegate = new NomadPathTracer::ViewDelegate(device);
  viewFactory.setViewDelegate(_viewDelegate);
  cppView->setDelegate(_viewDelegate);
  cppView->setPaused(false);
  return self;
}

- (void)dealloc {
  [_mtkView setDelegate:nil];
  if (_viewDelegate) {
    delete _viewDelegate;
    _viewDelegate = nullptr;
  }
  [super dealloc];
}

- (void)loadView {
  self.view = _mtkView;
}

@end

NS::ViewController *NomadPathTracer::RendererViewController::get(
    CGRect frame, MTL::Device *device) {
  RendererViewControllerBridge *controller =
      [[RendererViewControllerBridge alloc] initWithFrame:frame device:device];
  return (__bridge NS::ViewController *)controller;
}
