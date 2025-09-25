#import "ControllerView.hpp"
#import <MetalKit/MetalKit.h>
#import <AppKit/AppKit.h>

#include "InputSystem.h"

@interface ViewBridge : MTKView {
}
+ (void)load:(CGRect)frame;
+ (ViewBridge *)get;
+ (void)updateFPS:(double)fps;
+ (void)updateMemory:(double)mem;
@end

ViewBridge *adapter;
NSTextField *fpsLabel;
NSTextField *memoryLabel;

MTK::View *MetalCppPathTracer::ControllerView::get(CGRect frame) {
    [ViewBridge load: frame];
    return (__bridge MTK::View *)[ViewBridge get];
}

@implementation ViewBridge

+ (void)load:(CGRect)frame {
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    adapter = [[self alloc] initWithFrame:frame];
    [adapter init];
    fpsLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(10, frame.size.height - 30, 120, 20)];
    [fpsLabel setBezeled:NO];
    [fpsLabel setDrawsBackground:YES];
    [fpsLabel setBackgroundColor:[NSColor colorWithCalibratedWhite:0 alpha:0.5]];
    [fpsLabel setEditable:NO];
    [fpsLabel setSelectable:NO];
    [fpsLabel setTextColor:[NSColor whiteColor]];
    [fpsLabel setStringValue:@"0 FPS"];
    [adapter addSubview:fpsLabel];
    memoryLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(10, frame.size.height - 55, 120, 20)];
    [memoryLabel setBezeled:NO];
    [memoryLabel setDrawsBackground:YES];
    [memoryLabel setBackgroundColor:[NSColor colorWithCalibratedWhite:0 alpha:0.5]];
    [memoryLabel setEditable:NO];
    [memoryLabel setSelectable:NO];
    [memoryLabel setTextColor:[NSColor whiteColor]];
    [memoryLabel setStringValue:@"0 MB"];
    [adapter addSubview:memoryLabel];
    [pool release];
}

+ (ViewBridge *)get {
    return adapter;
}

+ (void)updateFPS:(double)fps {
    [fpsLabel setStringValue:[NSString stringWithFormat:@"%.1f FPS", fps]];
}

+ (void)updateMemory:(double)mem {
    [memoryLabel setStringValue:[NSString stringWithFormat:@"%.1f MB", mem]];
}

- (id)init {
    [self becomeFirstResponder];
    return self;
}

- (BOOL)acceptsFirstResponder {
    return YES;
}

- (void)mouseDragged:(NSEvent *)event {
    MetalCppPathTracer::InputSystem::rotationInput.x = [event deltaX];
    MetalCppPathTracer::InputSystem::rotationInput.y = [event deltaY];
}

- (void)keyDown:(NSEvent *)event {
    if(event.keyCode == 2) MetalCppPathTracer::InputSystem::movementInput.x = 1; // right - d
    else if(event.keyCode == 0) MetalCppPathTracer::InputSystem::movementInput.x = -1; // left - a

    if(event.keyCode == 49) MetalCppPathTracer::InputSystem::movementInput.y = 1; // up - space
    else if(event.keyCode == 8) MetalCppPathTracer::InputSystem::movementInput.y = -1; // down - c

    if(event.keyCode == 13) MetalCppPathTracer::InputSystem::movementInput.z = 1; // forward - w
    else if(event.keyCode == 1) MetalCppPathTracer::InputSystem::movementInput.z = -1; // backward - s

    if(event.keyCode == 15) MetalCppPathTracer::InputSystem::resetInput = 1;

    if(event.keyCode == 17) { // t - toggle TLAS debug
        MetalCppPathTracer::InputSystem::debugAS = (MetalCppPathTracer::InputSystem::debugAS == 1) ? 0 : 1;
    }
    if(event.keyCode == 11) { // b - toggle BLAS debug
        MetalCppPathTracer::InputSystem::debugAS = (MetalCppPathTracer::InputSystem::debugAS == 2) ? 0 : 2;
    }
}

- (void)keyUp:(NSEvent *)event {
    
    if(event.keyCode == 2) MetalCppPathTracer::InputSystem::movementInput.x = 0; // right - d
    else if(event.keyCode == 0) MetalCppPathTracer::InputSystem::movementInput.x = 0; // left - a
    
    if(event.keyCode == 49) MetalCppPathTracer::InputSystem::movementInput.y = 0; // up - space
    else if(event.keyCode == 8) MetalCppPathTracer::InputSystem::movementInput.y = 0; // down - c
    
    if(event.keyCode == 13) MetalCppPathTracer::InputSystem::movementInput.z = 0; // forward - w
    else if(event.keyCode == 1) MetalCppPathTracer::InputSystem::movementInput.z = 0; // backward - s
}

-(void)scrollWheel:(NSEvent *)event {
    MetalCppPathTracer::InputSystem::zoomInput = -event.scrollingDeltaY;
}

@end

void MetalCppPathTracer::updateFPS(double fps) {
    [ViewBridge updateFPS:fps];
}

void MetalCppPathTracer::updateMemoryUsage(double memoryMB) {
    [ViewBridge updateMemory:memoryMB];
}
