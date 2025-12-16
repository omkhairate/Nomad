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

MTK::View *NomadPathTracer::ControllerView::get(CGRect frame) {
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
    memoryLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(10, frame.size.height - 55, 150, 20)];
    [memoryLabel setBezeled:NO];
    [memoryLabel setDrawsBackground:YES];
    [memoryLabel setBackgroundColor:[NSColor colorWithCalibratedWhite:0 alpha:0.5]];
    [memoryLabel setEditable:NO];
    [memoryLabel setSelectable:NO];
    [memoryLabel setTextColor:[NSColor whiteColor]];
    [memoryLabel setStringValue:@"GPU: 0.0 MB"];
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
    [memoryLabel setStringValue:[NSString stringWithFormat:@"GPU: %.1f MB", mem]];
}

- (id)init {
    [self becomeFirstResponder];
    return self;
}

- (BOOL)acceptsFirstResponder {
    return YES;
}

- (void)mouseDragged:(NSEvent *)event {
    NomadPathTracer::InputSystem::rotationInput.x = [event deltaX];
    NomadPathTracer::InputSystem::rotationInput.y = [event deltaY];
}

- (void)keyDown:(NSEvent *)event {
    if(event.keyCode == 2) NomadPathTracer::InputSystem::movementInput.x = 1; // right - d
    else if(event.keyCode == 0) NomadPathTracer::InputSystem::movementInput.x = -1; // left - a

    if(event.keyCode == 49) NomadPathTracer::InputSystem::movementInput.y = 1; // up - space
    else if(event.keyCode == 8) NomadPathTracer::InputSystem::movementInput.y = -1; // down - c

    if(event.keyCode == 13) NomadPathTracer::InputSystem::movementInput.z = 1; // forward - w
    else if(event.keyCode == 1) NomadPathTracer::InputSystem::movementInput.z = -1; // backward - s

    if(event.keyCode == 15) NomadPathTracer::InputSystem::resetInput = 1;

    if(event.keyCode == 17) { // t - toggle TLAS debug
        NomadPathTracer::InputSystem::debugAS = (NomadPathTracer::InputSystem::debugAS == 1) ? 0 : 1;
    }
    if(event.keyCode == 11) { // b - toggle BLAS debug
        NomadPathTracer::InputSystem::debugAS = (NomadPathTracer::InputSystem::debugAS == 2) ? 0 : 2;
    }
    if(event.keyCode == 31) { // o - toggle observer camera
        NomadPathTracer::InputSystem::observerToggleRequest = true;
    }
}

- (void)keyUp:(NSEvent *)event {
    
    if(event.keyCode == 2) NomadPathTracer::InputSystem::movementInput.x = 0; // right - d
    else if(event.keyCode == 0) NomadPathTracer::InputSystem::movementInput.x = 0; // left - a
    
    if(event.keyCode == 49) NomadPathTracer::InputSystem::movementInput.y = 0; // up - space
    else if(event.keyCode == 8) NomadPathTracer::InputSystem::movementInput.y = 0; // down - c
    
    if(event.keyCode == 13) NomadPathTracer::InputSystem::movementInput.z = 0; // forward - w
    else if(event.keyCode == 1) NomadPathTracer::InputSystem::movementInput.z = 0; // backward - s
}

-(void)scrollWheel:(NSEvent *)event {
    NomadPathTracer::InputSystem::zoomInput = -event.scrollingDeltaY;
}

@end

void NomadPathTracer::updateFPS(double fps) {
    [ViewBridge updateFPS:fps];
}

void NomadPathTracer::updateMemoryUsage(double memoryMB) {
    [ViewBridge updateMemory:memoryMB];
}
