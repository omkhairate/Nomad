#import "ControllerView.hpp"
#import <MetalKit/MetalKit.h>
#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>
#include <algorithm>

#include "InputSystem.h"
#include "ViewDelegate.h"

@interface ViewBridge : MTKView {
}
+ (void)load:(CGRect)frame;
+ (ViewBridge *)get;
+ (void)updateFPS:(double)fps;
+ (void)updateMemory:(double)mem;
+ (void)setDelegate:(NomadPathTracer::ViewDelegate *)delegate;
+ (void)updateControlValues;
+ (void)refreshRateChanged:(NSSlider *)sender;
 + (void)maxRayDepthChanged:(NSSlider *)sender;
 + (void)cameraPositionChanged:(NSSlider *)sender;
 + (void)updateCameraControlValues;
@end

static NomadPathTracer::ViewDelegate *renderDelegate = nullptr;

static NSSlider *refreshRateSlider;
static NSTextField *refreshRateLabel;
static NSSlider *maxRayDepthSlider;
static NSTextField *maxRayDepthLabel;
static NSSlider *cameraPositionSliders[3];
static NSTextField *cameraPositionLabels[3];

static constexpr float kCameraMinPosition = -1000.0f;
static constexpr float kCameraMaxPosition = 1000.0f;

static NSTextField *createLabel(NSRect frame, NSString *text) {
    NSTextField *label = [[NSTextField alloc] initWithFrame:frame];
    [label setBezeled:NO];
    [label setDrawsBackground:NO];
    [label setEditable:NO];
    [label setSelectable:NO];
    [label setTextColor:[NSColor whiteColor]];
    [label setFont:[NSFont systemFontOfSize:12.0]];
    [label setStringValue:text];
    return label;
}

static void updateRefreshLabel(NSInteger fps) {
    if (refreshRateLabel) {
        [refreshRateLabel setStringValue:[NSString stringWithFormat:@"Refresh: %ld FPS", (long)fps]];
    }
}

static void updateRayDepthLabel(uint32_t depth) {
    if (maxRayDepthLabel) {
        [maxRayDepthLabel setStringValue:[NSString stringWithFormat:@"Max Ray Depth: %u", depth]];
    }
}

static float clampCameraPosition(float value) {
    return std::clamp(value, kCameraMinPosition, kCameraMaxPosition);
}

static void updateCameraLabel(NSInteger axis, float value) {
    if (axis < 0 || axis > 2)
        return;
    static const char *axisNames[] = {"X", "Y", "Z"};
    if (cameraPositionLabels[axis]) {
        [cameraPositionLabels[axis]
            setStringValue:[NSString stringWithFormat:@"Camera %s: %.1f",
                                                     axisNames[axis], value]];
    }
}

ViewBridge *adapter;
NSTextField *fpsLabel;
NSTextField *memoryLabel;

MTK::View *NomadPathTracer::ControllerView::get(CGRect frame) {
    [ViewBridge load: frame];
    return (__bridge MTK::View *)[ViewBridge get];
}

void NomadPathTracer::ControllerView::setViewDelegate(ViewDelegate *delegate) {
    [ViewBridge setDelegate:delegate];
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

    NSView *controlPanel = [[NSView alloc] initWithFrame:NSMakeRect(10, 10, 240, 240)];
    [controlPanel setWantsLayer:YES];
    controlPanel.layer.backgroundColor = [[NSColor colorWithCalibratedWhite:0 alpha:0.5] CGColor];
    controlPanel.layer.cornerRadius = 6.0;

    CGFloat labelWidth = 200;
    CGFloat sliderWidth = 200;
    CGFloat xOffset = 10;
    CGFloat labelHeight = 18;
    CGFloat sliderHeight = 20;
    CGFloat yOffset = 210;

    static const char *axisNames[] = {"X", "Y", "Z"};
    for (NSInteger axis = 0; axis < 3; ++axis) {
        cameraPositionLabels[axis] = createLabel(
            NSMakeRect(xOffset, yOffset, labelWidth, labelHeight),
            [NSString stringWithFormat:@"Camera %s: 0.0", axisNames[axis]]);
        [controlPanel addSubview:cameraPositionLabels[axis]];

        cameraPositionSliders[axis] = [[NSSlider alloc] initWithFrame:
                                           NSMakeRect(xOffset, yOffset - 22, sliderWidth, sliderHeight)];
        [cameraPositionSliders[axis] setMinValue:kCameraMinPosition];
        [cameraPositionSliders[axis] setMaxValue:kCameraMaxPosition];
        [cameraPositionSliders[axis] setFloatValue:0.0f];
        [cameraPositionSliders[axis] setTag:axis];
        [cameraPositionSliders[axis] setTarget:self];
        [cameraPositionSliders[axis] setAction:@selector(cameraPositionChanged:)];
        [controlPanel addSubview:cameraPositionSliders[axis]];

        yOffset -= 48;
    }

    refreshRateLabel = createLabel(NSMakeRect(xOffset, yOffset, labelWidth, labelHeight), @"Refresh: 60 FPS");
    [controlPanel addSubview:refreshRateLabel];
    refreshRateSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(xOffset, yOffset - 22, sliderWidth, sliderHeight)];
    [refreshRateSlider setMinValue:24];
    [refreshRateSlider setMaxValue:240];
    [refreshRateSlider setIntegerValue:60];
    [refreshRateSlider setTarget:self];
    [refreshRateSlider setAction:@selector(refreshRateChanged:)];
    [controlPanel addSubview:refreshRateSlider];

    maxRayDepthLabel = createLabel(NSMakeRect(xOffset, yOffset - 48, labelWidth, labelHeight), @"Max Ray Depth: 8");
    [controlPanel addSubview:maxRayDepthLabel];
    maxRayDepthSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(xOffset, yOffset - 70, sliderWidth, sliderHeight)];
    [maxRayDepthSlider setMinValue:1];
    [maxRayDepthSlider setMaxValue:64];
    [maxRayDepthSlider setIntegerValue:8];
    [maxRayDepthSlider setTarget:self];
    [maxRayDepthSlider setAction:@selector(maxRayDepthChanged:)];
    [controlPanel addSubview:maxRayDepthSlider];

    [adapter addSubview:controlPanel];
    [self updateControlValues];
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

+ (void)setDelegate:(NomadPathTracer::ViewDelegate *)delegate {
    renderDelegate = delegate;
    [self updateControlValues];
}

+ (void)updateControlValues {
    if (adapter && refreshRateSlider) {
        NSInteger fps = adapter.preferredFramesPerSecond;
        [refreshRateSlider setIntegerValue:fps];
        updateRefreshLabel(fps);
    }
    if (renderDelegate && maxRayDepthSlider) {
        uint32_t depth = renderDelegate->maxRayDepth();
        [maxRayDepthSlider setIntegerValue:static_cast<NSInteger>(depth)];
        updateRayDepthLabel(depth);
    }
    [self updateCameraControlValues];
}

+ (void)refreshRateChanged:(NSSlider *)sender {
    NSInteger fps = sender.integerValue;
    updateRefreshLabel(fps);
    if (adapter) {
        adapter.preferredFramesPerSecond = fps;
    }
}

+ (void)maxRayDepthChanged:(NSSlider *)sender {
    uint32_t depth = static_cast<uint32_t>(sender.integerValue);
    updateRayDepthLabel(depth);
    if (renderDelegate) {
        renderDelegate->setMaxRayDepth(depth);
    }
}

+ (void)updateCameraControlValues {
    bool cameraControlsEnabled = renderDelegate && !renderDelegate->hasCameraKeyframes();

    simd::float3 position = {0.0f, 0.0f, 0.0f};
    if (cameraControlsEnabled) {
        position = renderDelegate->activeCameraPosition();
    }

    for (NSInteger axis = 0; axis < 3; ++axis) {
        if (cameraPositionLabels[axis])
            [cameraPositionLabels[axis] setHidden:!cameraControlsEnabled];
        if (cameraPositionSliders[axis]) {
            [cameraPositionSliders[axis] setHidden:!cameraControlsEnabled];
            [cameraPositionSliders[axis] setEnabled:cameraControlsEnabled];
            if (cameraControlsEnabled) {
                float clampedValue = clampCameraPosition(position[axis]);
                [cameraPositionSliders[axis] setFloatValue:clampedValue];
                updateCameraLabel(axis, clampedValue);
            }
        }
    }
}

+ (void)cameraPositionChanged:(NSSlider *)sender {
    if (!renderDelegate || renderDelegate->hasCameraKeyframes())
        return;

    NSInteger axis = sender.tag;
    if (axis < 0 || axis > 2)
        return;

    simd::float3 position = renderDelegate->activeCameraPosition();
    float clampedValue = clampCameraPosition(sender.floatValue);
    position[axis] = clampedValue;
    renderDelegate->setCameraPosition(position);
    updateCameraLabel(axis, clampedValue);
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
