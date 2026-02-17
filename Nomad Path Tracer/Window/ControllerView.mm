#import "ControllerView.hpp"
#import <MetalKit/MetalKit.h>
#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>

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
 + (void)minSamplesChanged:(NSSlider *)sender;
 + (void)maxSamplesChanged:(NSSlider *)sender;
 + (void)residencyStrategyChanged:(NSPopUpButton *)sender;
 + (void)openScenePressed:(id)sender;
 + (void)reloadScenePressed:(id)sender;
 + (void)promptForSceneIfNeeded;
 + (void)updateSceneStatus;
@end

static NomadPathTracer::ViewDelegate *renderDelegate = nullptr;

static NSSlider *refreshRateSlider;
static NSTextField *refreshRateLabel;
static NSSlider *maxRayDepthSlider;
static NSTextField *maxRayDepthLabel;
static NSSlider *minSamplesSlider;
static NSTextField *minSamplesLabel;
static NSSlider *maxSamplesSlider;
static NSTextField *maxSamplesLabel;
static NSPopUpButton *strategyPopup;
static NSTextField *scenePathLabel;
static bool didPromptForScene = false;

static NSArray<NSString *> *residencyStrategyNames() {
    return @[@"Distance LOD", @"Energy", @"Ray-hit", @"Screen-space", @"Probabilistic", @"Always Resident", @"Environment", @"Predictive Env", @"Unified"];
}

static NomadPathTracer::ResidencyStrategy strategyFromIndex(NSInteger idx) {
    switch (idx) {
        case 1: return NomadPathTracer::ResidencyStrategy::EnergyImportance;
        case 2: return NomadPathTracer::ResidencyStrategy::RayHitBudget;
        case 3: return NomadPathTracer::ResidencyStrategy::ScreenSpaceFootprint;
        case 4: return NomadPathTracer::ResidencyStrategy::Probabilistic;
        case 5: return NomadPathTracer::ResidencyStrategy::AlwaysResident;
        case 6: return NomadPathTracer::ResidencyStrategy::EnvironmentHit;
        case 7: return NomadPathTracer::ResidencyStrategy::PredictiveEnvironment;
        case 8: return NomadPathTracer::ResidencyStrategy::UnifiedScore;
        default: return NomadPathTracer::ResidencyStrategy::DistanceLOD;
    }
}

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

static void updateMinSamplesLabel(uint32_t value) {
    if (minSamplesLabel) {
        [minSamplesLabel setStringValue:[NSString stringWithFormat:@"Min Samples: %u", value]];
    }
}

static void updateMaxSamplesLabel(uint32_t value) {
    if (maxSamplesLabel) {
        [maxSamplesLabel setStringValue:[NSString stringWithFormat:@"Max Samples: %u", value]];
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

    NSView *controlPanel = [[NSView alloc] initWithFrame:NSMakeRect(10, 10, 300, 320)];
    [controlPanel setWantsLayer:YES];
    controlPanel.layer.backgroundColor = [[NSColor colorWithCalibratedWhite:0 alpha:0.5] CGColor];
    controlPanel.layer.cornerRadius = 6.0;

    NSButton *openButton = [[NSButton alloc] initWithFrame:NSMakeRect(10, 284, 120, 26)];
    [openButton setTitle:@"Open Scene…"];
    [openButton setTarget:self];
    [openButton setAction:@selector(openScenePressed:)];
    [controlPanel addSubview:openButton];

    NSButton *reloadButton = [[NSButton alloc] initWithFrame:NSMakeRect(170, 284, 120, 26)];
    [reloadButton setTitle:@"Run / Reload"];
    [reloadButton setTarget:self];
    [reloadButton setAction:@selector(reloadScenePressed:)];
    [controlPanel addSubview:reloadButton];

    scenePathLabel = createLabel(NSMakeRect(10, 260, 280, 18), @"No scene loaded");
    [controlPanel addSubview:scenePathLabel];

    refreshRateLabel = createLabel(NSMakeRect(10, 234, 220, 18), @"Refresh: 60 FPS");
    [controlPanel addSubview:refreshRateLabel];
    refreshRateSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(10, 214, 280, 20)];
    [refreshRateSlider setMinValue:24];
    [refreshRateSlider setMaxValue:240];
    [refreshRateSlider setIntegerValue:60];
    [refreshRateSlider setTarget:self];
    [refreshRateSlider setAction:@selector(refreshRateChanged:)];
    [controlPanel addSubview:refreshRateSlider];

    maxRayDepthLabel = createLabel(NSMakeRect(10, 188, 220, 18), @"Max Ray Depth: 8");
    [controlPanel addSubview:maxRayDepthLabel];
    maxRayDepthSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(10, 168, 280, 20)];
    [maxRayDepthSlider setMinValue:1];
    [maxRayDepthSlider setMaxValue:64];
    [maxRayDepthSlider setIntegerValue:8];
    [maxRayDepthSlider setTarget:self];
    [maxRayDepthSlider setAction:@selector(maxRayDepthChanged:)];
    [controlPanel addSubview:maxRayDepthSlider];

    minSamplesLabel = createLabel(NSMakeRect(10, 142, 220, 18), @"Min Samples: 1");
    [controlPanel addSubview:minSamplesLabel];
    minSamplesSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(10, 122, 280, 20)];
    [minSamplesSlider setMinValue:1];
    [minSamplesSlider setMaxValue:16];
    [minSamplesSlider setIntegerValue:1];
    [minSamplesSlider setTarget:self];
    [minSamplesSlider setAction:@selector(minSamplesChanged:)];
    [controlPanel addSubview:minSamplesSlider];

    maxSamplesLabel = createLabel(NSMakeRect(10, 96, 220, 18), @"Max Samples: 4");
    [controlPanel addSubview:maxSamplesLabel];
    maxSamplesSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(10, 76, 280, 20)];
    [maxSamplesSlider setMinValue:1];
    [maxSamplesSlider setMaxValue:32];
    [maxSamplesSlider setIntegerValue:4];
    [maxSamplesSlider setTarget:self];
    [maxSamplesSlider setAction:@selector(maxSamplesChanged:)];
    [controlPanel addSubview:maxSamplesSlider];

    strategyPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(10, 34, 280, 28) pullsDown:NO];
    [strategyPopup addItemsWithTitles:residencyStrategyNames()];
    [strategyPopup setTarget:self];
    [strategyPopup setAction:@selector(residencyStrategyChanged:)];
    [controlPanel addSubview:strategyPopup];

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
    [self promptForSceneIfNeeded];
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

        uint32_t minSamples = renderDelegate->minSamplesPerPixel();
        uint32_t maxSamples = renderDelegate->maxSamplesPerPixel();
        [minSamplesSlider setIntegerValue:static_cast<NSInteger>(minSamples)];
        [maxSamplesSlider setIntegerValue:static_cast<NSInteger>(maxSamples)];
        updateMinSamplesLabel(minSamples);
        updateMaxSamplesLabel(maxSamples);

        auto strategy = renderDelegate->residencyStrategy();
        [strategyPopup selectItemAtIndex:static_cast<NSInteger>(strategy)];

        [self updateSceneStatus];
    }
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

+ (void)minSamplesChanged:(NSSlider *)sender {
    if (!renderDelegate)
        return;
    uint32_t minSamples = static_cast<uint32_t>(sender.integerValue);
    uint32_t maxSamples = renderDelegate->maxSamplesPerPixel();
    if (minSamples > maxSamples) {
        maxSamples = minSamples;
        [maxSamplesSlider setIntegerValue:static_cast<NSInteger>(maxSamples)];
    }
    renderDelegate->setSamplesPerPixel(minSamples, maxSamples);
    updateMinSamplesLabel(minSamples);
    updateMaxSamplesLabel(maxSamples);
}

+ (void)maxSamplesChanged:(NSSlider *)sender {
    if (!renderDelegate)
        return;
    uint32_t maxSamples = static_cast<uint32_t>(sender.integerValue);
    uint32_t minSamples = renderDelegate->minSamplesPerPixel();
    if (maxSamples < minSamples) {
        minSamples = maxSamples;
        [minSamplesSlider setIntegerValue:static_cast<NSInteger>(minSamples)];
    }
    renderDelegate->setSamplesPerPixel(minSamples, maxSamples);
    updateMinSamplesLabel(minSamples);
    updateMaxSamplesLabel(maxSamples);
}

+ (void)residencyStrategyChanged:(NSPopUpButton *)sender {
    if (!renderDelegate)
        return;
    renderDelegate->setResidencyStrategy(strategyFromIndex(sender.indexOfSelectedItem));
}

+ (void)openScenePressed:(id)sender {
    if (!renderDelegate)
        return;
    NSOpenPanel *panel = [NSOpenPanel openPanel];
    [panel setAllowsMultipleSelection:NO];
    [panel setCanChooseDirectories:NO];
    [panel setAllowedFileTypes:@[@"xml", @"obj", @"gltf", @"glb"]];
    if ([panel runModal] == NSModalResponseOK) {
        NSURL *url = panel.URL;
        if (url.path) {
            bool loaded = renderDelegate->loadScene(std::string(url.path.UTF8String));
            if (!loaded) {
                NSAlert *alert = [[NSAlert alloc] init];
                [alert setMessageText:@"Failed to load scene"];
                [alert setInformativeText:@"The selected scene could not be loaded."];
                [alert runModal];
            }
            [self updateControlValues];
        }
    }
}

+ (void)reloadScenePressed:(id)sender {
    if (!renderDelegate)
        return;
    if (!renderDelegate->hasSceneLoaded() || renderDelegate->scenePath().empty()) {
        NSAlert *alert = [[NSAlert alloc] init];
        [alert setMessageText:@"No scene loaded"];
        [alert setInformativeText:@"Choose a scene with Open Scene… before running reload."];
        [alert runModal];
        return;
    }

    bool loaded = renderDelegate->reloadScene();
    if (!loaded) {
        NSAlert *alert = [[NSAlert alloc] init];
        [alert setMessageText:@"Reload failed"];
        [alert setInformativeText:@"The current scene could not be reloaded."];
        [alert runModal];
    }
    [self updateControlValues];
}


+ (void)updateSceneStatus {
    if (!scenePathLabel)
        return;
    if (!renderDelegate || !renderDelegate->hasSceneLoaded() || renderDelegate->scenePath().empty()) {
        [scenePathLabel setStringValue:@"No scene loaded"];
        return;
    }

    std::string scenePath = renderDelegate->scenePath();
    [scenePathLabel setStringValue:[NSString stringWithFormat:@"Scene: %s", scenePath.c_str()]];
}

+ (void)promptForSceneIfNeeded {
    if (didPromptForScene || !renderDelegate)
        return;
    if (renderDelegate->hasSceneLoaded() || !renderDelegate->scenePath().empty())
        return;
    didPromptForScene = true;
    [self openScenePressed:nil];
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
