#pragma once

// why not <AppKit/NSApplication.h>? Because Cocoa Brings in AppKit and
// Foundation
#import <Cocoa/Cocoa.h>

#include "avk-macos-application.h"
#import "avk-primary-window.h"

@interface AVKApplicationDelegate : NSObject <NSApplicationDelegate>
// We are using ARC, hence retain shouldn't be necessary. stong is default
@property(nonatomic, readonly) NSWindow* window;
@property(assign, nonatomic, readonly) avk::MacosApplication* app;
@property(nonatomic, readonly) AVKVulkanViewController* viewController;
// init methods (instancetype is more type safe than id)
- (instancetype)init NS_DESIGNATED_INITIALIZER;
// NSApplicationDelegate protocol
@end