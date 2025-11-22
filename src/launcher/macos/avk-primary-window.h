#pragma once

// Frameworks
#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>

#import "avk-macos-application.h"

// --------------------------------------------------------------------------

@protocol AVKVulkanMetalViewDelegate;
@interface AVKVulkanMetalView : NSView
@property(nonatomic, weak) id<AVKVulkanMetalViewDelegate> delegate;
@property(nonatomic, readonly) CAMetalLayer* metalLayer;
@end

// --------------------------------------------------------------------------

@interface AVKVulkanViewController : NSViewController
@property(nonatomic, readonly) AVKVulkanMetalView* metalView;
@property(nonatomic, readonly) id<AVKVulkanMetalViewDelegate> bridge;

- (instancetype)initWithApp:(avk::MacosApplication*)app andFrame:(NSRect)frame;
@end

// --------------------------------------------------------------------------

// https://developer.apple.com/documentation/metal/achieving-smooth-frame-rates-with-a-metal-display-link?language=objc
@protocol AVKVulkanMetalViewDelegate <NSObject>
@optional
// Lifecycle
- (void)metalViewDidMoveToWindow:(AVKVulkanMetalView*)view;
- (void)metalView:(AVKVulkanMetalView*)view drawableSizeDidChange:(CGSize)size;
- (void)metalViewWillStartLiveResize:(AVKVulkanMetalView*)view;
- (void)metalViewDidEndLiveResize:(AVKVulkanMetalView*)view;

// Focus
- (void)viewBecameFirstResponder:(AVKVulkanMetalView*)view;
- (void)viewResignedFirstResponder:(AVKVulkanMetalView*)view;

// Rendering
- (void)metalViewDrawRequest:(AVKVulkanMetalView*)view;

// Input
- (void)metalView:(AVKVulkanMetalView*)view keyDown:(NSEvent*)event;
- (void)metalView:(AVKVulkanMetalView*)view keyUp:(NSEvent*)event;
- (void)metalView:(AVKVulkanMetalView*)view mouseMoved:(NSEvent*)event;
- (void)metalView:(AVKVulkanMetalView*)view mouseDown:(NSEvent*)event;
- (void)metalView:(AVKVulkanMetalView*)view mouseUp:(NSEvent*)event;
@end