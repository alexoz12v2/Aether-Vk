#import "avk-primary-window.h"

// framework
#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/CADisplayLink.h>
#import <QuartzCore/CAMetalLayer.h>

// --------------------------------------------------------------------------

@interface AVKVulkanMetalView () {
  CADisplayLink *_displayLink;
  NSTrackingArea *ivar_trackingArea;
}
@end
@implementation AVKVulkanMetalView
+ (Class)layerClass {
  return [CAMetalLayer class];
}
+ (BOOL)wantsUpdateLayer {
  return YES;
}

// utility to take the current NSView size and propagate it to our delegate
- (void)updateDrawableSize {
  NSSize backingSize = [self convertSizeToBacking:self.bounds.size];
  _metalLayer.drawableSize = CGSizeMake(backingSize.width, backingSize.height);

  if ([self.delegate respondsToSelector:@selector(metalView:
                                            drawableSizeDidChange:)]) {
    [self.delegate metalView:self
        drawableSizeDidChange:_metalLayer.drawableSize];
  }
}

- (void)viewWillStartLiveResize {
  if ([self.delegate
          respondsToSelector:@selector(metalViewWillStartLiveResize:)]) {
    [self.delegate metalViewWillStartLiveResize:self];
  }
}

- (void)viewDidEndLiveResize {
  if ([self.delegate
          respondsToSelector:@selector(metalViewDidEndLiveResize:)]) {
    [self.delegate metalViewDidEndLiveResize:self];
  }
}

// called in init
- (void)setupTrackingArea {
  if (ivar_trackingArea) [self removeTrackingArea:ivar_trackingArea];
  ivar_trackingArea = [[NSTrackingArea alloc]
      initWithRect:self.bounds
           options:NSTrackingMouseMoved | NSTrackingActiveAlways |
                   NSTrackingInVisibleRect
             owner:self
          userInfo:nil];
}

// called by controller, needed to setup callback to display link
- (void)startDisplayLink {
  // macOS 15+ -> NSView displayLink, not the one from CoreVideo
  if (_displayLink) return;
  __weak decltype(self) weakSelf = self;
  // https://developer.apple.com/documentation/quartzcore/cadisplaylink/init(target:selector:)?language=objc
  _displayLink = [self displayLinkWithTarget:weakSelf
                                    selector:@selector(displayLinkDidFire:)];
  // start it
  [_displayLink addToRunLoop:[NSRunLoop currentRunLoop]
                     forMode:NSDefaultRunLoopMode];
}

- (void)displayLinkDidFire:(CADisplayLink *)displayLink {
  if ([self.delegate respondsToSelector:@selector(metalViewDrawRequest:)]) {
    [self.delegate metalViewDrawRequest:self];
  }
}

- (void)stopDisplayLink {
  if (!_displayLink) return;
  [_displayLink invalidate];
  _displayLink = nil;
}

// ---- Override from NSView ----
- (BOOL)acceptsFirstResponder {
  return YES;
}

- (BOOL)becomeFirstResponder {
  if ([self.delegate respondsToSelector:@selector(viewBecameFirstResponder:)]) {
    [self.delegate viewBecameFirstResponder:self];
  }
  return YES;
}
- (BOOL)resignFirstResponder {
  if ([self.delegate
          respondsToSelector:@selector(viewResignedFirstResponder:)]) {
    [self.delegate viewResignedFirstResponder:self];
  }
  return YES;
}

- (CALayer *)makeBackingLayer {
  return [CAMetalLayer layer];
}

- (instancetype)initWithFrame:(NSRect)frame {
  self = [super initWithFrame:frame];
  if (self) {
    self.wantsLayer = YES;
    _metalLayer = (CAMetalLayer *)self.layer;  // TODO see
    // configure layer
    _metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;  // TODO better?
    _metalLayer.colorspace = CGColorSpaceCreateDeviceRGB();
    _metalLayer.framebufferOnly = NO;

    // initial size
    [self updateDrawableSize];
    [self setupTrackingArea];
  }
  return self;
}

- (void)setFrameSize:(NSSize)newSize {
  [super setFrameSize:newSize];
  [self updateDrawableSize];
}

- (void)viewDidMoveToWindow {
  [super viewDidMoveToWindow];
  if ([self.delegate respondsToSelector:@selector(metalViewDidMoveToWindow:)]) {
    [self.delegate metalViewDidMoveToWindow:self];
  }
}
// ---- Override from NSView: Input Events ----
- (void)keyDown:(NSEvent *)event {
  if ([self.delegate respondsToSelector:@selector(metalView:keyDown:)]) {
    [self.delegate metalView:self keyDown:event];
  }
}
- (void)keyUp:(NSEvent *)event {
  if ([self.delegate respondsToSelector:@selector(metalView:keyUp:)]) {
    [self.delegate metalView:self keyUp:event];
  }
}
- (void)mouseMoved:(NSEvent *)event {
  if ([self.delegate respondsToSelector:@selector(metalView:mouseMoved:)]) {
    [self.delegate metalView:self mouseMoved:event];
  }
}
- (void)mouseDown:(NSEvent *)event {
  if ([self.delegate respondsToSelector:@selector(metalView:mouseDown:)]) {
    [self.delegate metalView:self mouseDown:event];
  }
}
- (void)mouseUp:(NSEvent *)event {
  if ([self.delegate respondsToSelector:@selector(metalView:mouseUp:)]) {
    [self.delegate metalView:self mouseUp:event];
  }
}

@end

// --------------------------------------------------------------------------
@interface AVKVulkanAppBridge : NSObject <AVKVulkanMetalViewDelegate>
@property(nonatomic, readonly) avk::MacosApplication *app;
- (instancetype)initWithApp:(avk::MacosApplication *)app;
@end

@implementation AVKVulkanAppBridge
- (instancetype)initWithApp:(avk::MacosApplication *)app {
  self = [super init];
  if (self) {
    _app = app;
    NSAssert(app, @"App shouldn't be nullptr");
  }
  return self;
}
// Lifecycle
- (void)metalViewDidMoveToWindow:(AVKVulkanMetalView *)view {
  NSLog(@"metalViewDidMoveToWindow Called");
}
- (void)metalView:(AVKVulkanMetalView *)view
    drawableSizeDidChange:(CGSize)size {
  // this is detected by render thread
  // _app->onResize();
}

- (void)metalViewWillStartLiveResize:(AVKVulkanMetalView *)view {
  _app->onEnterResize();
}

- (void)metalViewDidEndLiveResize:(AVKVulkanMetalView *)view {
  _app->onExitResize();
}

// Focus
- (void)viewBecameFirstResponder:(AVKVulkanMetalView *)view {
  _app->resumeRendering();
}

- (void)viewResignedFirstResponder:(AVKVulkanMetalView *)view {
  _app->pauseRendering();
}

// Rendering
- (void)metalViewDrawRequest:(AVKVulkanMetalView *)view {
  // TODO signaling better
  _app->signalDisplayLinkReady();
}

// Input
- (void)metalView:(AVKVulkanMetalView *)view keyDown:(NSEvent *)event {
  NSLog(@"TODO: Implement keyDown");
}
- (void)metalView:(AVKVulkanMetalView *)view keyUp:(NSEvent *)event {
  NSLog(@"TODO: Implement keyUp");
}
- (void)metalView:(AVKVulkanMetalView *)view mouseMoved:(NSEvent *)event {
  NSLog(@"TODO: Implement mouseMoved");
}
- (void)metalView:(AVKVulkanMetalView *)view mouseDown:(NSEvent *)event {
  NSLog(@"TODO: Implement mouseDown");
}
- (void)metalView:(AVKVulkanMetalView *)view mouseUp:(NSEvent *)event {
  NSLog(@"TODO: Implement keyDowmouseUp");
}
@end

// --------------------------------------------------------------------------

// TODO refector for all pthread platforms
static void *renderThreadFunc(void *arg) {
  auto *app = reinterpret_cast<avk::MacosApplication *>(arg);
  avk::ApplicationBase::RTmain(app);
  pthread_exit(nullptr);
  return nullptr;
}

static void *updateThreadFunc(void *arg) {
  auto *app = reinterpret_cast<avk::MacosApplication *>(arg);
  avk::ApplicationBase::UTmain(app);
  pthread_exit(nullptr);
  return nullptr;
}

// TODO move to posix path
static pthread_t createThreadOrExit(void *(*proc)(void *),
                                    void *__restrict arg) {
  pthread_t thread = 0;
  if (pthread_create(&thread, nullptr, proc, arg)) {
    NSLog(@"Failed to create thread!");
    avk::showErrorScreenAndExit("Failed to create Thread");
  } else {
    NSLog(@"thread created");
  }
  return thread;
}

@implementation AVKVulkanViewController {
  NSRect ivar_frame;
  avk::MacosApplication *_app;
}
- (instancetype)initWithApp:(avk::MacosApplication *)app
                   andFrame:(NSRect)frame {
  NSAssert(app, @"App shouldn't be nullptr");
  self = [super init];
  if (self) {
    _app = app;
    _bridge = [[AVKVulkanAppBridge alloc] initWithApp:app];
    _metalView = nil;  // initialized by controller at loadView
    ivar_frame = frame;
  }
  return self;
}

// ------------------ Override from NSViewController ------------------------

// From:
// https://developer.apple.com/documentation/uikit/uiviewcontroller/loadview()?language=objc
// You should never call this method directly. The view controller calls this
// method when its view property is requested but is currently nil. This method
// loads or creates a view and assigns it to the view property.
//
// If you want to perform any additional initialization of your views, do so in
// the viewDidLoad method.
- (void)loadView {
  _metalView = [[AVKVulkanMetalView alloc] initWithFrame:ivar_frame];
  _metalView.delegate = _bridge;
  self.view = _metalView;

  [_metalView startDisplayLink];
}

// normal behaviour + register first responder for events as our view
- (void)viewDidLoad {
  [super viewDidAppear];
  [self.view.window makeFirstResponder:_metalView];

  // now initialize vulkan application (start Render Thread an update thread)
  // create pthreads, which are to be pthread_timed_kill_np at close?
  _app->MetalLayer = _metalView.metalLayer;
  _app->RenderThread = createThreadOrExit(renderThreadFunc, _app);
  _app->UpdateThread = createThreadOrExit(updateThreadFunc, _app);
}
- (void)viewDidDisappear {
  [super viewDidDisappear];
  [_metalView stopDisplayLink];
  // now join both threads
  _app->pauseRendering();
  _app->signalStopRendering();
  pthread_join(_app->RenderThread, nullptr);
  _app->RenderThread = nullptr;
  _app->signalStopUpdating();
  pthread_join(_app->UpdateThread, nullptr);
  _app->UpdateThread = nullptr;
}

@end

// --------------------------------------------------------------------------
