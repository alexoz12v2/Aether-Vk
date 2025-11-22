#import "avk-application-delegate.h"

#import "avk-primary-window.h"

@implementation AVKApplicationDelegate {
  avk::MacosApplication* ivar_app;
  NSWindow* ivar_window;
}
// declare memory backings for properties
@synthesize window = ivar_window;

- (instancetype)init {
  self = [super init];
  if (self) {
    ivar_app = new avk::MacosApplication;
    if (!ivar_app) return nil;  // fail if nullptr
    ivar_window = nil;          // initalized at applicationDidFinishLaunching
  }
  return self;
}

// ---- from the NSApplicationDelegate Protocol ----------------------

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)sender {
  // sent when last window, opened by this app, was closed. reopen or should
  // close?
  return YES;
}

- (void)applicationWillTerminate:(NSNotification*)notification {
  // destroy application class (render and update thread already joined on
  // primary window termination)
  delete ivar_app;
  ivar_app = nullptr;
}

- (void)applicationDidFinishLaunching:(NSNotification*)notification {
  NSLog(@"App Launched");

  // now create main window
  NSRect frame = NSMakeRect(100, 100, 800, 600);
  ivar_window = [[NSWindow alloc]
      initWithContentRect:frame
                styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                           NSWindowStyleMaskResizable)
                  backing:NSBackingStoreBuffered
                    defer:NO];
  [ivar_window setTitle:@"Window (To Be localized)"];
  // create controller and set as content
  _viewController = [[AVKVulkanViewController alloc] initWithApp:ivar_app
                                                        andFrame:frame];
  ivar_window.contentViewController = _viewController;

  // set key window and main window as primary window
  //--  *** Assertion failure in -[NSWindow _changeJustMain], NSWindow.m:14794
  //--  An uncaught exception was raised
  //--  Invalid parameter not satisfying: [self canBecomeMainWindow]
  // [ivar_window makeMainWindow];
  [ivar_window makeKeyAndOrderFront:nil];

  // (not needed since it's not a scoped variable anymore) Keep a reference
  // -> if this is not retain, ARC frees it
  // objc_setAssociatedObject([NSApplication sharedApplication], @"mainWindow",
  // window, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}
@end