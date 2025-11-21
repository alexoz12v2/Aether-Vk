#import "avk-application-delegate.h"

#import "avk-primary-window.h"

@implementation AvkApplicationDelegate {
  avk::MacosApplication* ivar_app;
  NSWindow* ivar_window;
}
// declare memory backings for properties
@synthesize window = ivar_window;
@synthesize app = ivar_app;

- (instancetype) init {
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

-(void)applicationWillTerminate:(NSNotification*)notification {
  // Kill Render thread and update thread
}

- (void)applicationDidFinishLaunching:(NSNotification*)notification {
  NSLog(@"App Launched");

  // now create main window
  NSRect frame = NSMakeRect(100, 100, 800, 600);
  ivar_window = [[AVKPrimaryWindow alloc]
      initWithContentRect:frame
                styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                           NSWindowStyleMaskResizable)
                  backing:NSBackingStoreBuffered
                    defer:NO];
  [ivar_window setTitle:@"Window (To Be localized)"];
  [ivar_window makeKeyAndOrderFront:nil];

  // (not needed since it's not a scoped variable anymore) Keep a reference
  // -> if this is not retain, ARC frees it
  // objc_setAssociatedObject([NSApplication sharedApplication], @"mainWindow", window, OBJC_ASSOCIATION_RETAIN_NONATOMIC);

  // now initialize vulkan application (start Render Thread an update thread)
  // create pthreads, which are to be pthread_timed_kill_np at close?
}
@end