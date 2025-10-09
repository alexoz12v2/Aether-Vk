#import <Cocoa/Cocoa.h>
#import <objc/runtime.h>

@interface MyWindow : NSWindow
@end

@interface AppDelegate : NSObject <NSApplicationDelegate>
@property (strong) MyWindow *window;
@end

@implementation MyWindow
- (void)keyDown:(NSEvent *)event {
  // check modifiers
  NSEventModifierFlags flags = [event modifierFlags];
  BOOL command = (flags & NSEventModifierFlagCommand);
  BOOL control = (flags & NSEventModifierFlagControl);
  BOOL shift = (flags & NSEventModifierFlagShift);

  // get pressed character
  NSString *chars = [event charactersIgnoringModifiers];

  // control + command + F -> toggle fullscreen
  if (!shift && control && command && [chars isEqualToString:@"f"]) {
    NSLog(@"Command + Control + F pressed!");
    [self toggleFullScreen:nil];
    return;
  }

  // command + W -> close window
  if (!shift && !control && command && [chars isEqualToString:@"w"]) {
    NSLog(@"Command + W pressed!");
    [self performClose:nil];
    return;
  }

  // otherwise pass to default handler
  [super keyDown:event];
}
@end

@implementation AppDelegate

// called when app finishes launching
- (void)applicationDidFinishLaunching:(NSNotification *)notification {
  NSLog(@"App Launched");

  // Create the main window
  NSRect frame = NSMakeRect(100, 100, 800, 600);
  self.window = [[MyWindow alloc] initWithContentRect:frame
      styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable)
      backing:NSBackingStoreBuffered
      defer:NO];
  [self.window setTitle:@"Vulkan Window"];
  [self.window makeKeyAndOrderFront:nil];

  // (not needed since it' not a scoped variable anymore) Keep a reference so ARC doesn't free it
  // objc_setAssociatedObject([NSApplication sharedApplication], @"mainWindow", window, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

// handle app termination
- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
  return YES;
}

@end

int main([[maybe_unused]] int argc, [[maybe_unused]] char const *argv[]) {
  @autoreleasepool {
    NSApplication *app = [NSApplication sharedApplication];

    // Set up delegate (similar to windows procedure handing)
    AppDelegate *appDelegate = [[AppDelegate alloc] init];
    [app setDelegate:appDelegate];

    // run main event loop (similiar to WinMain message loop)
    [app run];
  }
  return 0;
}