#import "avk-application-delegate.h"

// Frameworks
#import <Cocoa/Cocoa.h>

// https://developer.apple.com/documentation/appkit/nsapplication

int main([[maybe_unused]] int argc, [[maybe_unused]] char const *argv[]) {
  @autoreleasepool {
    [NSApplication sharedApplication];

    // Set up delegate (similar to windows procedure handing)
    [NSApp setDelegate:[[AVKApplicationDelegate alloc] init]];

    // run main event loop (similiar to WinMain message loop)
    [NSApp run];
  }
}