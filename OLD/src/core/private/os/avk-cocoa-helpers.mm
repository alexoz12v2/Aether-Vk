#import "avk-cocoa-helpers.h"

#import <Cocoa/Cocoa.h>

void avkShowNSAlertAndAbort(char const* str) {
  dispatch_async(dispatch_get_main_queue(), ^{
    @autoreleasepool {
      NSString* fmtMsg = [NSString stringWithUTF8String:str];
      NSAlert* alert = [[NSAlert alloc] init];
      [alert setMessageText:@"Error"];
      [alert setInformativeText:fmtMsg];
      [alert addButtonWithTitle:@"OK"];
      [alert setAlertStyle:NSAlertStyleCritical];
      [alert runModal];
      abort();
    }
  });

  // wait to die
  while (true) {
    [NSThread sleepForTimeInterval:0.01 /*seconds*/];
  }
}
