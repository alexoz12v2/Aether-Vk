#import "avk-cocoa-helpers.h"

#import <Cocoa/Cocoa.h>

void avkShowNSAlertAndAbort(char const* str) {
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
}
