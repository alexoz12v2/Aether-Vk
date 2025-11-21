#import "avk-primary-window.h"

@implementation AVKPrimaryWindow {

}
// Overridden from NSWindow
-(void)keyDown:(NSEvent*)event{
  // check modifiers
  NSEventModifierFlags flags = [event modifierFlags];
  BOOL command = (flags & NSEventModifierFlagCommand);
  BOOL control = (flags & NSEventModifierFlagControl);
  BOOL shift = (flags & NSEventModifierFlagShift);

  // get pressed character
  NSString *chars = [event charactersIgnoringModifiers];

  // control + command + F -> toggle fullscreen
  if (!shift && control && command && [chars isEqualToString:@"f"]) {
    NSLog(@"⌘ + ⌃ + F pressed!");
    [self toggleFullScreen:nil];
    return;
  }

  // command + W -> close window
  if (!shift && !control && command && [chars isEqualToString:@"w"]) {
    NSLog(@"⌘ + W pressed!");
    [self performClose:nil];
    return;
  }

  // TODO: handle Command + , -> Settings menu

  // otherwise pass to default handler
  [super keyDown:event];
}
@end