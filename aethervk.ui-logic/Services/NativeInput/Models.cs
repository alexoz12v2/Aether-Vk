using System;

namespace AetherVk.Logic.Services.NativeInput;

[Flags]
internal enum NativeModifierFlags
{
  None = 0,
  Shift = 1 << 1,
  Control = 1 << 2,
  Alt = 1 << 3,
  Super = 1 << 4, // Command on Mac, Windows Key on Windows
}

internal enum MouseButton { None, Left, Right, Middle }

// derived, non abstract records need to repeat these fields. This triggers a special behaviour on
// the equality operator, in the sense that, even for reference types, it doesn't default to
// reference equality, but class and field based equality
internal abstract record NativeInputEvent(NativeModifierFlags Modifiers, long Timestamp);

internal record NativeKeyEvent(
    uint KeyCode,
    bool IsDown,
    NativeModifierFlags Modifiers,
    long Timestamp) : NativeInputEvent(Modifiers, Timestamp);

internal record NativeMouseEvent(
    double X,
    double Y,
    MouseButton Button,
    bool IsDown,
    NativeModifierFlags Modifiers,
    long Timestamp) : NativeInputEvent(Modifiers, Timestamp);

internal record ComposedKeystroke(NativeKeyEvent[] Sequence, long Timestamp);

