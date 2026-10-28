using System.Collections.Generic;
using AetherVk.Logic.Services.NativeInput;

namespace AetherVk.Logic.Input;

/// <summary>
/// Converts types from the <c>NativeInput.*</c> domain into <c>Logic.Input.*</c> types.
/// Internal to ui-logic — never exposed to ui-app.
/// </summary>
internal static class NativeInputConverter
{
  // Platform-agnostic virtual key names keyed by OS virtual keycode.
  // Windows VK_* codes and X11 keysyms for printable ASCII letters share
  // the same 0x41–0x5A range. macOS keyCodes differ, but the platform
  // subclass is expected to normalise to this range before publishing.
  // Extend as new Viewport actions require additional keys.
  private static readonly Dictionary<uint, string> _keyNames = new()
  {
    [0x30] = "D0", [0x31] = "D1", [0x32] = "D2", [0x33] = "D3", [0x34] = "D4",
    [0x35] = "D5", [0x36] = "D6", [0x37] = "D7", [0x38] = "D8", [0x39] = "D9",
    [0x41] = "A", [0x42] = "B", [0x43] = "C", [0x44] = "D",
    [0x45] = "E", [0x46] = "F", [0x47] = "G", [0x48] = "H",
    [0x49] = "I", [0x4A] = "J", [0x4B] = "K", [0x4C] = "L",
    [0x4D] = "M", [0x4E] = "N", [0x4F] = "O", [0x50] = "P",
    [0x51] = "Q", [0x52] = "R", [0x53] = "S", [0x54] = "T",
    [0x55] = "U", [0x56] = "V", [0x57] = "W", [0x58] = "X",
    [0x59] = "Y", [0x5A] = "Z",
    [0x09] = "Tab",
    [0x1B] = "Escape",
    [0x0D] = "Return",
    [0x20] = "Space",
    [0x60] = "NumPad0", [0x61] = "NumPad1", [0x62] = "NumPad2", [0x63] = "NumPad3",
    [0x64] = "NumPad4", [0x65] = "NumPad5", [0x66] = "NumPad6", [0x67] = "NumPad7",
    [0x68] = "NumPad8", [0x69] = "NumPad9", [0x6E] = "NumPadDecimal",
  };

  /// <summary>
  /// With NumLock off, Windows reports the keypad keys as the navigation keys they double as
  /// (keypad 1 → VK_END, 5 → VK_CLEAR, . → VK_DELETE…), but without the extended-key flag
  /// (lParam bit 24) that the dedicated navigation keys carry. Maps those back to VK_NUMPAD0–9 /
  /// VK_DECIMAL so the keypad shortcuts work whatever the NumLock state; every other key, and any
  /// extended one, is returned unchanged.
  /// </summary>
  internal static uint NormalizeWin32NumPad(uint virtualKey, bool isExtended)
  {
    if (isExtended) return virtualKey;
    return virtualKey switch
    {
      0x2D => 0x60, // VK_INSERT → VK_NUMPAD0
      0x23 => 0x61, // VK_END    → VK_NUMPAD1
      0x28 => 0x62, // VK_DOWN   → VK_NUMPAD2
      0x22 => 0x63, // VK_NEXT   → VK_NUMPAD3
      0x25 => 0x64, // VK_LEFT   → VK_NUMPAD4
      0x0C => 0x65, // VK_CLEAR  → VK_NUMPAD5
      0x27 => 0x66, // VK_RIGHT  → VK_NUMPAD6
      0x24 => 0x67, // VK_HOME   → VK_NUMPAD7
      0x26 => 0x68, // VK_UP     → VK_NUMPAD8
      0x21 => 0x69, // VK_PRIOR  → VK_NUMPAD9
      0x2E => 0x6E, // VK_DELETE → VK_DECIMAL
      _ => virtualKey,
    };
  }

  /// <summary>
  /// Converts a <see cref="ComposedKeystroke"/> into an <see cref="InputChord"/>.
  /// Policy: modifier flags and key name are taken from the <em>last</em> event in the
  /// sequence, matching the mental model of "hold modifier, press key".
  /// </summary>
  /// <returns>
  /// <c>null</c> if the sequence is empty or the final keycode has no registered mapping.
  /// </returns>
  internal static InputChord? ToInputChord(ComposedKeystroke stroke)
  {
    if (stroke.Sequence.Length == 0) return null;

    var last = stroke.Sequence[stroke.Sequence.Length - 1];
    if (!_keyNames.TryGetValue(last.KeyCode, out var keyName)) return null;

    return new InputChord(
      Key: keyName,
      Shift: last.Modifiers.HasFlag(NativeModifierFlags.Shift),
      Ctrl: last.Modifiers.HasFlag(NativeModifierFlags.Control),
      Alt: last.Modifiers.HasFlag(NativeModifierFlags.Alt)
    );
  }

  /// <summary>
  /// Maps <see cref="NativeModifierFlags"/> to <see cref="InputModifiers"/>.
  /// Note: the bit values differ between the two enums, so an explicit
  /// per-flag mapping is required — no direct cast.
  /// </summary>
  internal static InputModifiers ToInputModifiers(NativeModifierFlags flags)
  {
    var result = InputModifiers.None;
    if (flags.HasFlag(NativeModifierFlags.Shift))   result |= InputModifiers.Shift;
    if (flags.HasFlag(NativeModifierFlags.Control)) result |= InputModifiers.Ctrl;
    if (flags.HasFlag(NativeModifierFlags.Alt))     result |= InputModifiers.Alt;
    return result;
  }

  internal static InputChord ToInputChord(NativeMouseEvent ev)
  {
    string pointerStr = ev.Button switch
    {
      MouseButton.Left => "LeftPressed",
      MouseButton.Middle => "MiddlePressed",
      MouseButton.Right => "RightPressed",
      _ => "Other"
    };

    return new InputChord(
      Key: null,
      Shift: ev.Modifiers.HasFlag(NativeModifierFlags.Shift),
      Ctrl: ev.Modifiers.HasFlag(NativeModifierFlags.Control),
      Alt: ev.Modifiers.HasFlag(NativeModifierFlags.Alt),
      Pointer: pointerStr
    );
  }
}
