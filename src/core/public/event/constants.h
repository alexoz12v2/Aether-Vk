#pragma once

#include "utils/bits.h"
#include "utils/integer.h"

namespace avk {

using ev_t = uint64_t;

// clang-format off
enum class KeyCode : uint16_t {
  // Letters
  A, B, C, D, E, F, G, H, I, J, K, L, M,
  N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
  // Numbers
  Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
  // Function Keys
  F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
  // Control Keys
  Escape, Tab, CapsLock, Shift, Control, Alt, Super, Menu, Space, Enter, Backspace,
  // Arrow Keys
  Left, Right, Up, Down,
  // Editing / Navigation
  Insert, Delete, Home, End, PageUp, PageDown,
  // Symbols / Punctuation
  Minus, Equal, LeftBracket, RightBracket, Backslash, Semicolon,
  Apostrophe, Comma, Period, Slash, GraveAccent,
  // Numpad
  NumPad0, NumPad1, NumPad2, NumPad3, NumPad4, NumPad5, NumPad6, NumPad7, NumPad8, NumPad9,
  NumPadDecimal, NumPadDivide, NumPadMultiply, NumPadSubtract, NumPadAdd, NumPadEnter,
  // Misc / Media
  PrintScreen, ScrollLock, Pause, NumLock, VolumeUp, VolumeDown, Mute, MediaNext, MediaPrev, MediaStop, MediaPlayPause,
  Unknown
};
// clang-format on
enum class MouseButton : uint8_t {
  Left = 0,
  Right = 1,
  Middle = 2,
  Button4 = 3,  // Typically "back" on some mice
  Button5 = 4,  // Typically "forward"
  Button6 = 5,
  Button7 = 6,
  Button8 = 7,
  None = 255  // No button
};

}  // namespace avk

namespace avk::events {
using namespace avk::literals;
// clang-format off
// Keyboard events (`key`)
inline ev_t constexpr EvKeyDown       = "KeyDown"_hash;
inline ev_t constexpr EvKeyUp         = "KeyUp"_hash;
inline ev_t constexpr EvKeyRepeat     = "KeyRepeat"_hash;      // Key held down
inline ev_t constexpr EvTextInput     = "TextInput"_hash;      // Text input (IME or normal)

// Mouse / Touch events (`mouse`)
inline ev_t constexpr EvMouseButtonDown   = "MouseButtonDown"_hash;
inline ev_t constexpr EvMouseButtonUp     = "MouseButtonUp"_hash;
inline ev_t constexpr EvMouseMove         = "MouseMove"_hash;
inline ev_t constexpr EvMouseScrollUp     = "MouseScrollUp"_hash;
inline ev_t constexpr EvMouseScrollDown   = "MouseScrollDown"_hash;
inline ev_t constexpr EvMouseEnter        = "MouseEnter"_hash;
inline ev_t constexpr EvMouseLeave        = "MouseLeave"_hash;

// Touch events (for mobile) (`touch`)
inline ev_t constexpr EvTouchBegin    = "TouchBegin"_hash;
inline ev_t constexpr EvTouchEnd      = "TouchEnd"_hash;
inline ev_t constexpr EvTouchMove     = "TouchMove"_hash;
inline ev_t constexpr EvTouchCancel   = "TouchCancel"_hash;

// Window events
inline ev_t constexpr EvWindowResize       = "WindowResize"_hash;
inline ev_t constexpr EvWindowClose        = "WindowClose"_hash;
inline ev_t constexpr EvWindowFocus        = "WindowFocus"_hash;
inline ev_t constexpr EvWindowLostFocus    = "WindowLostFocus"_hash;
inline ev_t constexpr EvWindowMinimize     = "WindowMinimize"_hash;
inline ev_t constexpr EvWindowMaximize     = "WindowMaximize"_hash;
inline ev_t constexpr EvWindowRestore      = "WindowRestore"_hash;

// System / application lifecycle events
inline ev_t constexpr EvAppPause       = "AppPause"_hash;       // e.g., Android/iOS backgrounded
inline ev_t constexpr EvAppResume      = "AppResume"_hash;      // e.g., Android/iOS foregrounded
inline ev_t constexpr EvAppLowMemory   = "AppLowMemory"_hash;   // OS signals low memory
// clang-format on
}  // namespace avk::events