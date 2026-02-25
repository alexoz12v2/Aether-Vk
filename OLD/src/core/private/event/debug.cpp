#include "event/debug.h"

#include "event/constants.h"
#include "os/avk-log.h"

// library
#include <sstream>
#include <string>
#include <string_view>

namespace avk {

static std::string_view stringFromkeyCode(KeyCode k) {
  using namespace std::string_view_literals;
  // clang-format off
  switch (k) {
    case KeyCode::A: return "A"sv; case KeyCode::B: return "B"sv; case KeyCode::C: return "C"sv;
    case KeyCode::D: return "D"sv; case KeyCode::E: return "E"sv; case KeyCode::F: return "F"sv;
    case KeyCode::G: return "G"sv; case KeyCode::H: return "H"sv; case KeyCode::I: return "I"sv;
    case KeyCode::J: return "J"sv; case KeyCode::K: return "K"sv; case KeyCode::L: return "L"sv;
    case KeyCode::M: return "M"sv; case KeyCode::N: return "N"sv; case KeyCode::O: return "O"sv;
    case KeyCode::P: return "P"sv; case KeyCode::Q: return "Q"sv; case KeyCode::R: return "R"sv;
    case KeyCode::S: return "S"sv; case KeyCode::T: return "T"sv; case KeyCode::U: return "U"sv;
    case KeyCode::V: return "V"sv; case KeyCode::W: return "W"sv; case KeyCode::X: return "X"sv;
    case KeyCode::Y: return "Y"sv; case KeyCode::Z: return "Z"sv; case KeyCode::Num0: return "Num0"sv;
    case KeyCode::Num1: return "Num1"sv; case KeyCode::Num2: return "Num2"sv; case KeyCode::Num3: return "Num3"sv;
    case KeyCode::Num4: return "Num4"sv; case KeyCode::Num5: return "Num5"sv; case KeyCode::Num6: return "Num6"sv;
    case KeyCode::Num7: return "Num7"sv; case KeyCode::Num8: return "Num8"sv; case KeyCode::Num9: return "Num9"sv;
    case KeyCode::F1: return "F1"sv; case KeyCode::F2: return "F2"sv; case KeyCode::F3: return "F3"sv;
    case KeyCode::F4: return "F4"sv; case KeyCode::F5: return "F5"sv; case KeyCode::F6: return "F6"sv;
    case KeyCode::F7: return "F7"sv; case KeyCode::F8: return "F8"sv; case KeyCode::F9: return "F9"sv;
    case KeyCode::F10: return "F10"sv; case KeyCode::F11: return "F11"sv; case KeyCode::F12: return "F12"sv;
    case KeyCode::Escape: return "Escape"sv; case KeyCode::Tab: return "Tab"sv; case KeyCode::CapsLock: return "CapsLock"sv;
    case KeyCode::Shift: return "Shift"sv; case KeyCode::Control: return "Control"sv; case KeyCode::Alt: return "Alt"sv;
    case KeyCode::Space: return "Space"sv; case KeyCode::Enter: return "Enter"sv; case KeyCode::Backspace: return "Backspace"sv;
    case KeyCode::Command: return "Command"sv; case KeyCode::Left: return "Left"sv; case KeyCode::Right: return "Right"sv;
    case KeyCode::Up: return "Up"sv; case KeyCode::Down: return "Down"sv; case KeyCode::Insert: return "Insert"sv;
    case KeyCode::Delete: return "Delete"sv; case KeyCode::Home: return "Home"sv; case KeyCode::End: return "End"sv;
    case KeyCode::PageUp: return "PageUp"sv; case KeyCode::PageDown: return "PageDown"sv; case KeyCode::Minus: return "Minus"sv;
    case KeyCode::Equal: return "Equal"sv; case KeyCode::LeftBracket: return "LeftBracket"sv; case KeyCode::RightBracket: return "RightBracket"sv;
    case KeyCode::Backslash: return "Backslash"sv; case KeyCode::Semicolon: return "Semicolon"sv; case KeyCode::Apostrophe: return "Apostrophe"sv;
    case KeyCode::Comma: return "Comma"sv; case KeyCode::Period: return "Period"sv; case KeyCode::Slash: return "Slash"sv;
    case KeyCode::GraveAccent: return "GraveAccent"sv; case KeyCode::NumPad0: return "NumPad0"sv; case KeyCode::NumPad1: return "NumPad1"sv;
    case KeyCode::NumPad2: return "NumPad2"sv; case KeyCode::NumPad3: return "NumPad3"sv; case KeyCode::NumPad4: return "NumPad4"sv;
    case KeyCode::NumPad5: return "NumPad5"sv; case KeyCode::NumPad6: return "NumPad6"sv; case KeyCode::NumPad7: return "NumPad7"sv;
    case KeyCode::NumPad8: return "NumPad8"sv; case KeyCode::NumPad9: return "NumPad9"sv; case KeyCode::NumPadDecimal: return "NumPadDecimal"sv;
    case KeyCode::NumPadDivide: return "NumPadDivide"sv; case KeyCode::NumPadMultiply: return "NumPadMultiply"sv; case KeyCode::NumPadSubtract: return "NumPadSubtract"sv;
    case KeyCode::NumPadAdd: return "NumPadAdd"sv; case KeyCode::NumPadEnter: return "NumPadEnter"sv; case KeyCode::PrintScreen: return "PrintScreen"sv;
    case KeyCode::ScrollLock: return "ScrollLock"sv; case KeyCode::Pause: return "Pause"sv; case KeyCode::NumLock: return "NumLock"sv;
    case KeyCode::VolumeUp: return "VolumeUp"sv; case KeyCode::VolumeDown: return "VolumeDown"sv; case KeyCode::Mute: return "Mute"sv;
    case KeyCode::MediaNext: return "MediaNext"sv; case KeyCode::MediaPrev: return "MediaPrev"sv; case KeyCode::MediaStop: return "MediaStop"sv;
    case KeyCode::MediaPlayPause: return "MediaPlayPause"sv;
    default:
      return "Unknown"sv;
  }
  // clang-format on
}

// TODO add more if needed
bool EventLogger::onEvent(Event const& ev) {
  std::stringstream content;
  content << "[EventLogger] onEvent: ( 0x" << std::hex << ev.emitterId << " ) ";
  bool key = false;
  switch (ev.type) {
    using namespace events;
    case EvKeyDown:
      content << "EvKeyDown { ";
      key = true;
      break;
    case EvKeyUp:
      content << "EvKeyDown { ";
      key = true;
      break;
    case EvKeyRepeat:
      content << "EvKeyDown { ";
      key = true;
      break;
  }

  if (key) {
    content << stringFromkeyCode(ev.u.key.key);
  }

  content << " }" << std::endl;
  LOGI << content.str() << std::flush;

  return false;
}

}  // namespace avk