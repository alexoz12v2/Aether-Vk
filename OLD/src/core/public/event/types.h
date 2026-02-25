#pragma once

#include <cstdint>

#include "event/constants.h"
#include "os/avk-time.h"
#include "utils/bits.h"
#include "utils/integer.h"

namespace avk {

inline int32_t constexpr MAX_POINTERS = 10;
inline int32_t constexpr MAX_CUSTOM_DATA = 128 / sizeof(uint64_t);

struct Event {
  ev_t type;
  os::timeus_t simTime;

  // might be an object for a custom event, might be a window for a input event
  // on desktop, ....
  id_t emitterId;

  union {
    // Keyboard input
    struct {
      KeyCode key;
      bool isRepeat;
    } key;

    // Mouse input
    struct {
      int x, y;
      MouseButton button;
      bool pressed;
    } mouse;

    // Touch / Multi-pointer input
    struct {
      int pointerCount;
      struct Pointer {
        int id;
        int x, y;
        float pressure;  // 0..1, optional
      } pointers[MAX_POINTERS];
    } touch;

    // Window events
    struct {
      int width;
      int height;
      bool focused;
      bool maximized;
      bool minimized;
    } window;

    // Custom user-defined events
    struct {
      // 8 byte alignment, hence it should be fine for almost anything
      uint64_t data[MAX_CUSTOM_DATA];
    } custom;
  } u;
};

class IEventListener {
 public:
  /// returns `true` if it should unregister
  virtual bool onEvent(Event const& ev) = 0;
  virtual ~IEventListener() = default;
};

}  // namespace avk
