#pragma once

namespace avk {

  // TODO: Move into a application-input file
  enum class EInputDevice {
    eKeyboard = 0,
    eMouse,
    eTouchScreen, // TODO more if needed
  };

  // singleton containing application level data
  // - input event queues
  // - input action mappings
  // - logger
  class ApplicationManager {
   public:
   private:
  };

}