#pragma once

#include "event/types.h"

namespace avk {

class EventLogger : public IEventListener {
 public:
  bool onEvent(Event const& ev) override;
};

}  // namespace avk
