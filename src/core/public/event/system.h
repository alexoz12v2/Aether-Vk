#pragma once

#include "event/types.h"
#include "utils/mixins.h"
#include "fiber/mpmc.h"

// lib
#include <unordered_map>
#include <unordered_set>
#include <shared_mutex>

namespace avk {

class EventSystem : public NonMoveable {
 public:
  explicit EventSystem(size_t cap = 1024);

  /// necessary before pushing events for a given type
  bool addEvent(ev_t evType);

  /// listener registration for an existing event type
  /// listener must clean up its registration before being destroyed
  bool subscribe(ev_t evType, IEventListener* listener);

  /// clean up listener registration
  /// \warning Do not call it from a `onEvent`
  void unsubscribe(ev_t evType, IEventListener* listener);

  /// new event of a registered type
  void publish(Event const& ev);

 /// Update thread processing
 /// \note if needed add timeout
 void UTprocessEvents();

 private:
  MPMCQueue<Event> m_queue;
  std::unordered_map<ev_t, std::unordered_set<IEventListener*>> m_listenerMap;
  mutable std::shared_mutex m_mapMtx;
};

}