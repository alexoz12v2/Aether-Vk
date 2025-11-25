#include "event/system.h"

#include <thread>

namespace avk {

EventSystem::EventSystem(size_t cap) : m_queue(cap) {
  m_listenerMap.reserve(256);
}

bool EventSystem::addEvent(ev_t evType) {
  std::unique_lock wLock{m_mapMtx};
  auto [it, wasInserted] = m_listenerMap.try_emplace(evType);
  if (wasInserted) {
    it->second.reserve(256);
  }
  return wasInserted;
}

bool EventSystem::subscribe(ev_t evType, IEventListener *listener) {
  std::unique_lock wLock{m_mapMtx};
  auto it = m_listenerMap.find(evType);
  assert(it != m_listenerMap.cend());
  auto [lit, wasInserted] = it->second.insert(listener);
  return wasInserted;
}

void EventSystem::unsubscribe(ev_t evType, IEventListener *listener) {
  std::unique_lock wLock{m_mapMtx};
  auto it = m_listenerMap.find(evType);
  if (it != m_listenerMap.end()) {
    it->second.erase(listener);
  }
}

void EventSystem::publish(const Event &ev) {
  while (!m_queue.push(ev)) {
    std::this_thread::yield();
  }
}

void EventSystem::UTprocessEvents() {
  Event e{};
  while (m_queue.pop(e)) {
    std::shared_lock rLock{m_mapMtx};
    auto it = m_listenerMap.find(e.type);
    if (it != m_listenerMap.end()) {
      for (auto lit = it->second.begin(); lit != it->second.end(); /*inside*/) {
        if ((*lit)->onEvent(e)) {
          lit = it->second.erase(lit);
        } else {
          ++lit;
        }
      }
    }
  }
}

}  // namespace avk