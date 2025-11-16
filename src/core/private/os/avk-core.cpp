#include <algorithm>
#include <cassert>

#include "os/avk-time.h"

#ifdef AVK_OS_WINDOWS
#  include <Windows.h>
#  include <timeapi.h>
#elif defined(AVK_OS_LINUX) || defined(AVK_OS_MACOS) || defined(AVK_OS_IOS) || \
    defined(AVK_OS_ANDROID)
#  include <time.h>
#else
#  error "Unsupported platform (could use <chrono> as fallback)"
#endif

namespace avk::os {

// value for 60 FPS, used to start
inline static timeus_t constexpr IdealDeltaTime = 16'667;

TimeInfo::TimeInfo(timeus_t fixedDeltaTime, timeus_t maximumDeltaTime,
                   float timeScale) noexcept
    : FixedDeltaTime(fixedDeltaTime),
      MaximumDeltaTime(maximumDeltaTime),
      TimeScale(timeScale) {
  assert(fixedDeltaTime < maximumDeltaTime);
  assert(timeScale > 0);
  for (uint32_t i = 0; i < DeltasWindowCount; ++i) {
    m_deltas[i] = IdealDeltaTime;
  }
#ifdef AVK_OS_WINDOWS
  {
    LARGE_INTEGER queryRes{};
    QueryPerformanceFrequency(&queryRes);
    m_frequency = queryRes.QuadPart;
    QueryPerformanceCounter(&queryRes);
    m_start = queryRes.QuadPart;
    m_start *= 1'000'000;
    m_start /= m_frequency;
  }
#elif defined(AVK_OS_LINUX) || defined(AVK_OS_ANDROID)
  {
    struct timespec tp{};
    int res = clock_gettime(CLOCK_MONOTONIC, &tp);
    assert(!res && "clock_gettime failed. check errno");
    m_start = static_cast<timeus_t>(tp.tv_sec) * 1'000'000 +
              static_cast<timeus_t>(tp.tv_nsec) / 1000;
  }
#elif defined(AVK_OS_MACOS) || defined(AVK_OS_IOS)
  {
    m_start =
        static_cast<timeus_t>(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) / 1000;
  }
#endif

  m_lastFixedUpdate = m_start;
  m_lastUpdate = m_start;
  m_lastRealUpdate = m_start;
}

static inline int32_t advanceIndex(int32_t& index) {
  static_assert(
      (TimeInfo::DeltasWindowCount & (TimeInfo::DeltasWindowCount - 1)) == 0,
      "TimeInfo::DeltasWindowCount not power of two");
  int32_t const result = index++;
  index &= TimeInfo::DeltasWindowCount - 1;
  return result;
}

void TimeInfo::UTupdate() {
  assert(FixedDeltaTime < MaximumDeltaTime);
  assert(TimeScale > 0);
  timeus_t timeRaw = 0;
#ifdef AVK_OS_WINDOWS
  {
    LARGE_INTEGER queryRes{};
    QueryPerformanceCounter(&queryRes);
    timeRaw = queryRes.QuadPart;
    timeRaw *= 1'000'000;
    timeRaw /= m_frequency;
  }
#elif defined(AVK_OS_LINUX) || defined(AVK_OS_ANDROID)
  {
    struct timespec tp{};
    int res = clock_gettime(CLOCK_MONOTONIC, &tp);
    assert(!res && "clock_gettime failed. check errno");
    timeRaw = static_cast<timeus_t>(tp.tv_sec) * 1'000'000 +
              static_cast<timeus_t>(tp.tv_nsec) / 1000;
  }
#elif defined(AVK_OS_MACOS) || defined(AVK_OS_IOS)
  {
    timeRaw =
        static_cast<timeus_t>(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) / 1000;
  }
#endif
  timeus_t const deltaRaw = timeRaw - m_lastRealUpdate;
  timeus_t const deltaScaled =
      deltaRaw * TimeScale.load(std::memory_order_relaxed);
  timeus_t const delta =
      std::min(deltaScaled, MaximumDeltaTime.load(std::memory_order_relaxed));
  const int32_t idx = m_deltasIndex.load(std::memory_order_relaxed);
  m_deltas[idx] = delta;  // normal write (sufficient?)
  m_deltasIndex.store(((idx + 1) & DeltasWindowCount - 1),
                      std::memory_order_release);

  m_lastRawDelta.store(deltaRaw, std::memory_order_release);
  m_lastRealUpdate.store(timeRaw, std::memory_order_release);
  m_lastUpdate.fetch_add(delta, std::memory_order_release);
}

void TimeInfo::UTfixedUpdate() {
  assert(FixedDeltaTime < MaximumDeltaTime);
  assert(TimeScale > 0);
  timeus_t const added = FixedDeltaTime.load(std::memory_order_relaxed);
  m_lastFixedUpdate += added;
}

bool TimeInfo::needsFixedUpdate() const {
  assert(FixedDeltaTime < MaximumDeltaTime);
  assert(TimeScale > 0);
  timeus_t const fixedDelta = FixedDeltaTime.load(std::memory_order_relaxed);
  return m_lastFixedUpdate + fixedDelta <= m_lastUpdate;
}

static inline timeus_t smooth(timeus_t const* deltas, const uint32_t start) {
  // Initialize EMA with the first sample
  int32_t index = start;
  timeus_t ema = deltas[index];
  advanceIndex(index);
  // Apply EMA update for remaining elements
  for (uint32_t i = 1; i < TimeInfo::DeltasWindowCount; ++i) {
    ema = (ema + deltas[index]) >> 1;  // (old + new) / 2
    advanceIndex(index);
  }
  return ema;
}

TimeReadings TimeInfo::current() const {
  assert(FixedDeltaTime < MaximumDeltaTime);
  assert(TimeScale > 0);
  TimeReadings time{};

  // Load all shared values with acquire semantics
  const int32_t idx = m_deltasIndex.load(std::memory_order_acquire);
  time.Time = m_lastUpdate.load(std::memory_order_acquire);
  time.FixedTime = m_lastFixedUpdate.load(std::memory_order_acquire);
  time.DeltaTime = m_deltas[idx];
  time.UnscaledTime = m_lastRealUpdate.load(std::memory_order_acquire);
  time.UnscaledDeltaTime = m_lastRawDelta.load(std::memory_order_acquire);

  // Smooth uses the deltas array and index (reading only)
  time.SmoothDeltaTime = smooth(m_deltas, idx);

  return time;
}

}  // namespace avk::os