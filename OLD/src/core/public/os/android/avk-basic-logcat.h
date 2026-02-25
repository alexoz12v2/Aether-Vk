#pragma once

#ifndef AVK_OS_ANDROID
#  error "Android specific header"
#endif

#include <android/log.h>

#include <sstream>

namespace avk {

/// Use this class to create an output stream that writes to logcat. By default,
/// a global one is defined as @a aout
/// TODO: add support for __android_log_print with multiple arguments
class AndroidOut : public std::stringbuf {
 public:
  /// Creates a new output stream for logcat
  /// @param kLogTag the log tag to output
  inline AndroidOut(const char* kLogTag,
                    android_LogPriority priority = ANDROID_LOG_DEBUG)
      : m_logTag(kLogTag), m_logLevel(priority) {}

 protected:
  int sync() override {
    __android_log_print(m_logLevel, m_logTag, "%s", str().c_str());
    str("");
    return 0;
  }

 private:
  const char* m_logTag;
  android_LogPriority m_logLevel;
};

}  // namespace avk
