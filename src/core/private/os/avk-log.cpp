#include "os/avk-log.h"

#ifdef AVK_OS_ANDROID
#  include "os/android/avk-basic-logcat.h"

static avk::AndroidOut s_clogiBuf{"AVK Core", ANDROID_LOG_INFO};
static avk::AndroidOut s_clogeBuf{"AVK Core", ANDROID_LOG_ERROR};
static avk::AndroidOut s_clogwBuf{"AVK Core", ANDROID_LOG_WARN};

std::ostream g_clogi{&s_clogiBuf};
std::ostream g_cloge{&s_clogeBuf};
std::ostream g_clogw{&s_clogwBuf};
#endif
