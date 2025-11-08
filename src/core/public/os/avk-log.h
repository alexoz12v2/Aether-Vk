#pragma once

#ifdef AVK_OS_ANDROID
#  include <ostream>
extern std::ostream g_clogi;
extern std::ostream g_cloge;
extern std::ostream g_clogw;

#  define LOGI g_clogi
#  define LOGW g_clogw
#  define LOGE g_cloge

# define AVK_LOG_RED ""
# define AVK_LOG_YLW ""
# define AVK_LOG_RST ""

#else

# define AVK_LOG_RED "\033[31m"
# define AVK_LOG_YLW "\033[93m"
# define AVK_LOG_RST "\033[0m"

#  include <iostream>

#  define LOGI std::cout
#  define LOGW std::cout
#  define LOGE std::cerr

#endif
