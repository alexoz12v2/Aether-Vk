#pragma once

// assumes Clang Toolchain
#define AVK_NO_CFI __attribute__((no_sanitize("cfi")))

#ifdef AVK_DEBUG
#  define AVK_DBK(...) __VA_ARGS__
#else
#  define AVK_DBK(...)
#endif