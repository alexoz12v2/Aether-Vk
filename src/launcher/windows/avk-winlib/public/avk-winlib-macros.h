#pragma once

#if defined(AVK_OS_WINDOWS) || defined(_WIN32)

#ifdef AVK_WIN32_COMUTILS_EXPORT
#define AVK_COMUTILS_API __declspec(dllexport)
#else
#define AVK_COMUTILS_API __declspec(dllimport)
#endif

#else
#error "This is a Windows only artifact"
#endif