#pragma once

#ifdef AVK_WINRUNTIME_EXPORTS
#define AVK_WINRUNTIME_API __declspec(dllexport)
#else
#define AVK_WINRUNTIME_API __declspec(dllimport)
#endif
