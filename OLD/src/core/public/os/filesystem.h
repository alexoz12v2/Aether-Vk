#pragma once

#include <filesystem>

// TODO better
#if defined(AVK_OS_ANDROID)
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#endif

namespace avk {
inline constexpr uint32_t MaxCharsForPath = 32 * 1024;

#if defined(AVK_OS_IOS)
# error "TODO"
#endif

#if defined(AVK_OS_ANDROID)
// TODO better (move into android folder)
std::vector<uint32_t> loadSpvFromAsset(AAssetManager* manager, char const* filename);
#endif

#if defined(AVK_OS_WINDOWS) || defined(AVK_OS_MACOS) || defined(AVK_OS_LINUX)
// TODO for linux use /proc/self/exe with std::filesystem::canonical
/// Get Executable file path (Desktop only)
std::filesystem::path getExecutablePath();
#endif

}  // namespace avk
