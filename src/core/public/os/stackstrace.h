#pragma once

#include <string>

// TODO: on macOS/Linux/android, needs linker option -rdynamic such that we keep
// symbols from dynamic libraries. This increases binary size, so we have a
// compiler definition to turn it off which is `AVK_NO_RDYNAMIC`

namespace avk {

std::string dumpStackTrace(uint32_t maxFrames = 32);

[[noreturn]] void showErrorScreenAndExit(char const* msg);

void printfWithStacktrace(char const* format, ...);

}  // namespace avk