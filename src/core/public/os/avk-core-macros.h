#pragma once

// assumes Clang Toolchain
#define AVK_NO_CFI __attribute__((no_sanitize("cfi")))
