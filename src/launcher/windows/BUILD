genrule(
    name = "asan_dll",
    srcs = ["@msvc_asan//:asan_dll"],
    outs = ["clang_rt.asan_dynamic-x86_64.dll"],
    cmd = "cp $(locations @msvc_asan//:asan_dll) $@",
)

cc_binary(
    name = "avk_windows_launcher.exe",
    srcs = ["main.cpp"],
    data = [":asan_dll"],
    defines = [
        # todo make substitution
        "AVK_OS_WINDOWS",
        "AVK_COMPILER_CLANG",
        "AVK_ARCH_X86_64",
        "WIN32_LEAN_AND_MEAN",
        "NOMINMAX",
        "UNICODE",
    ],
)