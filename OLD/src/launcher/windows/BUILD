load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

cc_binary(
    name = "avk_windows_launcher",
    srcs = ["main.cpp"],
    deps = select({
        "//:enable_asan": ["@cc_windows_x86_64_clang_config//:asan_dynamic_runtime"],
        "//conditions:default": [],
    }) + [
        "@vulkan_sdk_repo//:vulkan",
    ],
    copts = [ # assumes using clang++
        "-DAAAA_AAAA",
    ],
    features = select({
        "//:enable_asan": ["asan_ubsan"],
        "//conditions:default": [],
    }) + [
        "link_windows_gui",
    ],
    # Unusable cause it defaults to MSVC /D syntax even when using clang++
    #defines = [],
    target_compatible_with = ["@platforms//os:windows", "@platforms//cpu:x86_64"],
)

refresh_compile_commands(
    name = "avk_win_commands",
    targets = {
        ":avk_windows_launcher": "",
    },
)
