cc_binary(
    name = "avk_windows_launcher",
    srcs = ["main.cpp"],
    deps = [
        "@cc_windows_x86_64_clang_config//:asan_dynamic_runtime",
    ],
    copts = [ # assumes using clang++
        "-DAAAA_AAAA",
    ],
    # Unusable cause it defaults to MSVC /D syntax even when using clang++
    #defines = [],
    target_compatible_with = ["@platforms//os:windows", "@platforms//cpu:x86_64"],
)
