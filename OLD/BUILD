load("@bazel_skylib//rules:native_binary.bzl", "native_binary")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

# Example compilation with ASan (with clang windows toolchain:)
# bazel build //src/launcher/windows:avk_windows_launcher --verbose_failures --//:asan_ubsan --compilation_mode=fastbuild
bool_flag(
    name = "asan_ubsan",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "enable_asan",
    flag_values = {
        ":asan_ubsan": "true",
    },
)

alias(
    name = "slangc",
    actual = "@vulkan_sdk_repo//:slangc",
)

native_binary(
    name = "vkconfig-gui",
    src = select({
        "@platforms//os:windows": "@vulkan_sdk_repo//:sdk/Bin/vkconfig-gui.exe",
        "//conditions:default": "@vulkan_sdk_repo//:sdk/bin/vkconfig-gui",
    }),
)
