"""C++ toolchain configuration"""

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load("@rules_cc//cc:cc_toolchain_config_lib.bzl", "action_config", "tool", "feature", "flag_set", "flag_group", "with_feature_set")

# https://github.com/bazelbuild/bazel/blob/4dd8c2053d4631f14f28a5a19012c93d0c9f1e8e/src/main/starlark/builtins_bzl/common/cc/toolchain_config/cc_toolchain_config_info.bzl#L56
# https://bazel.build/docs/cc-toolchain-config-reference#cctoolchainconfiginfo-build-variables
# https://bazel.build/rules/lib/toplevel/cc_common#create_cc_toolchain_config_info
def _cc_toolchain_config_impl(ctx):
    """Implementation of the cc_toolchain_config rule for Windows x86_64 with Clang."""
    vs_path = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community"
    llvm_path = "C:/Program Files/LLVM"
    llvm_bin = llvm_path + "\\bin\\"
    llvm_include_base = llvm_path + "\\lib\\clang\\21\\"
    sanitizers_path = llvm_path + "\\lib\\clang\\21\\lib\\windows\\"

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "windows_x86_64_clang", # path component
        host_system_name = "windows",
        target_system_name = "windows",
        target_cpu = "x86_64",
        compiler = "clang", # one of https://github.com/bazelbuild/rules_cc/blob/main/cc/compiler/BUILD
        cxx_builtin_include_directories = [
            llvm_include_base + "share/",
            llvm_include_base + "include/",
            "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/ATLMFC/include",
            "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/include",
            "C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/ucrt",
            "C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/shared",
            "C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/um",
            "C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/winrt",
            "C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/cppwinrt",
        ],
        features = [
            # get rid of default options
            feature(name = "default_link_flags", enabled = False),
            # feature(name = "strip", enabled = False), (disable the action)
            feature(name = "no_stripping", enabled = False),
            feature(name = "output_execpath_flags", enabled = False),
            # add custom options
            feature(
                name = "default_compiler_flags",
                enabled = True,
                flag_sets = [
                    flag_set(
                        actions = [
                            ACTION_NAMES.cpp_compile,
                            ACTION_NAMES.cpp_header_parsing,
                            ACTION_NAMES.cpp_module_compile,
                        ],
                        flag_groups = [
                            flag_group(flags = [
                                "-DAVK_ARCH_X86_64", "-DAVK_COMPILER_CLANG", "-DAVK_OS_WINDOWS",
                                "-DNOMINMAX", "-DUNICODE", "-DWIN32_LEAN_AND_MEAN",
                                # system include directories are already specified on cxx_builtin_include_directories
                                # "-isystem \"C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/ATLMFC/include\"",
                                # "-isystem \"C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/include\"",
                                # "-isystem \"C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/ucrt\"",
                                # "-isystem \"C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/shared\"",
                                # "-isystem \"C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/um\"",
                                # "-isystem \"C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/winrt\"",
                                # "-isystem \"C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/cppwinrt\"",
                                "-fvisibility=hidden",
                                "-fno-fast-math",
                                "-faddrsig", "-fstrict-aliasing",
                                "-nogpuinc", "-nogpulib",
                                "-fstack-protector-all",
                                "-march=x86-64-v3",
                                "-fsanitize=address", "-fsanitize=undefined",
                                "-flto", "-fsanitize=cfi",
                                "-g", "-O0" ,
                                "-std=gnu++17",
                                "-D_DEBUG", "-D_DLL", "-D_MT", # equals /MDd 
                                "-Xclang", "--dependent-lib=msvcrtd",
                                "-Xclang", "-gcodeview",
                                "-Wall", "-Wextra", "-pedantic", "-Werror",
                            ]),
                        ],
                    ),
                ],
            ),
            feature(
                name = "msvc_linker_flags",
                enabled = True,
                flag_sets = [
                    flag_set(
                        actions = [
                            ACTION_NAMES.cpp_link_executable,
                            ACTION_NAMES.cpp_link_dynamic_library,
                        ],
                        flag_groups = [
                            flag_group(
                                flags = [
                                    "/NOLOGO",
                                    "/DEBUG",
                                    "/SUBSYSTEM:WINDOWS",
                                    "/MACHINE:X64",
                                    "/INCREMENTAL:NO",
                                    "User32.lib",
                                    "Ole32.lib",
                                    "Kernel32.lib",
                                    "/OUT:%{output_execpath}",  # Bazel will substitute this
                                ] + [ 
                                    #"/WHOLEARCHIVE:" +  # uncomment if you switch to /MT or /MTd
                                    sanitizers_path + f for f in [
                                    # "clang_rt.asan-x86_64.lib" # uncomment if you switch to /MT or /MTd
                                    "clang_rt.asan_dynamic-x86_64.lib", # must for /MDd
                                    "clang_rt.asan_dynamic_runtime_thunk-x86_64.lib",
                                ] ]
                            ),
                        ],
                    ),
                ],
            ),
        ],
        action_configs = [
            action_config(
                action_name = ACTION_NAMES.cpp_compile,
                enabled = True,
                tools = [tool(path = llvm_bin + "clang++.exe")],
            ),
            action_config(
                action_name = ACTION_NAMES.c_compile,
                enabled = True,
                tools = [tool(path = llvm_bin + "clang.exe")],
            ),
            action_config(
                action_name = ACTION_NAMES.assemble,
                enabled = True,
                tools = [tool(path = llvm_bin + "clang.exe")],
            ),
            action_config(
                action_name = ACTION_NAMES.preprocess_assemble,
                enabled = True,
                tools = [tool(path = llvm_bin + "clang.exe")],
            ),
            action_config(
                action_name = ACTION_NAMES.cpp_link_executable,
                enabled = True,
                tools = [
                    tool(
                        #with_features = [with_feature_set(["msvc_linker_flags"])],
                        path = llvm_bin + "lld-link.exe",
                    )
                ],
                #flag_sets = [
                #    flag_set(
                #        flag_groups = [
                #            flag_group(
                #                flags = [
                #                    "/NOLOGO",
                #                    "/DEBUG",
                #                    "/SUBSYSTEM:WINDOWS",
                #                    "/MACHINE:X64",
                #                    "/INCREMENTAL:NO",
                #                    "/OUT:%{output_execpath}",
                #                    # Add any required libraries here
                #                ]
                #            ),
                #        ],
                #    ),
                #],
            ),
            action_config(
                action_name = ACTION_NAMES.cpp_link_dynamic_library,
                enabled = True,
                tools = [tool(path = llvm_bin + "lld-link.exe")],
            ),
            action_config(
                action_name = ACTION_NAMES.cpp_link_static_library,
                enabled = True,
                tools = [tool(path = llvm_bin + "llvm-ar.exe")],
            ),
            #action_config(
            #    action_name = ACTION_NAMES.strip,
            #    enabled = False,
            #    tools = [tool(path = llvm_bin + "llvm-strip.exe")],
            #),
            action_config(
                action_name = ACTION_NAMES.objcopy_embed_data,
                enabled = True,
                tools = [tool(path = llvm_bin + "llvm-objcopy.exe")],
            ),
            action_config(
                action_name = ACTION_NAMES.dwp,
                enabled = True,
                tools = [tool(path = llvm_bin + "llvm-dwp.exe")],
            ),
            # Add more actions as needed, e.g., lto_indexing, lto_backend, etc.
        ]
    )

cc_toolchain_config = rule(
    implementation = _cc_toolchain_config_impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)