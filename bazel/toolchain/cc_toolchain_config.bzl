"""C++ toolchain configuration"""

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load(
    "@rules_cc//cc:cc_toolchain_config_lib.bzl",
    "action_config",
    "artifact_name_pattern",
    "env_entry",
    "env_set",
    "feature",
    "flag_group",
    "flag_set",
    "make_variable",
    "tool",
    "tool_path",
    "variable_with_value",
    "with_feature_set",
)
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

all_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.assemble,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.cpp_module_deps_scanning,
    ACTION_NAMES.cpp20_module_compile,
    ACTION_NAMES.cpp20_module_codegen,
    ACTION_NAMES.clif_match,
    ACTION_NAMES.lto_backend,
]

all_cpp_compile_actions = [
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.cpp_module_deps_scanning,
    ACTION_NAMES.cpp20_module_compile,
    ACTION_NAMES.cpp20_module_codegen,
    ACTION_NAMES.clif_match,
]

preprocessor_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.cpp_module_deps_scanning,
    ACTION_NAMES.cpp20_module_compile,
    ACTION_NAMES.clif_match,
]

codegen_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.assemble,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.cpp20_module_codegen,
    ACTION_NAMES.lto_backend,
]

all_link_actions = [
    ACTION_NAMES.cpp_link_executable,
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

_FEATURE_NAMES = struct(
    COMPILE = struct(
        UNFILTERED_COMPILE_FLAGS = "unfiltered_compile_flags",
        USER_COMPILE_FLAGS = "user_compile_flags",
        DEFAULT_CPP_STD = "default_cpp_std",
        DEFAULT_COMPILE_FLAGS = "default_compile_flags",
        INCLUDE_PATHS = "include_paths",
        EXTERNAL_INCLUDE_PATHS = "external_include_paths",
        FRAME_POINTER = "frame_pointer",
        COMPILER_OUTPUT_FLAGS = "compiler_output_flags",
        SMALLER_BINARY = "smaller_binary",
        REMOVE_UNREFERENCED_CODE = "remove_unreferenced_code",
        COMPILER_INPUT_FLAGS = "compiler_input_flags",
        DISABLE_ASSERTIONS = "disable_assertions",
    ),
    LINK = struct(
        SHARED_FLAG = "shared_flag",
        INPUT_PARAM_FLAGS = "input_param_flags",
        ARCHIVER_FLAGS = "archiver_flags",
        DEFAULT_LINK_FLAGS = "default_link_flags",
        STATIC_LINK_MSVCRT = "static_link_msvcrt",
        DYNAMIC_LINK_MSVCRT = "dynamic_link_msvcrt",
        USER_LINK_FLAGS = "user_link_flags",
        GENERATE_LINKMAP = "generate_linkmap",
        OUTPUT_EXECPATH_FLAGS = "output_execpath_flags",
        LINKER_PARAM_FILE = "linker_param_file",
        IGNORE_NOISY_WARNINGS = "ignore_noisy_warnings",
        LINKSTAMPS = "linkstamps",
        LINKER_SUBSYSTEM_FLAG = "linker_subsystem_flag",
        DEF_FILE = "def_file",
        SYMBOL_CHECK = "symbol_check",
    ),
    COMMON = struct(
        MSVC_ENV = "msvc_env",
        MSVC_COMPILE_ENV = "msvc_compile_env",
        PREPROCESSOR_DEFINES = "preprocessor_defines",
        MSVC_LINK_ENV = "msvc_link_env",
        SYSROOT = "sysroot",
        PARSE_SHOWINCLUDES = "parse_showincludes",
        TREAT_WARNINGS_AS_ERRORS = "treat_warnings_as_errors",
        WINDOWS_EXPORT_ALL_SYMBOLS = "windows_export_all_symbols",
        NO_WINDOWS_EXPORT_ALL_SYMBOLS = "no_windows_export_all_symbols",
        TARGETS_WINDOWS = "targets_windows",
    ),
    OPTIONS = struct(
        ARCHIVE_PARAM_FILE = "archive_param_file",
        COMPILER_PARAM_FILE = "compiler_param_file",
        COPY_DYNAMIC_LIBRARIES_TO_BINARY = "copy_dynamic_libraries_to_binary",
        SUPPORTS_INTERFACE_SHARED_LIBRARIES = "supports_interface_shared_libraries",
        GENERATE_PDB_FILE = "generate_pdb_file",
        HAS_CONFIGURED_LINKER_PATH = "has_configured_linker_path",
        SUPPORTS_DYNAMIC_LINKER = "supports_dynamic_linker",
        NO_LEGACY_FEATURES = "no_legacy_features",
        NO_DOTD_FILE = "no_dotd_file",
        SHORTEN_VIRTUAL_INCLUDES = "shorten_virtual_includes",
        NO_STRIPPING = "no_stripping",

        FASTBUILD = "fastbuild",
        DBG = "dbg",
        OPT = "opt",
    ),
)

# https://github.com/bazelbuild/bazel/blob/4dd8c2053d4631f14f28a5a19012c93d0c9f1e8e/src/main/starlark/builtins_bzl/common/cc/toolchain_config/cc_toolchain_config_info.bzl#L56
# https://bazel.build/docs/cc-toolchain-config-reference#cctoolchainconfiginfo-build-variables
# https://bazel.build/rules/lib/toplevel/cc_common#create_cc_toolchain_config_info
def _cc_toolchain_config_impl(ctx):
    """Implementation of the cc_toolchain_config rule for Windows x86_64 with Clang."""
    llvm_cl = ctx.attr.llvm_bin_path + "/clang++.exe"
    llvm_link = ctx.attr.llvm_bin_path + "/lld-link.exe"
    llvm_ml = ctx.attr.llvm_bin_path + "/llvm-lm.exe"
    llvm_lib = ctx.attr.llvm_bin_path + "/llvm-lib.exe"

    ###############################################################################################################################
    # Action configs
    ###############################################################################################################################
    cpp_link_nodeps_dynamic_library_action = action_config(
        action_name = ACTION_NAMES.cpp_link_nodeps_dynamic_library,
        implies = [
            _FEATURE_NAMES.LINK.SHARED_FLAG,
            _FEATURE_NAMES.LINK.LINKSTAMPS,
            _FEATURE_NAMES.LINK.OUTPUT_EXECPATH_FLAGS,
            _FEATURE_NAMES.LINK.INPUT_PARAM_FLAGS,
            _FEATURE_NAMES.LINK.USER_LINK_FLAGS,
            _FEATURE_NAMES.LINK.LINKER_SUBSYSTEM_FLAG,
            _FEATURE_NAMES.LINK.LINKER_PARAM_FILE,
            _FEATURE_NAMES.COMMON.MSVC_ENV,
            _FEATURE_NAMES.OPTIONS.NO_STRIPPING,
            _FEATURE_NAMES.OPTIONS.HAS_CONFIGURED_LINKER_PATH,
            _FEATURE_NAMES.LINK.DEF_FILE,
        ],
        tools = [tool(path = llvm_link)],
    )

    cpp_link_static_library_action = action_config(
        action_name = ACTION_NAMES.cpp_link_static_library,
        implies = [
            _FEATURE_NAMES.LINK.ARCHIVER_FLAGS,
            _FEATURE_NAMES.LINK.INPUT_PARAM_FLAGS,
            _FEATURE_NAMES.LINK.LINKER_PARAM_FILE,
            _FEATURE_NAMES.COMMON.MSVC_ENV,
        ],
        tools = [tool(path = llvm_lib)],
    )

    assemble_action = action_config(
        action_name = ACTION_NAMES.assemble,
        implies = [
            _FEATURE_NAMES.COMPILE.COMPILER_INPUT_FLAGS,
            _FEATURE_NAMES.COMPILE.COMPILER_OUTPUT_FLAGS,
            _FEATURE_NAMES.COMMON.MSVC_ENV,
            _FEATURE_NAMES.COMMON.SYSROOT,
        ],
        tools = [tool(path = llvm_ml)]
    )

    preprocess_assemble_action = action_config(
        action_name = ACTION_NAMES.preprocess_assemble,
        implies = [
            _FEATURE_NAMES.COMPILE.COMPILER_INPUT_FLAGS,
            _FEATURE_NAMES.COMPILE.COMPILER_OUTPUT_FLAGS,
            _FEATURE_NAMES.COMMON.MSVC_ENV,
            _FEATURE_NAMES.COMMON.SYSROOT,
        ],
        tools = [tool(path = llvm_ml)]
    )

    c_compile_action = action_config(
        action_name = ACTION_NAMES.c_compile,
        implies = [
            _FEATURE_NAMES.COMPILE.COMPILER_INPUT_FLAGS,
            _FEATURE_NAMES.COMPILE.COMPILER_OUTPUT_FLAGS,
            _FEATURE_NAMES.COMMON.MSVC_ENV,
            _FEATURE_NAMES.COMMON.SYSROOT,
            _FEATURE_NAMES.COMPILE.USER_COMPILE_FLAGS,
        ],
        tools = [tool(path = llvm_cl)],
    )

    linkstamp_compile_action = action_config(
        action_name = ACTION_NAMES.linkstamp_compile,
        implies = [
            _FEATURE_NAMES.COMPILE.COMPILER_INPUT_FLAGS,
            _FEATURE_NAMES.COMPILE.COMPILER_OUTPUT_FLAGS,
            _FEATURE_NAMES.COMPILE.DEFAULT_COMPILE_FLAGS,
            _FEATURE_NAMES.COMMON.MSVC_ENV,
            _FEATURE_NAMES.COMPILE.USER_COMPILE_FLAGS,
            _FEATURE_NAMES.COMMON.SYSROOT,
            _FEATURE_NAMES.COMPILE.UNFILTERED_COMPILE_FLAGS,
        ],
        tools = [tool(path = llvm_cl)],
    )

    cpp_compile_action = action_config(
        action_name = ACTION_NAMES.cpp_compile,
        implies = [
            _FEATURE_NAMES.COMPILE.COMPILER_INPUT_FLAGS,
            _FEATURE_NAMES.COMPILE.COMPILER_OUTPUT_FLAGS,
            _FEATURE_NAMES.COMMON.MSVC_ENV,
            _FEATURE_NAMES.COMPILE.USER_COMPILE_FLAGS,
            _FEATURE_NAMES.COMMON.SYSROOT,
        ],
        tools = [tool(path = llvm_cl)],
    )

    cpp_link_executable_action = action_config(
        action_name = ACTION_NAMES.cpp_link_executable,
        implies = [
            _FEATURE_NAMES.LINK.LINKSTAMPS,
            _FEATURE_NAMES.LINK.OUTPUT_EXECPATH_FLAGS,
            _FEATURE_NAMES.LINK.INPUT_PARAM_FLAGS,
            _FEATURE_NAMES.LINK.USER_LINK_FLAGS,
            _FEATURE_NAMES.LINK.LINKER_SUBSYSTEM_FLAG,
            _FEATURE_NAMES.LINK.LINKER_PARAM_FILE,
            _FEATURE_NAMES.COMMON.MSVC_ENV,
            _FEATURE_NAMES.OPTIONS.NO_STRIPPING,
        ],
        tools = [tool(path = llvm_link)],
    )

    cpp_link_dynamic_library_action = action_config(
        action_name = ACTION_NAMES.cpp_link_dynamic_library,
        implies = [
            _FEATURE_NAMES.LINK.SHARED_FLAG,
            _FEATURE_NAMES.LINK.LINKSTAMPS,
            _FEATURE_NAMES.LINK.OUTPUT_EXECPATH_FLAGS,
            _FEATURE_NAMES.LINK.INPUT_PARAM_FLAGS,
            _FEATURE_NAMES.LINK.USER_LINK_FLAGS,
            _FEATURE_NAMES.LINK.LINKER_SUBSYSTEM_FLAG,
            _FEATURE_NAMES.LINK.LINKER_PARAM_FILE,
            _FEATURE_NAMES.COMMON.MSVC_ENV,
            _FEATURE_NAMES.OPTIONS.NO_STRIPPING,
            _FEATURE_NAMES.OPTIONS.HAS_CONFIGURED_LINKER_PATH,
            _FEATURE_NAMES.LINK.DEF_FILE,
        ],
        tools = [tool(path = llvm_link)],
    )

    action_configs = [
        assemble_action,
        preprocess_assemble_action,
        c_compile_action,
        linkstamp_compile_action,
        cpp_compile_action,
        cpp_link_executable_action,
        cpp_link_dynamic_library_action,
        cpp_link_nodeps_dynamic_library_action,
        cpp_link_static_library_action,
        # Lacking C++20 modules actions
        # WARNING: Inserting any other action here will cause incorrect flags to be injected in command line (like -Xlinker -rpath!)
    ]

    ###############################################################################################################################
    # Feature definitions
    ###############################################################################################################################
    msvc_link_env_feature = feature(
        name = _FEATURE_NAMES.COMMON.MSVC_LINK_ENV,
        env_sets = [
            env_set(
                actions = all_link_actions +
                            [ACTION_NAMES.cpp_link_static_library],
                env_entries = [env_entry(key = "LIB", value = ctx.attr.env_lib)],
            ),
        ],
    )

    shared_flag_feature = feature(
        name = _FEATURE_NAMES.LINK.SHARED_FLAG,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(flags = ["/DLL"])],
            ),
        ],
    )

    sysroot_feature = feature(
        name = _FEATURE_NAMES.COMMON.SYSROOT,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_module_deps_scanning,
                    ACTION_NAMES.cpp20_module_compile,
                    ACTION_NAMES.cpp20_module_codegen,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["--sysroot=%{sysroot}"],
                        iterate_over = "sysroot",
                        expand_if_available = "sysroot",
                    ),
                ],
            ),
        ],
    )

    unfiltered_compile_flags_feature = feature(
        name = _FEATURE_NAMES.COMPILE.UNFILTERED_COMPILE_FLAGS,
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_module_deps_scanning,
                    ACTION_NAMES.cpp20_module_compile,
                    ACTION_NAMES.cpp20_module_codegen,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["%{unfiltered_compile_flags}"],
                        iterate_over = "unfiltered_compile_flags",
                        expand_if_available = "unfiltered_compile_flags",
                    ),
                ],
            ),
        ],
    )

    archive_param_file_feature = feature(
        name = "archive_param_file",
        enabled = True,
    )

    compiler_param_file_feature = feature(
        name = "compiler_param_file",
        enabled = True,
    )

    copy_dynamic_libraries_to_binary_feature = feature(
        name = "copy_dynamic_libraries_to_binary",
    )

    input_param_flags_feature = feature(
        name = "input_param_flags",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["/IMPLIB:%{interface_library_output_path}"],
                        expand_if_available = "interface_library_output_path",
                    ),
                ],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["%{libopts}"],
                        iterate_over = "libopts",
                        expand_if_available = "libopts",
                    ),
                ],
            ),
            flag_set(
                actions = all_link_actions +
                          [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        iterate_over = "libraries_to_link",
                        flag_groups = [
                            flag_group(
                                iterate_over = "libraries_to_link.object_files",
                                flag_groups = [flag_group(flags = ["%{libraries_to_link.object_files}"])],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file_group",
                                ),
                            ),
                            flag_group(
                                flag_groups = [flag_group(flags = ["%{libraries_to_link.name}"])],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file",
                                ),
                            ),
                            flag_group(
                                flag_groups = [flag_group(flags = ["%{libraries_to_link.name}"])],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "interface_library",
                                ),
                            ),
                            flag_group(
                                flag_groups = [
                                    flag_group(
                                        flags = ["%{libraries_to_link.name}"],
                                        expand_if_false = "libraries_to_link.is_whole_archive",
                                    ),
                                    flag_group(
                                        flags = ["/WHOLEARCHIVE:%{libraries_to_link.name}"],
                                        expand_if_true = "libraries_to_link.is_whole_archive",
                                    ),
                                ],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "static_library",
                                ),
                            ),
                        ],
                        expand_if_available = "libraries_to_link",
                    ),
                ],
            ),
        ],
    )

    fastbuild_feature = feature( # to further enhance
        name = _FEATURE_NAMES.OPTIONS.FASTBUILD,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-O0", "-g"])],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["/DEBUG", "/INCREMENTAL:NO"],
                    ),
                ],
            ),
        ],
        implies = ["generate_pdb_file"],
    )

    user_compile_flags_feature = feature(
        name = _FEATURE_NAMES.COMPILE.USER_COMPILE_FLAGS,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_module_deps_scanning,
                    ACTION_NAMES.cpp20_module_compile,
                    ACTION_NAMES.cpp20_module_codegen,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["%{user_compile_flags}"],
                        iterate_over = "user_compile_flags",
                        expand_if_available = "user_compile_flags",
                    ),
                ],
            ),
        ],
    )

    archiver_flags_feature = feature(
        name = _FEATURE_NAMES.LINK.ARCHIVER_FLAGS,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = ["/OUT:%{output_execpath}"],
                        expand_if_available = "output_execpath",
                    ),
                    flag_group(
                        flags = ["%{user_archiver_flags}"],
                        iterate_over = "user_archiver_flags",
                        expand_if_available = "user_archiver_flags",
                    ),
                    flag_group(
                        flags = [""],
                    ),
                ],
            ),
        ],
    )

    default_link_flags_feature = feature(
        name = _FEATURE_NAMES.LINK.DEFAULT_LINK_FLAGS,
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = [
                            "/NOLOGO",
                            "/MACHINE:X64", # ABI fixed
                            "User32.lib", # LINK WINDOWS
                            "Ole32.lib",
                            "Kernel32.lib",
                        ],
                        #flags = [
                        #    "/DEBUG", # CTX.CONFIG?
                        #    "/SUBSYSTEM:WINDOWS",
                        #    "/MACHINE:X64", # MAYBE MACHINE?
                        #    "/INCREMENTAL:NO",
                        #    "User32.lib", # LINK WINDOWS?
                        #    "Ole32.lib",
                        #    "Kernel32.lib",
                        #] 
                    ),
                ],
            ),
        ],
    )

    static_link_msvcrt_feature = feature(
        name = _FEATURE_NAMES.LINK.STATIC_LINK_MSVCRT,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-D_MT"])],
                with_features = [with_feature_set(not_features = [_FEATURE_NAMES.OPTIONS.DBG])],
            ),
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-D_MT", "-D_DEBUG"])],
                with_features = [with_feature_set(features = [_FEATURE_NAMES.OPTIONS.DBG])],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["/DEFAULTLIB:libcmt.lib"])],
                with_features = [with_feature_set(not_features = [_FEATURE_NAMES.OPTIONS.DBG])],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["/DEFAULTLIB:libcmtd.lib"])],
                with_features = [with_feature_set(features = [_FEATURE_NAMES.OPTIONS.DBG])],
            ),
        ],
    )

    dynamic_link_msvcrt_feature = feature(
        name = _FEATURE_NAMES.LINK.DYNAMIC_LINK_MSVCRT,
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-D_MT", "-D_DLL"])],
                with_features = [with_feature_set(not_features = [_FEATURE_NAMES.OPTIONS.DBG, _FEATURE_NAMES.LINK.STATIC_LINK_MSVCRT])],
            ),
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-D_MD", "-D_DLL", "-D_DEBUG"])],
                with_features = [with_feature_set(features = [_FEATURE_NAMES.OPTIONS.DBG], not_features = [_FEATURE_NAMES.LINK.STATIC_LINK_MSVCRT])],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["/DEFAULTLIB:msvcrt.lib"])],
                with_features = [with_feature_set(not_features = [_FEATURE_NAMES.OPTIONS.DBG, _FEATURE_NAMES.LINK.STATIC_LINK_MSVCRT])],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["/DEFAULTLIB:msvcrtd.lib"])],
                with_features = [with_feature_set(features = [_FEATURE_NAMES.OPTIONS.DBG], not_features = [_FEATURE_NAMES.LINK.STATIC_LINK_MSVCRT])],
            ),
        ],
    )

    dbg_feature = feature(
        name = _FEATURE_NAMES.OPTIONS.DBG,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-O0", "-g"])],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["/DEBUG", "/INCREMENTAL:NO"],
                    ),
                ],
            ),
        ],
        implies = [_FEATURE_NAMES.OPTIONS.GENERATE_PDB_FILE],
    )

    opt_feature = feature(
        name = _FEATURE_NAMES.OPTIONS.OPT,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-O2"])],
            ),
        ],
        implies = [_FEATURE_NAMES.COMPILE.FRAME_POINTER],
    )

    supports_interface_shared_libraries_feature = feature(
        name = _FEATURE_NAMES.OPTIONS.SUPPORTS_INTERFACE_SHARED_LIBRARIES,
        enabled = True,
    )

    user_link_flags_feature = feature(
        name = _FEATURE_NAMES.LINK.USER_LINK_FLAGS,
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["%{user_link_flags}"],
                        iterate_over = "user_link_flags",
                        expand_if_available = "user_link_flags",
                    ),
                ],
            ),
        ],
    )

    default_cpp_std_feature = feature(
        name = _FEATURE_NAMES.COMPILE.DEFAULT_CPP_STD,
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_cpp_compile_actions,
                flag_groups = [
                    flag_group(
                        flags = ["-std=c++17"],
                    ),
                ],
            ),
        ],
    )

    default_compile_flags_feature = feature(
        name = _FEATURE_NAMES.COMPILE.DEFAULT_COMPILE_FLAGS,
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_module_deps_scanning,
                    ACTION_NAMES.cpp20_module_compile,
                    ACTION_NAMES.cpp20_module_codegen,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.clif_match,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-DAVK_ARCH_X86_64", "-DAVK_COMPILER_CLANG", "-DAVK_OS_WINDOWS",
                            "-DNOMINMAX", "-DUNICODE", "-DWIN32_LEAN_AND_MEAN",
                            "-fvisibility=hidden",
                            "-fno-fast-math",
                            "-faddrsig", "-fstrict-aliasing",
                            "-nogpuinc", "-nogpulib",
                            "-fstack-protector-all",
                            "-march=x86-64-v3",
                            "-fsanitize=address", "-fsanitize=undefined",
                            "-flto", "-fsanitize=cfi",
                            "-Wall", "-Wextra", "-pedantic",
                            #"/DNOMINMAX",
                            #"/D_WIN32_WINNT=0x0601",
                            #"/D_CRT_SECURE_NO_DEPRECATE",
                            #"/D_CRT_SECURE_NO_WARNINGS",
                            #"/bigobj",
                            #"/Zm500",
                            #"/EHsc",
                            #"/wd4351",
                            #"/wd4291",
                            #"/wd4250",
                            #"/wd4996",
                        ],
                    ),
                ],
            ),
        ],
    )

    msvc_compile_env_feature = feature(
        name = _FEATURE_NAMES.COMMON.MSVC_COMPILE_ENV,
        env_sets = [
            env_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_deps_scanning,
                    ACTION_NAMES.cpp20_module_compile,
                    ACTION_NAMES.cpp20_module_codegen,
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                ],
                env_entries = [env_entry(key = "INCLUDE", value = ctx.attr.env_include)],
            ),
        ],
    )

    preprocessor_defines_feature = feature(
        name = _FEATURE_NAMES.COMMON.PREPROCESSOR_DEFINES,
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_deps_scanning,
                    ACTION_NAMES.cpp20_module_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["/D%{preprocessor_defines}"],
                        iterate_over = "preprocessor_defines",
                    ),
                ],
            ),
        ],
    )

    generate_pdb_file_feature = feature(
        name = _FEATURE_NAMES.OPTIONS.GENERATE_PDB_FILE,
    )

    generate_linkmap_feature = feature(
        name = _FEATURE_NAMES.LINK.GENERATE_LINKMAP,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_executable,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "/MAP:%{output_execpath}.map",
                        ],
                        expand_if_available = "output_execpath",
                    ),
                ],
            ),
        ],
    )

    output_execpath_flags_feature = feature(
        name = _FEATURE_NAMES.LINK.OUTPUT_EXECPATH_FLAGS,
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["/OUT:%{output_execpath}"],
                        expand_if_available = "output_execpath",
                    ),
                ],
            ),
        ],
    )

    disable_assertions_feature = feature(
        name = _FEATURE_NAMES.COMPILE.DISABLE_ASSERTIONS,
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-DNDEBUG"])],
                with_features = [with_feature_set(features = [_FEATURE_NAMES.OPTIONS.OPT])],
            ),
        ],
    )

    has_configured_linker_path_feature = feature(name = _FEATURE_NAMES.OPTIONS.HAS_CONFIGURED_LINKER_PATH)

    supports_dynamic_linker_feature = feature(name = _FEATURE_NAMES.OPTIONS.SUPPORTS_DYNAMIC_LINKER, enabled = True)

    no_stripping_feature = feature(name = _FEATURE_NAMES.OPTIONS.NO_STRIPPING)

    linker_param_file_feature = feature(
        name = _FEATURE_NAMES.LINK.LINKER_PARAM_FILE,
        flag_sets = [
            flag_set(
                actions = all_link_actions +
                            [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = ["@%{linker_param_file}"],
                        expand_if_available = "linker_param_file",
                    ),
                ],
            ),
        ],
    )

    ignore_noisy_warnings_feature = feature(
        name = _FEATURE_NAMES.LINK.IGNORE_NOISY_WARNINGS,
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [flag_group(flags = ["/ignore:4221"])],
            ),
        ],
    )

    no_legacy_features_feature = feature(name = _FEATURE_NAMES.OPTIONS.NO_LEGACY_FEATURES)

    parse_showincludes_feature = feature(
        name = _FEATURE_NAMES.COMMON.PARSE_SHOWINCLUDES,
        enabled = False,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_deps_scanning,
                    ACTION_NAMES.cpp20_module_compile,
                ],
                flag_groups = [flag_group(flags = ["-M"])],
            ),
        ],
        env_sets = [
            env_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_header_parsing,
                ],
                # Force English (and thus a consistent locale) output so that Bazel can parse
                # the /showIncludes output without having to guess the encoding.
                env_entries = [env_entry(key = "VSLANG", value = "1033")],
            ),
        ],
    )

    # MSVC does not emit .d files.
    no_dotd_file_feature = feature(
        name = _FEATURE_NAMES.OPTIONS.NO_DOTD_FILE,
        enabled = True,
    )

    shorten_virtual_includes_feature = feature(
        name = _FEATURE_NAMES.OPTIONS.SHORTEN_VIRTUAL_INCLUDES,
        enabled = False,
    )

    treat_warnings_as_errors_feature = feature(
        name = _FEATURE_NAMES.COMMON.TREAT_WARNINGS_AS_ERRORS,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile] + all_link_actions,
                flag_groups = [flag_group(flags = ["-Werror"])],
            ),
        ],
    )

    windows_export_all_symbols_feature = feature(name = _FEATURE_NAMES.COMMON.WINDOWS_EXPORT_ALL_SYMBOLS)

    no_windows_export_all_symbols_feature = feature(name = _FEATURE_NAMES.COMMON.NO_WINDOWS_EXPORT_ALL_SYMBOLS)

    include_paths_feature = feature(
        name = _FEATURE_NAMES.COMPILE.INCLUDE_PATHS,
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-I%{quote_include_paths}"],
                        iterate_over = "quote_include_paths",
                    ),
                    flag_group(
                        flags = ["-I%{include_paths}"],
                        iterate_over = "include_paths",
                    ),
                    flag_group(
                        flags = ["-I%{system_include_paths}"],
                        iterate_over = "system_include_paths",
                    ),
                ],
            ),
        ],
    )

    external_include_paths_feature = feature(
        name = _FEATURE_NAMES.COMPILE.EXTERNAL_INCLUDE_PATHS,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_deps_scanning,
                    ACTION_NAMES.cpp20_module_compile,
                    ACTION_NAMES.clif_match,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-I%{external_include_paths}"],
                        iterate_over = "external_include_paths",
                        expand_if_available = "external_include_paths",
                    ),
                ],
            ),
        ],
    )

    linkstamps_feature = feature(
        name = _FEATURE_NAMES.LINK.LINKSTAMPS,
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["%{linkstamp_paths}"],
                        iterate_over = "linkstamp_paths",
                        expand_if_available = "linkstamp_paths",
                    ),
                ],
            ),
        ],
    )

    targets_windows_feature = feature(
        name = _FEATURE_NAMES.COMMON.TARGETS_WINDOWS,
        enabled = True,
        implies = [_FEATURE_NAMES.OPTIONS.COPY_DYNAMIC_LIBRARIES_TO_BINARY],
    )

    linker_subsystem_flag_feature = feature(
        name = _FEATURE_NAMES.LINK.LINKER_SUBSYSTEM_FLAG,
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["/SUBSYSTEM:WINDOWS"])],
            ),
        ],
    )

    frame_pointer_feature = feature( # maybe remove since x64 has lots of registers
        name = _FEATURE_NAMES.COMPILE.FRAME_POINTER,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-fomit-frame-pointer"])],
            ),
        ],
    )

    compiler_output_flags_feature = feature(
        name = _FEATURE_NAMES.COMPILE.COMPILER_OUTPUT_FLAGS,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.assemble],
                flag_groups = [
                    flag_group(
                        flag_groups = [
                            flag_group(
                                flags = ["-o%{output_file}", "-g"],
                                expand_if_available = "output_file",
                                expand_if_not_available = "output_assembly_file",
                            ),
                        ],
                        expand_if_not_available = "output_preprocess_file",
                    ),
                ],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_module_deps_scanning,
                    ACTION_NAMES.cpp20_module_compile,
                    ACTION_NAMES.cpp20_module_codegen,
                ],
                flag_groups = [
                    flag_group(
                        flag_groups = [
                            flag_group(
                                flags = ["-o%{output_file}"],
                                expand_if_not_available = "output_preprocess_file",
                            ),
                        ],
                        expand_if_available = "output_file",
                        expand_if_not_available = "output_assembly_file",
                    ),
                    flag_group(
                        flag_groups = [
                            flag_group(
                                flags = ["-S%{output_file}"],
                                expand_if_available = "output_assembly_file",
                            ),
                        ],
                        expand_if_available = "output_file",
                    ),
                    flag_group(
                        flag_groups = [
                            flag_group(
                                flags = ["-E", "-o%{output_file}"],
                                expand_if_available = "output_preprocess_file",
                            ),
                        ],
                        expand_if_available = "output_file",
                    ),
                ],
            ),
        ],
    )

    smaller_binary_feature = feature(
        name = _FEATURE_NAMES.COMPILE.SMALLER_BINARY,
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-ffunction-sections", "-fdata-sections"])],
                with_features = [with_feature_set(features = [_FEATURE_NAMES.OPTIONS.OPT])],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["/OPT:ICF", "/OPT:REF"])],
                with_features = [with_feature_set(features = [_FEATURE_NAMES.OPTIONS.OPT])],
            ),
        ],
    )

    remove_unreferenced_code_feature = feature(
        name = _FEATURE_NAMES.COMPILE.REMOVE_UNREFERENCED_CODE,
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = [""])],
            ),
        ],
    )

    compiler_input_flags_feature = feature(
        name = _FEATURE_NAMES.COMPILE.COMPILER_INPUT_FLAGS,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_module_deps_scanning,
                    ACTION_NAMES.cpp20_module_compile,
                    ACTION_NAMES.cpp20_module_codegen,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-c", "%{source_file}"],
                        expand_if_available = "source_file",
                    ),
                ],
            ),
        ],
    )

    def_file_feature = feature(
        name = _FEATURE_NAMES.LINK.DEF_FILE,
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["/DEF:%{def_file_path}", "/ignore:4070"],
                        expand_if_available = "def_file_path",
                    ),
                ],
            ),
        ],
    )

    msvc_env_feature = feature(
        name = _FEATURE_NAMES.COMMON.MSVC_ENV,
        env_sets = [
            env_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_deps_scanning,
                    ACTION_NAMES.cpp20_module_compile,
                    ACTION_NAMES.cpp20_module_codegen,
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_static_library,
                ],
                env_entries = [
                    env_entry(key = "PATH", value = ctx.attr.env_path),
                    env_entry(key = "TMP", value = ctx.attr.env_temp),
                    env_entry(key = "TEMP", value = ctx.attr.env_temp),
                ],
            ),
        ],
        implies = [_FEATURE_NAMES.COMMON.MSVC_COMPILE_ENV, _FEATURE_NAMES.COMMON.MSVC_LINK_ENV],
    )

    symbol_check_feature = feature(
        name = _FEATURE_NAMES.LINK.SYMBOL_CHECK,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [flag_group(flags = ["/WX:4006"])],
            ),
        ],
    )

    features = [
        no_legacy_features_feature,
        has_configured_linker_path_feature,
        no_stripping_feature,
        targets_windows_feature,
        copy_dynamic_libraries_to_binary_feature,
        default_cpp_std_feature,
        default_compile_flags_feature,
        msvc_env_feature,
        msvc_compile_env_feature,
        msvc_link_env_feature,
        include_paths_feature,
        external_include_paths_feature,
        preprocessor_defines_feature,
        parse_showincludes_feature,
        no_dotd_file_feature,
        shorten_virtual_includes_feature,
        generate_pdb_file_feature,
        generate_linkmap_feature,
        shared_flag_feature,
        linkstamps_feature,
        output_execpath_flags_feature,
        archiver_flags_feature,
        input_param_flags_feature,
        linker_subsystem_flag_feature,
        user_link_flags_feature,
        default_link_flags_feature,
        linker_param_file_feature,
        static_link_msvcrt_feature,
        dynamic_link_msvcrt_feature,
        dbg_feature,
        fastbuild_feature,
        opt_feature,
        frame_pointer_feature,
        disable_assertions_feature,
        treat_warnings_as_errors_feature,
        smaller_binary_feature,
        remove_unreferenced_code_feature,
        ignore_noisy_warnings_feature,
        user_compile_flags_feature,
        sysroot_feature,
        unfiltered_compile_flags_feature,
        archive_param_file_feature,
        compiler_param_file_feature,
        compiler_output_flags_feature,
        compiler_input_flags_feature,
        def_file_feature,
        windows_export_all_symbols_feature,
        no_windows_export_all_symbols_feature,
        supports_dynamic_linker_feature,
        supports_interface_shared_libraries_feature,
        symbol_check_feature,
    ]

    ###############################################################################################################################
    # Artifact name patterns (Necessary to generate appropriate extensions from target names!)
    ###############################################################################################################################
    artifact_name_patterns = [
        artifact_name_pattern(
            category_name = "object_file",
            prefix = "",
            extension = ".obj",
        ),
        artifact_name_pattern(
            category_name = "static_library",
            prefix = "",
            extension = ".lib",
        ),
        artifact_name_pattern(
            category_name = "alwayslink_static_library",
            prefix = "",
            extension = ".lo.lib",
        ),
        artifact_name_pattern(
            category_name = "executable",
            prefix = "",
            extension = ".exe",
        ),
        artifact_name_pattern(
            category_name = "dynamic_library",
            prefix = "",
            extension = ".dll",
        ),
        artifact_name_pattern(
            category_name = "interface_library",
            prefix = "",
            extension = ".if.lib",
        ),
    ]
    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        action_configs = action_configs,
        artifact_name_patterns = artifact_name_patterns,
        cxx_builtin_include_directories = ctx.attr.system_include_paths,
        toolchain_identifier = ctx.attr.toolchain_identifier,
        host_system_name = "x64_windows",
        target_system_name = "x64_windows",
        target_cpu = "x64_windows",
        target_libc = "",
        compiler = "clang-cl",
        tool_paths = [],
        make_variables = [],
    )


cc_toolchain_config = rule(
    implementation = _cc_toolchain_config_impl,
    attrs = {
        "toolchain_identifier": attr.string(mandatory = True, doc = "An identifier for this toolchain. Should be usable as a path component."),
        "llvm_bin_path": attr.string(default = "C:/Program Files/LLVM/bin", doc = "Path to the LLVM bin directory."),
        "system_include_paths": attr.string_list(default = [], doc = "List of system include paths."),
        "env_path": attr.string(mandatory = True, doc = "Value for the PATH environment variable."),
        "env_temp": attr.string(mandatory = True, doc = "Value for the TEMP and TMP environment variables."),
        "env_lib": attr.string(mandatory = True, doc = "Value for the LIB environment variable."),
        "env_include": attr.string(mandatory = True, doc = "Value for the INCLUDE environment variable."),
        "fastbuild_mode_debug_flag": attr.string(default = ""),
    },
    provides = [CcToolchainConfigInfo],
)