"""Simple Windows DLL library rule from """

def windows_dll_library(
        name,
        srcs = [],
        deps = [],
        hdrs = [],
        visibility = None,
        **kwargs):
    """A simple windows_dll_library rule for builing a DLL Windows."""

    dll_name = name + ".dll"
    import_lib_name = name + "_import_lib"
    import_target_name = name + "_dll_import"

    # Build the shared library
    native.cc_binary(
        name = dll_name,
        srcs = srcs + hdrs,
        deps = deps,
        linkshared = 1,
        **kwargs
    )

    # Get the import library for the dll
    native.filegroup(
        name = import_lib_name,
        srcs = [":" + dll_name],
        output_group = "interface_library",
    )

    # Because we cannot directly depend on cc_binary from other cc rules in deps attribute,
    # we use cc_import as a bridge to depend on the dll.
    native.cc_import(
        name = import_target_name,
        interface_library = ":" + import_lib_name,
        shared_library = ":" + dll_name,
    )

    # Create a new cc_library to also include the headers needed for the shared library
    native.cc_library(
        name = name,
        hdrs = hdrs,
        visibility = visibility,
        deps = deps + [
            ":" + import_target_name,
        ],
    )

def _cc_binary_with_dll_impl(ctx):
    binary = ctx.attr.binary[DefaultInfo].files_to_run.executable
    dlls = []
    for f in ctx.files.data:
        if f.basename.endswith(".dll"):
            dlls.append(f)

    # Original cc_binary output
    orig_binary = ctx.attr.binary[DefaultInfo].files_to_run.executable

    # Declare a new output executable with .exe suffix
    exe_out = ctx.actions.declare_file(orig_binary.basename + ".exe")

    out_files = [exe_out]  # start with the .exe output

    # Copy the original binary into the new .exe
    ctx.actions.run_shell(
        inputs=[orig_binary],
        outputs=[exe_out],
        command="cp %s %s" % (orig_binary.path, exe_out.path),
    )

    for dll in dlls:
        out_dll = ctx.actions.declare_file(dll.basename)
        ctx.actions.run_shell(
            inputs = [dll],
            outputs = [out_dll],
            command = "cp %s %s" % (dll.path, out_dll.path),
        )
        out_files.append(out_dll)

    return [
        DefaultInfo(
            files = depset(out_files),
            executable = exe_out,  # declare executable
        )
    ]

cc_binary_with_dll = rule(
    implementation = _cc_binary_with_dll_impl,
    attrs = {
        "binary": attr.label(cfg = "target", executable = True),
        "data": attr.label_list(allow_files = True),
    },
    executable = True,
)