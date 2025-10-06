"""Module extension for Windows C++ toolchain configuration."""

#load("@bazel_skylib//lib:paths.bzl", "paths")
load(":repo.bzl", "repo_windows_cc_config")

# TODO manage BAZEL_VC and BAZEL_VS environment variables
def _windows_cc_impl(module_ctx):
    """Configure the Windows C++ toolchain."""

    # this is the os.name Java system property, lower case
    if not module_ctx.os.name.startswith("windows"):
        repo_windows_cc_config(
            name = "cc_windows_x86_64_clang_config",
            msvc_include_path = "",
            msvc_mfc_include_path = "",
            llvm_include_path = "",
            llvm_share_path = "",
            llvm_sanitizer_path = "",
            windows_sdk_base_path = "",
            windows_sdk_version = "",
            llvm_bin_path = "",
            msvc_bin_path = "",
        )
        return  # No-op on non-Windows platforms

    # Ensure that MSVC is available and compute its path
    if module_ctx.which("vswhere.exe") == None:
        if "PROGRAMFILES(X86)" not in module_ctx.os.environ:
            fail("PROGRAMFILES(X86) environment variable is not set")
        program_files_x86 = module_ctx.os.environ["PROGRAMFILES(X86)"]
        vswhere = module_ctx.path(program_files_x86 + "/Microsoft Visual Studio/Installer/vswhere.exe")
        if not vswhere.exists or vswhere.is_dir:
            fail("Could not find vswhere.exe, please install Visual Studio or set PATH environment variable to find it")
    else:
        vswhere = module_ctx.which("vswhere.exe")

    vswhere_res = module_ctx.execute([vswhere, "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Llvm.Clang", "-property", "installationPath"], quiet = False)
    if vswhere_res.return_code != 0:
        fail("Could not find a Visual Studio installation with the LLVM/Clang toolchain. Please install Visual Studio with the 'Desktop development with C++' workload and the 'C++ Clang tools for Windows' component.")

    visual_studio_path = vswhere_res.stdout.strip()
    msvc_parent = module_ctx.path(visual_studio_path + "/VC/Tools/MSVC")
    msvc_ver_res = module_ctx.execute(["powershell.exe", "-Command", "(gci '{path}' | select name | sort -Descending)[0].Name".format(path = msvc_parent)], quiet = True)
    if msvc_ver_res.return_code != 0:
        fail("Could not find MSVC tools in the Visual Studio installation at %s with base path %s" % (visual_studio_path, msvc_parent))

    msvc_version = msvc_ver_res.stdout.strip()
    export_include_msvc = [
        visual_studio_path + "/VC/Tools/MSVC/" + msvc_version + "/include", visual_studio_path + "/VC/Tools/MSVC/" + msvc_version + "/ATLMFC/include"]

    msvc_bin_path = visual_studio_path + "/VC/Tools/MSVC/" + msvc_version + "/bin/HostX64/x64"

    # Now try to find LLVM/Clang tools
    if module_ctx.which("clang++.exe") != None:
        module_ctx.execute(["powershell", "-Command", "'Write-Host Setup LLVM From Environment Variables'"], quiet = False)

        # {path}/bin/clang++.exe, so get parent twice to get {path}
        clangxx = module_ctx.which("clang++.exe")
        llvm_path = module_ctx.path(str(clangxx)[0:str(clangxx).find("bin") - 1])  # remove path separator
    else:
        llvm_path = module_ctx.path("C:/Program Files/LLVM")
        if not llvm_path.exists or not llvm_path.is_dir:
            module_ctx.execute(["powershell", "-Command", "'Write-Host LLVM Not found in program files, trying in VS Installation'"], quiet = False)
            llvm_path = module_ctx.path(visual_studio_path + "/VC/Tools/LLvm")
            if not llvm_path.exists or not llvm_path.is_dir:
                fail("Could not find LLVM/Clang tools, please install them")

    llvm_res = module_ctx.execute(["powershell", "-Command", "(gci '{llvm_path}/lib/clang' | sort -Descending)[0].FullName.Replace(\"\\\", \"/\") + \"/\"".format(llvm_path = llvm_path)], quiet = False)
    llvm_base = llvm_res.stdout.strip()
    export_llvm_include_include = llvm_base + "include"
    export_llvm_include_share = llvm_base + "share"
    export_llvm_sanitizer_path = llvm_base + "lib/windows"

    # Find Windows SDK
    windows_sdk_res = module_ctx.execute(["powershell", "-Command", "(Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots' -Name KitsRoot10).KitsRoot10.Replace('\\', '/')"])
    if windows_sdk_res.return_code != 0:
        fail("Could not find Windows SDK, please install it")
    windows_sdk_path = windows_sdk_res.stdout.strip()  # should end with a slash
    windows_sdk_version_res = module_ctx.execute(["powershell", "-Command", "(Get-ChildItem 'HKLM:\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots' | Sort-Object -Descending)[0].Name.Split('\\')[-1]"])
    if windows_sdk_version_res.return_code != 0:
        fail("Could not find Windows SDK version, please install it")

    windows_sdk_version = windows_sdk_version_res.stdout.strip()
    if not module_ctx.path(windows_sdk_path + "include/" + windows_sdk_version).exists:
        fail("Could not find Windows SDK version %s in %s, please install it" % (windows_sdk_version, windows_sdk_path))

    # now export everything into a repo by calling a repo rule
    repo_windows_cc_config(
        name = "cc_windows_x86_64_clang_config",
        msvc_include_path = export_include_msvc[0],
        msvc_mfc_include_path = export_include_msvc[1],
        llvm_include_path = export_llvm_include_include,
        llvm_share_path = export_llvm_include_share,
        llvm_sanitizer_path = export_llvm_sanitizer_path,
        windows_sdk_base_path = windows_sdk_path,
        windows_sdk_version = windows_sdk_version,
        llvm_bin_path = str(llvm_path) + "/bin",
        msvc_bin_path = msvc_bin_path,
    )


_exec = tag_class(attrs = {})
windows_cc = module_extension(
    implementation = _windows_cc_impl,
    tag_classes = {"exec": _exec},
    doc = "Module extension for Windows C++ toolchain configuration.",
)
