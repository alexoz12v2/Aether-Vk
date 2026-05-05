# Packaging Guide for Aether-Vk

This guide explains how to package the Aether-Vk application into distributable formats for macOS, Windows, and Linux. The application relies on a bundled native runtime library and OS-specific dependencies (such as Vulkan loaders and MoltenVK on macOS). By using target-specific deployments, we ensure that each bundle only contains the required native assets for its respective platform.

## Packaging Strategy

The strategy utilizes `dotnet publish` with specific Runtime Identifiers (RIDs), like `osx-arm64` or `win-x64`, ensuring a self-contained output. Because `NativeRuntimeService.cs` loads assets and libraries relative to the executable (`AppDomain.CurrentDomain.BaseDirectory`), the native files (e.g., `aethervk_core_cdylib` and the `vulkan/` directory) are preserved alongside the binary in the output directories.

For macOS specifically, we structure the output into a `.app` bundle (e.g., `AetherVk.app`). The Vulkan explicit layers and `MoltenVK_icd.json` will reside in `Contents/MacOS/vulkan/` alongside the executable, preserving the internal relative paths needed by the native runtime loader.

## Scripts

The `scripts/` directory contains packaging scripts for macOS, Windows, and Linux, along with scripts to validate the resulting package structures.

### macOS
**Script:** `scripts/package_macos.sh`

This bash script builds for a specified architecture (defaults to `osx-arm64`) and assembles the `.app` bundle.

```bash
# Build for Apple Silicon (arm64)
./scripts/package_macos.sh osx-arm64

# Build for Intel Mac (x64)
./scripts/package_macos.sh osx-x64
```
**Output:** `bin/publish/AetherVk.app`

### Windows
**Scripts:** `scripts/package_windows.ps1` and `scripts/package_windows_msix.ps1`

- `package_windows.ps1` builds for a specified architecture (defaults to `win-x64`) and archives the build into a standard ZIP file.
- `package_windows_msix.ps1` generates an `.msix` package. This requires `MakeAppx.exe` (included in the Windows SDK). The script will automatically try to find `MakeAppx.exe` and pack the `.msix` using an auto-generated `AppxManifest.xml`. Ensure you have run `dotnet publish` or `package_windows.ps1` beforehand to populate the publish directory.

```powershell
# Build for x64 Windows as a ZIP
.\scripts\package_windows.ps1 -Arch win-x64

# Package the built output into an MSIX (requires Windows SDK MakeAppx.exe)
.\scripts\package_windows_msix.ps1 -Arch x64
```
**Output:** `bin\publish\AetherVk_win-x64.zip` and `bin\publish\AetherVk_x64.msix`

### Linux
**Scripts:** `scripts/package_linux.sh` and `scripts/package_linux_deb.sh`

- `package_linux.sh` builds for a specified architecture (defaults to `linux-x64`) and produces a `.tar.gz` tarball.
- `package_linux_deb.sh` converts the output into a Debian package (`.deb`). This relies on `dpkg-deb` which is standard on Debian/Ubuntu systems. Ensure you have the `dotnet publish` directory ready before running this script.

```bash
# Build for x64 Linux (tar.gz)
./scripts/package_linux.sh linux-x64

# Package as a Debian .deb file
./scripts/package_linux_deb.sh linux-x64
```
**Output:** `bin/publish/AetherVk_linux-x64.tar.gz` and `bin/publish/AetherVk_1.0.0_amd64.deb`

## Validation

After packaging, you can ensure that the native structure (Vulkan layers on macOS, core native libraries across all OSes) are safely enclosed within the generated formats using the validation scripts:

```bash
# Validate macOS Application Bundle
./scripts/validate_package.sh macos bin/publish/AetherVk.app

# Validate Linux tarball or deb
./scripts/validate_package.sh linux bin/publish/AetherVk_linux-x64.tar.gz
./scripts/validate_package.sh linux bin/publish/AetherVk_1.0.0_amd64.deb
```

```powershell
# Validate Windows ZIP
.\scripts\validate_package.ps1 -Type windows -Path bin\publish\AetherVk_win-x64.zip

# Validate Windows MSIX
.\scripts\validate_package.ps1 -Type msix -Path bin\publish\AetherVk_x64.msix
```

## Important Note on macOS Vulkan Layers
When packaging for macOS, the MSBuild targets defined in `AetherVk.Logic.csproj` automatically copy the macOS Vulkan explicit validation layers and `MoltenVK_icd.json` (found via the `VULKAN_SDK` environment variable) to the output directory.

If `VULKAN_SDK` is not defined during the execution of `dotnet publish`, the validation layers will **not** be included in the bundle. The `validate_package.sh` script verifies the existence of these macOS layers to alert you if they are missing.