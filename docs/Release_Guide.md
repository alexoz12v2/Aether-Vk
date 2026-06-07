# Release Guide for Aether-Vk

This guide explains how to create, tag, and publish new releases for Aether-Vk. The release process relies on GitHub Actions to automatically package and attach built artifacts to GitHub Releases. The local scripts handle orchestrating this process by talking to the GitHub API (`gh`).

## Prerequisites

1. **GitHub CLI (`gh`)**: You must have `gh` installed and authenticated (`gh auth login`) to create releases.
2. **Passing CI**: The commit you want to release *must* have successfully passed the CI testing pipeline.

## Official vs Side-by-Side (SxS) Builds

Aether-Vk supports two primary release channels:

*   **Official Releases (`official`)**: These are standard, stable releases (e.g., `v1.0.0`).
*   **Side-by-Side / Development Releases (`sxs`)**: These are pre-release builds meant for testing alongside an official build. They append `-sxs` to the version tag (e.g., `v1.0.0-sxs`) and are marked as "Pre-release" on GitHub.

You can release either an official build, an SxS build, or both (as entirely separate releases) for any given commit.

## Creating a Release

The process is managed by `scripts/create_release.sh` (macOS/Linux) or `scripts/create_release.ps1` (Windows).

### Step 1: Ensure your target branch is up-to-date
The release scripts fetch the latest commit hash directly from the remote GitHub branch (default: `main`). Make sure the commit you want to release has been pushed and has successfully passed CI.

### Step 2: Run the Release Script
Run the script specifying the action (`create`), the version number, the build type (`official` or `sxs`), and the target branch (`main`).

**For an Official Release:**
```bash
# macOS / Linux
./scripts/create_release.sh create 1.0.0 official main
```
```powershell
# Windows
.\scripts\create_release.ps1 -Action create -Version 1.0.0 -Type official -TargetBranch main
```

**For an SxS (Development) Release:**
```bash
# macOS / Linux
./scripts/create_release.sh create 1.0.0 sxs main
```
```powershell
# Windows
.\scripts\create_release.ps1 -Action create -Version 1.0.0 -Type sxs -TargetBranch main
```

### What happens behind the scenes:
1. The script queries the GitHub API for the latest commit on the specified branch.
2. It verifies that the CI workflow for that commit has a `success` status.
3. It creates a GitHub Release targeting that exact commit hash.
4. *For SxS builds*, it automatically appends `-sxs` to the version and marks it as a pre-release.
5. Once the GitHub Release is created, the GitHub Actions release workflow is automatically triggered to compile, package, and upload the final binaries directly to the release page.

## Manually Building and Uploading Artifacts (Optional)

If the automated GitHub Actions pipeline does not attach all the necessary artifacts, or if you need to build and attach a local artifact manually, you can use the packaging scripts to generate the binaries and then upload them.

**Note:** The version tag must perfectly match the existing GitHub release (e.g., `1.0.0` or `1.0.0-sxs`).

### 1. Build the Packages

First, use the platform-specific packaging scripts to compile and bundle the application for your target host.

**For macOS (creates an `.app` bundle):**
```bash
./scripts/package_macos.sh osx-arm64
# Output: bin/publish/AetherVk.app
```

**For Linux (creates a `.tar.gz` and optionally `.deb`):**
```bash
./scripts/package_linux.sh linux-x64
# Output: bin/publish/AetherVk_linux-x64.tar.gz

./scripts/package_linux_deb.sh linux-x64
# Output: bin/publish/AetherVk_1.0.0_amd64.deb
```

**For Windows (creates `.zip` and `.msix`):**
```powershell
.\scripts\package_windows.ps1 -Arch win-x64
# Output: bin\publish\AetherVk_win-x64.zip

.\scripts\package_windows_msix.ps1 -Arch x64
# Output: bin\publish\AetherVk_x64.msix
```

*(For more details on packaging specifics, refer to [Packaging.md](Packaging.md)).*

### 2. Upload the Artifacts

Once you have generated the artifacts locally, use the `upload` action to attach them to your GitHub release.

**macOS / Linux:**
```bash
./scripts/create_release.sh upload 1.0.0 ./bin/publish/AetherVk_linux-x64.tar.gz
```

**Windows:**
```powershell
.\scripts\create_release.ps1 -Action upload -Version 1.0.0 -File .\bin\publish\AetherVk_win-x64.zip
```

## Deleting or Undoing a Release

If you made a mistake (e.g., triggered a release on the wrong commit or with the wrong version), you can easily delete the release and its associated git tag so that you can recreate it properly.

Run the following command using the GitHub CLI, replacing `v1.0.0-sxs` with the target version:

```bash
gh release delete v1.0.0-sxs --cleanup-tag -y
```

*   `--cleanup-tag`: Ensures the git tag is deleted from the remote repository as well, preventing tag collision when you attempt to redo the release.
*   `-y`: Skips the confirmation prompt.

## An example to test in local with a VM exposed with SSH

After `brew` installing `sshpass` to insert local VM user pass automatically

```bash
sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no -p 2222 alessio@localhost "cd /d Z:\ && powershell -ExecutionPolicy Bypass -File .\scripts\package_windows.ps1 -Arch win-arm64 && powershell -ExecutionPolicy Bypass -File .\scripts\package_windows_msix.ps1 -
Arch arm64"
```

Note: actually, in shared folders, at least in UTM, you might run into troubles unless you do the cargo build in an NTFS partition
inside the internal drive of the VM, therefore you'd need to `robocopy` the rust workspace to build the cdylib first, and then
complete the build in the shared folder

```powershell
# example: shared folder here is Z:\
robocopy Z:\ C:\Aether-Vk /E /XD target .git .aider* .agents .antigravitycli .venv bin obj /XF *.tar.xz *.log *.txt /R:0 /W:0
```
