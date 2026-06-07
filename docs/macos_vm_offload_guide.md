# macOS VM Offload Guide for Cargo (Apple Silicon)

When running heavy Vulkan/MoltenVK test suites (`cargo run`, `cargo nextest run`) natively on an Apple Silicon Mac, high RAM consumption can occasionally exhaust the system memory. When macOS runs completely out of memory, it forcefully terminates the WindowServer, resulting in a sudden crash to the login screen and loss of unsaved work.

To prevent this, you can seamlessly **offload the execution** of your binaries to a lightweight macOS Virtual Machine using UTM. This way:
1. **Host (Your Mac):** Compiles the code quickly and manages the test runner UI (`nextest`).
2. **VM (UTM):** Executes the heavy GPU code in an isolated environment. If memory is exhausted, only the VM crashes—leaving your host perfectly safe.

### Why not Docker?
You cannot run MoltenVK inside Docker, even for headless compute tests without a GUI. Docker on a Mac runs a **Linux** Virtual Machine in the background. MoltenVK requires Apple's native **Metal** API and GPU drivers to function, which physically do not exist on Linux.

If you run Vulkan tests in a Docker container (like `Dockerfile.test.arm64`), they will actually execute using `lavapipe` (a Linux CPU software rasterizer). This works, but it runs on your CPU, heavily downgrades performance, and won't catch bugs specific to Apple Silicon GPU hardware. To test the true **Metal / MoltenVK** stack natively and safely, a macOS UTM VM is the only option.

---

## 1. UTM Virtual Machine Setup

We use Apple's native `Virtualization.framework` via UTM for near-native CPU performance and direct VirtioFS folder sharing.

1. **Install UTM** (https://macget.app/utm/ or `brew install --cask utm`).
2. **Create a new Virtual Machine:**
   - Click the **+** button -> **Virtualize**.
   - Select **macOS 15+**. (or any other version)
   - If UTM prompts you to update your Mac when you click Download, you can bypass this by downloading the IPSW for your **current** macOS version instead. You can find official Apple IPSW links for older versions at [Mr. Macintosh's IPSW Database](https://mrmacintosh.com/apple-silicon-m1-mac-macos-restore-ipsw-firmware-files-database/).
   - Once downloaded, click **Browse** in UTM and select the downloaded IPSW file.
   - Assign reasonable RAM (e.g., 8GB or 16GB) to ensure your host always has a buffer.
   - Assign at least 4 CPU cores.
   - Save and boot the VM. Go through the standard macOS installation process.

## 2. Directory Sharing (VirtioFS)

For the Cargo Runner offload to work seamlessly, the absolute path to your `Aether-Vk` project on the host **must perfectly match** the path inside the VM.

1. Shut down the VM.
2. Right-click the VM in UTM -> **Edit**.
3. Go to **Sharing**. Ensure **Directory Share Mode** is set to **VirtioFS**.
4. Click **+** to add a shared directory.
   - Choose the root of the drive where your code lives (e.g., `/Volumes/ExtData`).
5. Boot the VM.
6. Open Terminal in the VM and mount or link it so the absolute path matches.
   - If your host path is `/Volumes/ExtData/alessioext/Dev/Aether-Vk`, ensure that exact path is accessible in the VM.
   - UTM usually auto-mounts VirtioFS shares in `/Volumes/MySharedDir`. You may need to create a symlink:
     `sudo ln -s /Volumes/MySharedDir /Volumes/ExtData`

## 3. SSH and SDK Configuration in the VM

To allow Cargo to send execution commands to the VM:

1. **Enable SSH in the VM:**
   - Open **System Settings** -> **General** -> **Sharing**.
   - Turn on **Remote Login** (Allow access for all users).
   - Note the VM's hostname (e.g., `aether-vm.local`) or IP address.

2. **Set up Passwordless SSH (On Host):**
   Run this on your host Mac to copy your SSH keys to the VM so Cargo won't prompt for a password:
   ```bash
   # Replace with your VM's hostname and user
   ssh-copy-id username@aether-vm.local
   ```

3. **Install Vulkan SDK (Inside the VM):**
   SSH into your VM and navigate to the shared Aether-Vk directory. Run the helper script to extract the Vulkan SDK:
   ```bash
   ssh username@aether-vm.local
   cd /Volumes/ExtData/alessioext/Dev/Aether-Vk
   ./scripts/setup-vm-vulkan.sh
   exit
   ```

---

## 4. Enabling the Cargo VM Runner

Cargo has a built-in feature to intercept execution using a `runner`. We have provided a script at `scripts/vm-runner.sh`.

### Temporary Activation
If you just want to test it out temporarily, prepend the environment variables to your command:

```bash
# Set your VM details (add to your .zshrc for convenience)
export AETHER_VM_HOST="aether-vm.local"
export AETHER_VM_USER="your_vm_username"

# Run a test using the custom runner
CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER="scripts/vm-runner.sh" cargo nextest run test_energy_conservation_bounce
```

### Permanent Activation
To make this permanent for your Aether-Vk workspace, create or append to `.cargo/config.toml` in your project root:

```bash
mkdir -p .cargo
cat <<EOF >> .cargo/config.toml
[target.aarch64-apple-darwin]
runner = "scripts/vm-runner.sh"
EOF
```

## How It Works

Once the runner is configured, your workflow remains completely unchanged:
- You type `cargo run` or `cargo nextest run`.
- **Cargo builds the Mach-O binaries natively** on your Host M4 CPU (fast, multi-core, no RAM issues).
- Instead of executing the binary locally, Cargo invokes `scripts/vm-runner.sh /absolute/path/to/binary`.
- The runner automatically SSHes into the VM, injects the `VULKAN_SDK` environment variables, and executes the binary in the VM's shared directory.
- Standard output and error stream back to your host terminal exactly as if it ran locally. `cargo nextest` parallelization UI works flawlessly!

---

## 5. Xcode Integration (LLDB & Metal Frame Capture)

If you are using Xcode to debug your Rust executables (e.g., via a project generated by `cargo-xcode`), **do not use the `vm-runner.sh` script**. Wrapping your executable in a shell script will cause Xcode's `debugserver` to attach to `bash` instead of your Rust binary, breaking LLDB and Metal Frame Capture.

Instead, use Xcode's native **"Remote Mac"** feature, which perfectly preserves LLDB syntax highlighting, breakpoints, variable inspection, and Metal Frame Capture over the network.

### 1. Pair the UTM VM with Xcode
1. Ensure your UTM VM is booted and connected to your network.
2. In Xcode on your host Mac, go to **Window -> Devices and Simulators**.
3. Select the **Mac** tab and click **Add Remote Mac**.
4. Follow the prompts to pair your UTM VM (it will ask for the VM's SSH credentials).

### 2. Configure the Xcode Run Scheme
You need to tell Xcode where to find the Vulkan SDK inside the VM and set the working directory (so your app can find local assets).

1. In Xcode, click on your target scheme (e.g., `spawn_comet_debug`) and select **Edit Scheme...**
2. In the **Run** phase -> **Options** tab:
   - Check **"Use custom working directory"**.
   - Set it to the absolute path of your VirtioFS shared directory (e.g., `/Volumes/ExtData/alessioext/Dev/Aether-Vk`).
3. In the **Run** phase -> **Arguments** tab, under **Environment Variables**, add the Vulkan SDK variables pointing to the VM's paths:
   - `VULKAN_SDK` = `/Users/your_vm_username/vulkan_sdk/macOS`
   - `VK_ICD_FILENAMES` = `/Users/your_vm_username/vulkan_sdk/macOS/share/vulkan/icd.d/MoltenVK_icd.json`
   - `DYLD_LIBRARY_PATH` = `/Users/your_vm_username/vulkan_sdk/macOS/lib`

### 3. Run and Debug
1. At the top of Xcode where you select the device simulator, choose your **Remote Mac (UTM VM)** instead of "My Mac".
2. Click the **Run** (Play) button.

Xcode will natively build the binary on your fast M4 host, automatically copy it over the network to the UTM VM, and attach LLDB natively. You can now use Xcode's visual debugger and Metal Frame Capture completely isolated from the host WindowServer!

---

## 6. .NET UI App Integration (Avalonia)

If you need to test the front-end C# Avalonia application (`aethervk.ui-app`), you must ensure that the .NET runtime can locate the Rust `cdylib` (`libaethervk_core_cdylib.dylib`) inside the VM.

We provide a dedicated runner script that:
1. Compiles the Rust core library natively on your host.
2. Compiles the .NET C# app natively on your host.
3. SSHes into the VM and invokes the .NET application, passing the correct `DYLD_LIBRARY_PATH` so it can find both MoltenVK and the Rust `cdylib`.

### Running the UI App
Simply execute the provided runner script from your terminal:

```bash
# Debug build (default)
./scripts/vm-dotnet-runner.sh

# Release build
./scripts/vm-dotnet-runner.sh -c Release
```

**Note on macOS GUI Apps via SSH:**
When the script runs the .NET application via SSH, the Avalonia GUI window will actually **appear on the VM's desktop display**. You should have the UTM window open or use macOS Screen Sharing to interact with the UI.