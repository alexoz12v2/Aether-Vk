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
   - *Note: You are completely correct that UTM's GUI only lets you populate a table of directories without customizing the mount point. This is a known UTM limitation.*
5. Boot the VM.
6. Open Terminal inside the newly booted VM to fix the mount point manually.
   - Because UTM auto-mounts your shared folders to a generic path (usually `/Volumes/MySharedDir` or `/Volumes/Mac`), you must use a symlink so that the absolute path in the VM perfectly matches your host.
   - Run this inside the VM's terminal:
     ```bash
     # Example: If your code is on the host at /Volumes/ExtData/...
     # First ensure the base directory exists (if it's not the main drive)
     sudo mkdir -p /Volumes/ExtData

     # Symlink the generic UTM mount to your required absolute path
     sudo ln -s /Volumes/MySharedDir /Volumes/ExtData
     ```

## 3. SSH and SDK Configuration in the VM

To allow Cargo to send execution commands to the VM:

1. **Enable SSH in the VM:**
   - Open **System Settings** -> **General** -> **Sharing**.
   - Turn on **Remote Login** (Allow access for all users).
   - Note the VM's hostname (e.g., `aether-vm.local`) or IP address.

2. **Set up Passwordless SSH (On Host):**
   Run this on your host Mac to copy your SSH keys to the VM so Cargo won't prompt for a password. *(Note: If you get an `ERROR: No identities found`, it means you don't have an SSH key yet. Run `ssh-keygen -t ed25519` and press Enter for all prompts to create one first!)*
   ```bash
   # Replace with your VM's hostname and user
   ssh-copy-id username@aether-vm.local
   ```

3. **Install Vulkan SDK (Inside the VM):**
   SSH into your VM and navigate to the shared Aether-Vk directory. Run the helper script to download and install the Vulkan SDK:
   ```bash
   ssh username@aether-vm.local
   cd /Volumes/ExtData/alessioext/Dev/Aether-Vk
   ./scripts/setup-vm-vulkan.sh

   # batch Mode
   ssh -o BatchMode=yes alessio@aether-vm.local "bash /Volumes/ExtData/alessioext/Dev/Aether-Vk/scripts/setup-vm-vulkan.sh"
   ```
   After installation, the script automatically generates an environment setup file. If you plan to run commands manually inside the VM terminal, you should source this file to load `VULKAN_SDK` and its related library paths:
   ```bash
   source ~/vulkan_sdk/setup-env.sh

   # Optional: Add it to your profile so it loads automatically on every login
   echo "source ~/vulkan_sdk/setup-env.sh" >> ~/.zshrc
   exit
   ```

---

## 4. Enabling the Cargo VM Runner

Cargo has a built-in feature to intercept execution using a `runner`. We have provided a script at `scripts/vm-runner.sh`.

### Temporary Activation (Terminal)
If you just want to activate offloading for your current terminal session, we have provided a handy script that sets up all the variables for you:

```bash
# Usage: source scripts/macos_vm_env.sh [vm_username] [vm_hostname]
# Defaults to alessio / aether-vm.local if no arguments are given
source scripts/macos_vm_env.sh

# Now Cargo will automatically use the VM offload runner!
cargo nextest run --features collisions test_energy_conservation_bounce
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

### Speeding Up Tests via Concurrency Limits

Because `vm-runner.sh` copies the binary via `virtiofs` and spawns concurrent SSH connections per test, `nextest`'s default behavior of aggressively spawning processes based on your CPU core count can easily overwhelm the VM's I/O and RAM (especially with the heavy Validation Layer active).

To prevent the VM from thrashing and significantly speed up the overall test suite execution, limit the concurrent processes using the `-j` (or `--test-threads`) flag:

```bash
cargo nextest run -j 4
```
*(You can adjust `4` to find the sweet spot for your VM's core configuration, e.g., 2, 4, or 6).*

If you prefer to make this permanent without typing it every time, you can set the environment variable locally on your host:

```bash
export NEXTEST_TEST_THREADS=4
```

---

## 5. Remote Debugging (Terminal & Xcode)

Because the test binaries execute physically on the VM, standard host-side `cargo run` commands won't attach the debugger to the remote process. However, debugging is absolutely supported!

### 5.1 Terminal Debugging (`rust-lldb`)
If you just want to drop into a terminal debugger for a specific binary, you can simply SSH into the VM, source the environment, and run `lldb` manually:
```bash
ssh -t alessio@aether-vm.local "source ~/vulkan_sdk/setup-env.sh && rust-lldb -- /Volumes/ExtData/.../target/debug/deps/your_test_binary"
```
*(Note: Because this is an Apple platform, use `rust-lldb` or `lldb` rather than `gdb`).*

### 5.2 Xcode Integration (Recommended)
If you prefer visual debugging and want **Metal Frame Capture**, Xcode handles remote debugging flawlessly. **Do not use the `vm-runner.sh` script for this**. Wrapping your executable in a shell script will cause Xcode's `debugserver` to attach to `bash` instead of your Rust binary, breaking LLDB and Metal Frame Capture.

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

--

## Running Windows on UTM

To connect to your UTM Windows VM from your Mac host via SSH, the easiest and most reliable method is to use Port Forwarding. This keeps the VM isolated behind UTM's shared network but opens a specific tunnel just for SSH.
Here is the step-by-step guide to setting this up.

### Step 1: Configure Port Forwarding in UTM

You must configure this while the virtual machine is powered off. [DEV Community](https://dev.to/smyekh/completing-your-local-oci-lab-a-guide-to-port-forwarding-in-utm-hgp#:~:text=Make%20sure%20the%20VM%20is,while%20the%20VM%20is%20running.)

1. Open UTM and select your Windows 11 VM from the left sidebar.
2. Click the Edit button (the sliders icon) in the top right corner to open the VM settings.
3. In the settings menu, click on Network.
4. Ensure the Network Mode is set to "Emulated VLAN"
5. Look for the Port Forwarding section at the bottom and click New....
6. Fill out the rule with the following details:
7. Protocol: TCP
8. Guest Address: (Leave blank)
9. Guest Port: 22 (The standard SSH port inside Windows)
10. Host Address: (Leave blank)
11. Host Port: 2222 (A custom port on your Mac that will forward to the VM)
12. Click Save to apply the settings.

### Step 2: Enable SSH inside Windows 11

By default, Windows 11 has the OpenSSH client installed, but the OpenSSH Server is turned off.

1. Start your Windows 11 VM and log in.
2. Open Settings (Win + I) and go to Apps > Optional features
   (or Installed apps -> Optional features or System -> Optional features depending on your Windows build). [IONOS](https://www.ionos.com/digitalguide/server/configuration/windows-11-ssh/#:~:text=own%20OpenSSH%20server.-,Under%20Optional%20Features%20in%20Windows%2011%2C%20you,install%20your%20own%20OpenSSH%20server.)
   - if can't find it, then powershell
   ```powershell
   # install it
   Add-WindowsCapability -Online -Name OpenSSH.Server
   # start it
   Start-Service sshd
   # to make it start automatically
   Set-Service -Name sshd -StartupType 'Automatic'
   ```
4. Click View features next to "Add an optional feature".
5. Search for OpenSSH Server, check the box, and click Next then Install.
6. Once installed, open the Start Menu, search for Services, and run it as Administrator.
7. Scroll down to find OpenSSH SSH Server.
8. Right-click it, select Properties, change the Startup type to Automatic, and click the Start button. Click Apply and OK.

⚠️ Note on Firewalls: Windows usually configures the firewall rule automatically when you install the optional feature.
If you can't connect later, open
Windows Defender Firewall with Advanced Security and ensure the inbound rule for "OpenSSH SSH Server (TCP-In)" is enabled.

```powershell
if (!(Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue | Select-Object Name, Enabled)) {
    Write-Output "Firewall Rule 'OpenSSH-Server-In-TCP' does not exist, creating it..."
    New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
} else {
    Write-Output "Firewall rule already exists."
}
```

Profile : Private
The Problem: Public vs. Private Network Profiles
Your firewall rule is perfectly configured, but it is restricted to work only when Windows considers the current network to be a "Private" network (like a trusted home Wi-Fi).
Because UTM uses an "emulated VLAN" (Shared Network mode), Windows often doesn't recognize the virtual router providing the connection. By default, Windows plays it safe and classifies unidentified or new networks as Public.
If your UTM network is currently set to Public, the Windows Firewall is silently dropping your SSH packets because the rule only allows traffic on Private networks.
The Fix
You have PowerShell open already, so the fastest way to fix this is to modify that existing firewall rule to allow SSH traffic across Any network profile.
Run this exact command in your Administrator PowerShell window:
PowerShell
Set-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -Profile Any
Alternative Fix (Change the Network Profile)
If you'd rather keep the firewall rule strict, you can tell Windows to trust the UTM network instead:
Open Windows Settings (Win + I).
Go to Network & internet -> Ethernet.
Under Network profile type, change it from Public to Private.
Once you apply either of these fixes, try running ssh alessio@localhost -p 2222 from your Mac terminal again. It should immediately prompt you for your password or to accept the host key.

### Step 3: Connect from your Mac Terminal

configuring an SSH key first

1. On your Mac:
   Generate a new SSH key (if you don't already have one) by running:

   ```bash
   ssh-keygen -t ed25519
   ```

   (Press Enter to accept the default file location, and optionally add a passphrase).

2. Display your new public key and copy the output to your clipboard: (substitute the name of your pub key, and `pbcopy` is mac-only, use `xclip` or `wl-copy` on linux)

   ```bash
   cat ~/.ssh/id_ed25519.pub | pbcopy
   ```

3. On your Windows machine:

   ```powershell
   mkdir C:\Users\your_username\.ssh
   ```

4. Open Notepad, paste the public key you copied from your Mac, and save the file exactly as authorized_keys (with no .txt extension) inside that .ssh folder.
   Once that is saved, try logging in from your Mac again. It should log you right in without asking for your Windows password!

Now that both sides are configured, you can connect from your Mac.
Open your macOS Terminal and run the following command:

```bash
# after configuring the ssh-keygen and adding it to trusted keys in guest windows side
ssh windows_username@localhost -p 2222

# example
ssh alessio@localhost -p 2222
```

Replace windows_username with your actual Windows account username (if you use a Microsoft Account, it is usually the first 5 letters of your email address, or check the name of your user folder in C:\Users\).
Enter your Windows account password when prompted.

- Alternative: Bridged Network

  If you want the Windows VM to act like a completely separate machine on your physical local Network
  (with its own unique IP address from your router), you can
  change the Network Mode in UTM from Shared Network to Bridged (Advanced).
  If you do this, you won't need port forwarding;
  you will just SSH directly to the Windows IP address (ssh username@192.168.x.x).
  However, note that bridging does not always work reliably over Mac Wi-Fi adapters,
  which is why Port Forwarding is preferred.
