Step 1: Make Xcode dump its environment

We are going to tell Xcode to write its exact state to a file right before it tries to build your project.

Go back to your script box in Xcode (Build Rules tab -> expand the Cargo.toml rule).

Scroll down to right before the cargo build command (it's near the bottom, starting with { cargo build --manifest-path...).

Add this exact line right above it:

Bash
export -p > /tmp/xcode_env.sh
Hit Cmd + B to build. It will still fail, but it has now secretly dumped its entire environment state to your /tmp folder.

Step 2: Open a "Pristine" Terminal

Standard terminal tabs automatically load your ~/.zshrc and all your Homebrew paths. We need to bypass that entirely.

Open your standard Mac Terminal (or iTerm).

Run this command to launch a completely blank bash shell with zero environment variables (ignoring your profile and rc files):

Bash
env -i bash --norc --noprofile
(Your prompt will likely change to a simple bash-3.2$ indicating you are in a raw shell).

Step 3: Inject the Xcode Environment

Now, we load the variables Xcode dumped into this blank shell.

Source the file you generated:

Bash
source /tmp/xcode_env.sh
You are now officially running in a 1:1 clone of Xcode's build environment!