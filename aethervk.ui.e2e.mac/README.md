# aethervk.ui.e2e.mac — macOS end-to-end tests

Appium's Mac2 driver (XCTest) drives the packaged `.app`. Two kinds of tests live here:

- `MacHarnessTests` checks the harness itself: launch and quit, accessibility ids, the menu bar, both
  file panels, the CoreGraphics pixel probe, and keyboard input to the Metal viewport. **Run these
  first on a new machine.**
- `SharedScenarioTests` holds the same scenarios as the Windows suite, with the same test names.
  The scenarios live in `aethervk.ui.e2e.shared/` and are linked as source.

The project compiles on any OS with `-p:BuildE2E=true`. Off macOS, every test skips at runtime.
Without the flag, Build and Test do nothing, so the solution build skips it.

## One-time setup

1. **Xcode.** Install the full app, not only the command line tools. Open it once to accept the
   licence, then run `sudo xcode-select -s /Applications/Xcode.app`.
2. **Appium and the Mac2 driver.** These need Node 18 or newer.

       npm i -g appium
       appium driver install mac2

   The suite starts `appium` when nothing is listening on `127.0.0.1:4723`. It searches `PATH`,
   `/opt/homebrew/bin` and `/usr/local/bin`; set `AETHERVK_E2E_APPIUM=/path/to/appium` to choose
   one. The server's log goes to `TestResults/e2e-mac/appium.log`.
3. **Accessibility for the WebDriverAgent runner.** The first session builds and starts
   `WebDriverAgentRunner-Runner`. When macOS asks, allow it in
   System Settings › Privacy & Security › Accessibility. If no prompt appears, add the runner from
   `~/Library/Developer/Xcode/DerivedData/WebDriverAgentMac-*/Build/Products/Debug/`.
   Allow Xcode Helper as well if it asks.
4. **Screen Recording for the terminal.** Grant System Settings › Privacy & Security ›
   Screen Recording to the terminal (or IDE) that runs `dotnet test`. The pixel probe uses
   CoreGraphics, and without this permission it reads only the wallpaper or black.
   `PixelProbe_MatchesElementScreenshot` fails with a message that names this permission.
   Restart the terminal after granting it.
5. **The app bundle.** From the repository root:

       scripts/package_macos.sh osx-arm64 Release SideBySide

   This builds `bin/publish/AetherVk-SxS-1.0.1-Release.app` (bundle id
   `com.example.AetherVk.sxs.1-0-1.release`). Repackage after every change to the app or the
   native library; the suite does not rebuild anything.
6. **The fixtures.** The tests fetch them on demand from the test-assets release, just as the
   Windows suite does. These are `saves/review_67p_2021-12` and `fits_lt_67p_header_4x3.fits`.

Keep the app on the main display and don't touch the mouse or keyboard during a run. The tests
type, click and read screen pixels.

## Running

Run **one test class at a time**, harness first. Every test starts its own app and Appium
session. A class that fails at the harness level makes every test after it fail as well.

    dotnet test aethervk.ui.e2e.mac -p:BuildE2E=true --filter "FullyQualifiedName~MacHarnessTests"
    dotnet test aethervk.ui.e2e.mac -p:BuildE2E=true --filter "FullyQualifiedName~SharedScenarioTests"

On a new machine, run the harness tests one by one in this order:

    for t in Launch_Attach_Quit_3x TreeDump MenuBar_LayoutRadio_ReflectsState OpenDialog_RoundTrip \
             SaveDialog_RoundTrip PixelProbe_MatchesElementScreenshot Keyboard_ReachesNativeViewport; do
      dotnet test aethervk.ui.e2e.mac -p:BuildE2E=true --filter "FullyQualifiedName~MacHarnessTests.$t" || break
    done

`TreeDump` writes the accessibility tree to `TestResults/e2e-mac/tree/`, as page-source XML plus
screenshots of each classic tab and of the compact panel. It also logs which compact AutomationIds
are missing. If a locator fails later, look there first.

Artifacts (logs, PNGs, saves) go to `TestResults/e2e-mac/<scenario>/`.

### Environment variables

| Variable | Effect |
|---|---|
| `AETHERVK_E2E_APP` | The `.app` to test, or its `Contents/MacOS/AetherVk` binary. The default is the SideBySide Release bundle above. |
| `AETHERVK_E2E_BUNDLE_ID` | The bundle id for Mac2, if you test another packaging. |
| `AETHERVK_E2E_MAC_LAUNCH=driver` | Lets Mac2 launch the app with its `arguments`/`environment`, instead of the suite starting the binary and Mac2 attaching with `noReset`/`skipAppKill`. Use this when the attach relaunches the app. |
| `AETHERVK_E2E_APPIUM` | The path to the `appium` executable. |

The app gets `AETHERVK_E2E=1` and `AETHERVK_UI_CULTURE=en-US`. Its log is read from the process's
stdout. In driver-launch mode it comes from `~/Library/Logs/Aether-Vk/`. Crash reports are read
from `~/Library/Logs/DiagnosticReports/AetherVk*` and from `aethervk-crash-*.txt`.

## Compile check on Linux

    dotnet build aethervk.ui.e2e.mac -p:BuildE2E=true
    dotnet test aethervk.ui.e2e.shared.tests

The second command runs the shared math (frame metrics, point → pixel scaling, the PNG codec, the
id contract). It needs no display.

## Handoff patch rules

Fixes made on the Mac come back to the main tree as a patch. A Mac patch may touch only:

- `aethervk.ui.e2e.mac/`
- `scripts/package_macos.sh`
- `TODO_macos_packaging.md`
- any `*.Mac.cs`

Everything shared is off limits in a handoff patch. That covers:

- `aethervk.ui.e2e.shared/`, which holds the scenarios and the probe math;
- XAML and view models;
- resx;
- `aethervk.ui-logic/Automation/CompactAutomationIds.cs`;
- the Rust code.

If a scenario needs a change, report it so the change is made in the main tree for both platforms.
Platform locator differences go in `MacE2EApp`, never in a scenario. Scenarios never branch on the
OS.

Check a patch before you send it:

    scripts/check_handoff_patch.sh mac-fixes.patch macos

The script lists the files from the patch's `+++ b/` headers. If any path falls outside the
allowlist, it exits with status 1 and prints the offending paths. The Windows rule is the same,
with `aethervk.ui.e2e/` and `*.Windows.cs` (`... windows`).
