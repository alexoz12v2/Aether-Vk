# Android Notes

## How many concurrent pointers can a device handle

```powershell
PS Y:\Aether-Vk> adb shell getevent -p
add device 1: /dev/input/event5
  name:     "uinput-silead"
  events:
    KEY (0001): 0066  008b  009e  00ac  00d4  0132  0161
  input props:
    <none>
add device 2: /dev/input/event0
  name:     "xiaomi-touch"
  events:
    KEY (0001): 0100  0153
  input props:
    <none>
add device 3: /dev/input/event3
  name:     "NVTCapacitiveTouchScreen"
  events:
    KEY (0001): 0074  014a  020b
    ABS (0003): 002f  : value 0, min 0, max 9, fuzz 0, flat 0, resolution 0
                0030  : value 0, min 0, max 255, fuzz 0, flat 0, resolution 0
                0035  : value 0, min 0, max 719, fuzz 0, flat 0, resolution 0
                0036  : value 0, min 0, max 1649, fuzz 0, flat 0, resolution 0
                0039  : value 0, min 0, max 65535, fuzz 0, flat 0, resolution 0
                003a  : value 0, min 0, max 1000, fuzz 0, flat 0, resolution 0
  input props:
    INPUT_PROP_DIRECT
add device 4: /dev/input/event4
  name:     "swtp"
  events:
    KEY (0001): 00f9  00fa
  input props:
    <none>
add device 5: /dev/input/event1
  name:     "ACCDET"
  events:
    KEY (0001): 0072  0073  00a4  00e2  0101  0102  0246
    SW  (0005): 0002  0004  0006  0007
  input props:
    <none>
add device 6: /dev/input/event2
  name:     "mtk-kpd"
  events:
    KEY (0001): 0072  0073  0074  0198
  input props:
    <none>
```

In particular

```sh
ABS (0003): 002f  : value 0, min 0, max 9, fuzz 0, flat 0, resolution 0
```

hence 10 touches

## Particular paths on android project

- Where does cpp libraries goto?

  ```src/launcher/android/app/build/intermediates/cxx/Debug/4324q3vl/obj/arm64-v8a/libaethervk.so```

- `android:debuggable` application attribute must be true if you want to be able to perform
  remote debugging over `adb` (in our case, Dual (java/cpp))
  - should not be inserted manually, but inserted through gradle's `buildTypes`, where each
    build type defines the member variable `isDebuggable`
  - you cannot easily change the application id with `applicationidSuffix = ".debug"` when
    JNI is involved, because it relies on some paths to automatically load the library
    (TODO: See how to edit the search path behaviour based on build type)

## Warning for ELF native libraries on android

They should have 16 KB support <https://developer.android.com/guide/practices/page-sizes#cmake>

Vulkan Validation Layers Don't have that!

- TODO: Need to integrate extension `VK_GOOGLE_display_timing`
- Use AStorageManager (Scoped Storage) or app->activity->internalDataPath to save files.

## Debugging

You have two possible Options: Google's AGI or ARM's performance studio shipped RenderDoc.

If you accidentally tried to run AGI even if it reported that
you android phone was not supported, the Vulkan Application might be waiting for the AGI server to connect indefinitely with logs like

```sh
GAPII awiting connection on pipe gapii (abstract)
```

a possible solution might be to disable all vulkan layers related
property with the `getprop` and `setprop` android shell commands,

and then make sure to uninstall with `pm` any google packages
related to it, and if necessary, restart the device

```sh
adb shell pm uninstall com.google.android.gapid.arm64v8a
adb reboot
```
