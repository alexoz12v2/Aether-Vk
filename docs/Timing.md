# Timing

## Why is it necessary

## Approach

Our architecture includes
- *main thread* which handles interaction with native
  windowing system and enqueues any input/lifecycle events
  and coordinates the other threads
- *render thread* which starts a rander loop iteration every time it is signaled by
  the update thread (and it should refresh only the necessary content?)
- *update thread* responsible to run the update loop, which performs two kinds of
  updates for any dynamic entity in the application (reference From Unity <https://learn.unity.com/tutorial/update-and-fixedupdate>)
  - *Update*: runs once per frame, independently of any physical simulation, passing its
    elapsed time from the previous frame. This is **not** called on a regular timeline, as
    if one frame takes more time, elapsed time will be different
    - called every frame
    - should check for input, as input is enqueued on a per frame (iteration) basis
    - any non-physics related logic
      - Note: Non-Physics based logic still does take into account delta time, to
        avoid frame-dependant behaviour (Cit: "Game Engine Architecture, 3rd ed")
  - *FixedUpdate*: can run zero, one, or more times per frame iteration, as it is
    called on a regular timeline, meaning elapsed time should roughly be the same.
    After a fixed update is called, the physical simulation (collision detection and
    response are done)
    - Called every physics step
    - elapsed time between calls should be consistent
    - adjust physics simulation parameters

Other reference: <https://docs.unity3d.com/6000.2/Documentation/ScriptReference/MonoBehaviour.FixedUpdate.html>

Time handling is explained on this Unity page: <https://docs.unity3d.com/6000.2/Documentation/Manual/time-handling-variations.html>

- `Time.time` capped game time from the start of the application
- `Time.deltaTime` capped game time from the last frame
- `Time.timeScale` time scaling applied from real time
- `Time.unscaledTime`, `Time.unscaledDeltaTime` real time counterparts of `time`,`deltaTime`
  - Needed when simulation is paused `Time.timeScale = 0`, but some parts should still be updated normally, (eg. UI)
- `Time.maximumDeltaTime` maximum variation `time` and `deltaTime` should have per frame. Unscaled, real time versions,
  ignore this cap
- `Time.smoothDeltaTime` reports an [Exponentially Smoothed](https://en.wikipedia.org/wiki/Exponential_smoothing) version
  of the `deltaTime` (with a given, fixed, time window)
  - Used when we need to avoid numerical fluctuations on non-physics logic

As we explain later, Physics Simulations need their code to be executed on a regular timeline, hence the need to
maintain the `Time.fixedTime` and `Time.fixedDeltaTime` variables

- `Time.fixedDeltaTime` interval in (uncapped) game time in which fixed updates should occur
  - Example: fixedDeltaTime of 1 second in a game with a Time.timeScale of 0.5 means fixed updates occur every 2 seconds of real time.
- `Time.fixedTime` value at which the current fixed update started (game time). It is updated before every fixed update
  by adding `Time.fixedDeltaTime`

Every time we compute the `deltaTime`, we check whether the `fixedTime` is behind `time` by at least one `fixedDeltaTime`,
and if so, execute a fixed update

- Note: In one update loop iteration, we can process at most `floor(maximumDeltaTime / fixedDeltaTime)` fixed updates,
  so that in cases the processing speed drops, we'd rather slow down the physical simulation rather than fall back
  on frame processing and rendering

## Numeric Data Types for time

As "Game Engine Architecture, 3rd ed" and [This website](https://geoffviola.github.io/2019/02/18/chrono-basics.html)
suggest, floating point formats are too unstable as their magnitude increases, hence we prefer integer formats

- floating point might still be needed for convenience, hence a utility method which reports a delta in float32 is fine

As the website suggests, using a `int64_t` allows us to track time in nanoseconds up to 292 Years, so that's what we'll
go with that, but with **Microseconds**

## `<chrono>` clock types

We prefer interacting directly with the ABI, but there are a few considerations worth mentioning in the C++ standard
`<chrono>` library. In particular [Steady vs System Clock](https://stackoverflow.com/questions/31552193/difference-between-steady-clock-vs-system-clock)

- The System clock periodically corrects itself to synchronize with an atomic clock reference machine through the Network Time Protocol (NTP)
- [`high_resolution_clock` Shouldn't be used](https://en.cppreference.com/w/cpp/chrono/high_resolution_clock.html), as we don't know whether it's steady or not

## Small footnote for debugging

When you insert a breakpoint inside a loop iteration on the update thread, time
measurements will be falsified

"Game Engine Architecture, 3rd ed" (paragraph 8.5) suggests checking for a max value
on debug builds and assume the elapsed time is equal to our target rate

```cpp
#ifdef AVK_DEBUG
if (dt > LIMIT) {
  dt = TARGET;
}
#endif
```

## How Steady Time is Queried on the different Systems

- Complete Example on how to query number of cycles since power-on [Here](https://github.com/google/benchmark/blob/v1.1.0/src/cycleclock.h#L116)
  - Shouldn't be used directly. From [Windows Docs](https://learn.microsoft.com/en-us/windows/win32/sysinfo/acquiring-high-resolution-time-stamps)
    - "We strongly discourage using the RDTSC or RDTSCP processor instruction to directly query the TSC because you won't get reliable results on some versions of Windows"
  - Is there a cost in querying time through a system call? Yes, and It's cost we are willing to afford

Hence, take a look at how operating systems allow you to query steady time in terms of CPU cycles and frequency

- Will these kind of CPU timestamps ever wrap around? **No**, as they are typically 64-bit wide, and it would take
  Thousands of years, even at high CPU clock frequencies, to fully overflow the timestamp counter from system boot.

### Windows

The never-failing Functions `QueryPerformanceFrequency` and `QueryPerformanceCounter`

- The frequency of the processor is set at system boot and never changes, hence it should be queried only at startup
- Does it work on [Windows on ARM? Yes](https://learn.microsoft.com/en-us/windows/arm/overview)
- Should read the `rdtsc` on a x86/x86_64 platform

Example

```cpp
LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
LARGE_INTEGER Frequency;
// At start
QueryPerformanceFrequency(&Frequency);
QueryPerformanceCounter(&StartingTime);
// every time you query for time. Note how you multiply before dividing by frequency
QueryPerformanceCounter(&EndingTime);
ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
ElapsedMicroseconds.QuadPart *= 1'000'000; // 1 s = 1e6 ns
ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
```

### Android/Linux (POSIX)

While on android it is possible to perform [hardware register queries](https://www.gamedeveloper.com/programming/getting-high-precision-timing-on-android),
a simpler way is to query the Linux kernel for time.

- Note: If you have a rooted Android device, you can query for hardware timers with
  
  ```shell
  adb shell cat /sys/devices/system/clocksource/clocksource0/current_clocksource 
  adb shell cat /sys/devices/system/clocksource/clocksource0/available_clocksource
  ```

In particular, we want to look at the [`clock_gettime`](https://man7.org/linux/man-pages/man3/clock_gettime.3.html)
function.

- On systems implementing time queries with Linux's [former vsyscalls, now vDSO](https://0xax.gitbooks.io/linux-insides/content/SysCall/linux-syscall-3.html)

  ```shell
  adb shell "cat /proc/self/maps | grep vdso"
  ```
  
  to query support for vDSO on your device. if something gets printed, it's active
  
  To check whether your Android's bionic libc++ implementation has vDSO, use
  
  ```shell
  # on newer systems this is a symlink to the new location, see below
  adb shell "readelf -s /system/lib64/libc.so | grep -E vdso"
  # example output:
  # 1995: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS vdso.cpp
  # 2004: 000000000008f800  1248 FUNC    LOCAL  HIDDEN    14 _Z16__libc_init_vdsoP12libc_globals
  adb shell "zcat /proc/config.gz | grep VDSO"
  # example output:
  # CONFIG_HAVE_GENERIC_VDSO=y
  # to check where, in the cat process, vdso shared library is loaded (virutal address, hence different result every time)
  adb shell "cat /proc/self/maps | grep '\[vdso\]' | awk '{print $1}' | cut -d'-' -f1"
  # you cannot copy process memory to disk unless you have a rooted device, hence
  
  # new location of bionic libc++
  adb shell "readelf -Ws /apex/com.android.runtime/lib64/bionic/libc.so | grep vdso"
  # example output:
  # 1995: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS vdso.cpp
  # 2004: 000000000008f800  1248 FUNC    LOCAL  HIDDEN    14 _Z16__libc_init_vdsoP12libc_globals
  ```
  
  To definitely establish whether a given POSIX function uses a true system call or is vDSO optimized,
  we can use the [`strace`](https://man7.org/linux/man-pages/man1/strace.1.html), a command-line utility for Linux and
  other Unix-like systems used to debug and diagnose processes by tracing system calls and signals

  ```shell
  adb shell strace -e write sh -c "echo hello > /data/local/tmp/test.txt"
  # output: there's a write system call!
  # write(1, "\n", 1)                       = 1
  # +++ exited with 0 +++
  adb shell strace -e clock_gettime date
  # output: you see the output of the program without any syscalls, hence clock_gettime(CLOCK_REALTIME, &ts) has been
  # vDSO optimized. To test the CLOCK_MONOTONIC you'd have to compile a small binary yourself and `adb push` it inside
  # your Android Phone:
  # Sun Nov 16 11:38:34 CET 2025
  # +++ exited with 0 +++
  ```
  
  see paragraph about more [testing on android](#testing-on-android)

(Cit. "The Linux Programming Interface", ch. 23.5) The main 2 functions from `time.h`
are

```cpp
int clock_gettime(clockid_t clockid, struct timespec* tp);
int clock_getres(clockid_t clockid, struct timespec* res);
```

the `clock_getres` specifies the granularity of the time value returned by `clock_gettime`. We require at least
**Microseconds** granularity, which should be supported by every device, hence we just log a warning otherwise

There are 2 (4 on Linux >= 2.6.12) clock types, being `CLOCK_REALTIME`, which is the system clock, hence affected by
the user modifying it and synchronization through the NTP; `CLOCK_MONOTONIC` is the monotonic clock provided through
hardware instructions, hence unaffected by other events. The latter is what we are interested in

### MacOS/iOS/iPadOS

There are the functions from **Mach Time**, even though some are considered [unsafe](https://mjtsai.com/blog/2024/05/06/swifts-native-clocks-are-very-inefficient/)
As apparently there's an exploit which allows retrieving the fingerprint of the user from [`mach_absolute_time`](https://developer.apple.com/documentation/kernel/1462446-mach_absolute_time)

Hence, it's preferred to use the BSD api which is [`clock_gettime_nsec_np(CLOCK_UPTIME_RAW)`](https://www.manpagez.com/man/3/clock_gettime_nsec_np/) from `time.h`, whose value
needs to be converted into microseconds to be ready for usage

## Testing on android

On my Android API Level 34, ARMv8 64 bit ABI, there are no real system calls, hence getting both realtime and monotonic
time goes through vDSO

```powershell
PS C:\Users\alessio> $clang="C:\Users\alessio\AppData\local\Android\sdk\ndk\28.2.13676358\toolchains\llvm\prebuilt\windows-x86_64\bin\aarch64-linux-android34-clang++.cmd"
PS C:\Users\alessio> cd Y:\Aether-Vk\docs
PS Y:\Aether-Vk\docs> &$clang clock_test.c -o clock_test
PS Y:\Aether-Vk\docs> adb push clock_test /data/local/tmp
clock_test: 1 file pushed, 0 skipped. 1.5 MB/s (6392 bytes in 0.004s)
PS Y:\Aether-Vk\docs> adb shell chmod +x /data/local/tmp/clock_test
# executable needs to find in its LD_LIBRARY_PATH/RPATH the libc++ shared. just copy it
PS Y:\Aether-Vk\docs> $libc_shared="C:\Users\alessio\AppData\Local\Android\Sdk\ndk\28.2.13676358\toolchains\llvm\prebuilt\windows-x86_64\sysroot\usr\lib\aarch64-linux-android\libc++_shared.so"
PS Y:\Aether-Vk\docs> adb push $libc_shared /data/local/tmp
C:\Users\alessio\AppData\Local\Android\Sdk\ndk\28.2.136763...le pushed, 0 skipped. 221.7 MB/s (9236352 bytes in 0.040s)
PS Y:\Aether-Vk\docs> adb shell ls "/data/local/tmp | grep libc"
libc++_shared.so
PS Y:\Aether-Vk\docs> adb shell ls "/data/local/tmp | grep clock_test"
clock_test
PS Y:\Aether-Vk\docs> adb shell "export LD_LIBRARY_PATH=/data/local/tmp:`$LD_LIBRARY_PATH ;  strace -e clock_gettime /data/local/tmp/clock_test"
MONOTOMIC: 80087.670150465
REALTIME: 1763290601.683383321
+++ exited with 0 +++
```
