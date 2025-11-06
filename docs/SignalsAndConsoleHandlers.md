# Signals and Console Handlers

## Introduction

Some events, in particular we are now focusing on console events, ie commands sent by the
user through some special keyboard combinations (eg. `CTRL + C`, `CTRL + Z`) interact with
the foreground running process through a notification mechanism

## Posix Systems

### Prereq: Process Credentials (PID and User IDs)

Each process has

- *pid* and *pgid* (Process group ID)
  - by default, when a process starts, its pid === pgid, meaning it creates a new process group
  - the parent process can change the pgid of its children, such that they belong to the
    same process group, in which the parent is the **process group leader**
  - Process Groups are useful for **Job Control**, ie sending signals (see below) to a group
    of processses which are related to each other
  - on a shell, there may be multiple process groups children of a shell instance
    - **Foreground process group** PGID currently in execution
    - **Background process group** PGID running but not attached to the terminal
    - **Suspended process group** PGID suspended

- A process has 3 User IDs
  - refresher on `setuid`, `setgid`, `sticky bit` (t) on permissions: <https://www.cbtnuggets.com/blog/technology/system-admin/linux-file-permissions-understanding-setuid-setgid-and-the-sticky-bit>
    - modern security practice is to never set the `setuid` bit on binaries, but to
      set the capabilities you need
      - Link: <https://linux-audit.com/kernel/capabilities/linux-capabilities-hardening-linux-binaries-by-removing-setuid/#why-not-use-setuid>
  - *RUID* (Real User ID)
    - who started the process
    - Wikipedia link: <https://en.wikipedia.org/wiki/User_identifier#Saved_user_ID>
  - *EUID* (Effective User ID)
    - starts as equal to RUID. Can be changed by the process itself by calling `setuid`
      - `setuid(...)` can be called only if the process has `CAP_SETUID`
  - *SUID* (Saved-Set User ID)
    - when you start as priviledged, and, for safety, you want to switch to unpriviledged,
      there needs to be a way to go back to priviledged mode when you need it. SUID is a
      **Backup** of that priviledged UID, saved when you call
      `seteuid(non-priviledged-user-id)`... (example if you started with normal user `seteuid(getuid())`)
    - ...then, to restore the priviledged id, use `seteuid(geteuid())`
      - `geteuid()` gets the SUID, which is the EUID the process had before the `seteuid` call
    - More details on the man page about credentials: <https://man7.org/linux/man-pages/man7/credentials.7.html>

### Signals

References:

- The Linux Programming Interface

**Signals** are the notification channel sent by a process (eg. the shell) to another
process (our application), which inform the latter of one of 3 types

- Hardware exception (eg `SIGFPE`)
- user typed special character, eg an *interrupt character*
  (`CTRL + C`) or *suspend character* (`CTRL + Z`)

Signals are divided into **Standard** (1-31) and **Realtime**

Signals are delivered to a process as soon as it is *Scheduled to run again*, or immediately
if it is already running

- When we need a code segment to not be interrupted with signals, we can add them to a
  **Signal Mask**, which tells us which signals have their delivery **Blocked**, and they
  remain pending until they are unblocked later

There are 3 kinds of default actions for a signal

- Process is terminated abnormally
- Process is suspended
- Process is resumed (after being suspended)
- Signal is ignored by both the kernel and the process

We can ignore the default action by *Change the signal's* **Disposition**

- Default Action (depends on the signal, look it up)
- Ignore
- Execute *Signal Handler*

On Linux, you can lookup the current state of all signal dispositions and masks by
inspecting `/proc/PID/status`

To change the signal's disposition, use the `sigaction` function

### Sending signals

To send signals to another process, `kill(pid, signum)` (or `killpg` if you specify PGID,
which is equal to `kill(-pgid, signum)`), where if

- `signum = 0` you can check if a process exists.
- `pid = 0` you send the signal to all processes in the same **process group** (including
  calling process)
- `pid < -1` you send the signal to processes in the same process group with PGID is equal
  to `absolute(pid)`
- `pid = -1` broadcast to everybody except pid = 1 (`systemd`) and the calling process

And

- if no process is found, `ESRCH`
- The process needs permissions to send signals to another process
  - A process with capability `CAP_KILL` can send signals to any process
  - An unpriviledged process can send a signal to a receiving process only if
    - sender RUID == receiver RUID OR
    - sender EUID == receiver RUID OR
    - sender RUID == receiver SUID OR
    - sender EUID == receiver SUID
      - Note: The receiver's EUID is **NOT** used because we don't want to depend on the
        current settings of the receiver

### Uncatchable signals

- `SIGKILL`, `SIGSTOP`

No, you can't catch the uncatchable signals as it is always caught by the default
handler implemented by the kernel.

SIGKILL always terminates the process. Even if the process attempts to handle the
SIGKILL signal by registering a handler for it, still the control would always land
in the default SIGKILL handler which would terminate the program.

This is what happens when you try to shut down your system.

- First, the system sends a SIGTERM signal to all the processes and waits for a while
  giving those processes a *grace period*.
- If it still doesn't stop even after the grace period, the system forcibly terminates
  all the process by using SIGKILL signal.

### signal sets

signal-related System calls work on `sigset_t` (more signals at once) (eg `sigaction`, `sigprocmask`, `sigpending`)
hence we need to create a signal set

```cpp
int sigemptyset(sigset* outSet); // empty signal set. 0 = success
int sigfillset(sigset* outSet);  // full  signal set. 0 = success
int sigaddset(sigset_t* set, int sig); // add signal from set
int sigdelset(sigset_t* set, int sig); // remove signal from set
int sigismember(const sigset_t* set, int sig); // 1 if membership
```

### Signal Mask, Pending Signals, *No Queueing*

For each thread in a process, the kernel maintains a **Signal Mask**, set of signals
which are blocked for a given thread. we can set the signal mask for all threads in a process
with `sigprocmask`.

```cpp
// add signals in `set` in the signal mask of all threads. `old` is the set of signals in
// the mask before the set union operation (optional?)
int sigprocmask(SIG_BLOCK, const sigset_t* set, sigset_t* old);
// set difference of the old set from signal mask of all threads, returned in `old`, and
  // signals in `set`
int sigprocmask(SIG_UNBLOCK, const sigset_t* set, sigset_t* old);
// sets the signal mask for all threads as `set`
int sigprocmask(SIG_SETMASK, const sigset_t* set, sigset_t* old);
```

The per-thread function to manipulate a single signal mask is `pthread_sigmask`

If a thread receives a signal which is currently blocking, you can retrieve it by calling

```cpp
int sigpending(sigset_t* set);
```

Note:

- **Standard Signals are not queued**, hence you don't know how many times a
  pending signal was received
- *Uncatchable Signals* cannot be blocked (it will be ignored by the kernel)
- **Realtime signals are queued**
- You can suspend the **current thread** with `pause()` until a signal wakes it up

### Signals: Thread vs Process

Reference: *The Linux Programming Interface*, chapter 33

- **Actions are process-wide**: unhandled signals whose default action is terminate,
  the process is terminated
- **Signal Dispositions are process-wide** (`sigaction`)

A signal can be directed to the whole process or to a single thread.

A signal is Thread Directed if

- hardware exception raised on the thread which caused it (eg `SIGSEGV`)
- `SIGPIPE` generated when thread tried to write to a broken pipe
- Sent with `pthread_kill` or `pthread_sigqueue`
- Otherwise it is process directed. Some Examples
  - `kill()`, `sigqueue()` functions
  - resize terminal `SIGWINCH`
  - expiration of timer `SIGALRM`

Process-Wide Delivered signal select a **Random Thread** to execute the raised signal
(even a thread running a system call might be interrupted)

- alternate signal stack established with `sigaltstack` is per thread
- `pthread_mutex_lock` and `pthread_cont_wait` are **spuriously woken up**, hence
  you should always check again the state of a synchronization primitive after acquiring it

Since, in a multihtreaded program, there are many issues with dealing with signals
asynchronously, you should always:

- Block all signals in all threads
- have a dedicated thread to handle incoming signals with `sigwaitinfo`, `sigtimedwait`,
  or `sigwait` on a loop
  - or `signalfd` if on Linux 2.6.22 (it's not a standard, but linux only feature)

### Signal Handlers in practice

```cpp
struct sigaction {
  union {
    void     (*sa_handler)(int),  // handler (SIG_DFL or SIG_IGN for default or ignore)
    void     (*sa_sigaction)(int, siginfo_t*, void*) // see below
  } __sigaction_handler, // macros sa_handler and sa_sigaction for convenience
  sigset_t   sa_mask,           // signals to block during handler execution
  int        sa_flags,          // flags controlling handler invocation
  void     (*sa_restorer)(void) // always nullptr
};
```

- Don’t change the disposition of ignored terminal-generated signals
  - if, at process start, some of the following signals have disposition `SIG_IGN`, don't
    change it
    - SIGHUP, SIGINT, SIGQUIT, SIGTTIN, SIGTTOU, and SIGTSTP
- you should **NOT** return for hardware-generated signals. Either terminate the process,
 accept their default action (which is terminate), or perform a **non-local goto** (below)
- you don't need to add in the `sa_mask` the signal itself
- use `SA_ONSTACK` to establich an alternate stack using `sigaltstack`
- by default, additional signal delivery is blocked until its handler finishes executing
  - to remove this, you specify `SA_NODEFER`, hence you will execute the handler again.
    **Signals are not queued**, hence this cannot be used for counting (example on book 26.3.1)

Signal handlers are possibly called asynchronously with respect to program execution, hence
you cannot call **Non-Reentrant Functions** (meaning non thread-safe) like `malloc()`

- only a handful of posix functions can be callled safely in a signal handler
  (*The Linux Programming Interface, page 426*)

  - Even for reentrant functions there's a catch: **errno**
    - the signal handler should save `errno` and restore it at the end

For handling shared state inside a signal handler, the SUSv3 standard specified the datatype

```cpp
// can only be set and read, not even ++ or -- are defined
volatile sig_atomic_t flag;
```

Other than `return`, `kill()`, `raise()`, `_exit()`, `abort()`, you can conclude a signal
handler with a **Non Local goto** by using `sigsetjmp` or `siglongjmp`

- Example in *The Linux Programming Interface, page 430*

#### `SA_INFO` and system calls behaviour

If you set the `SA_INFO` flag, you need to use the `sa_sigaction` signature, which
gives us the `siginfo_t` struct carrying additional per-signal specific information

- *The Linux Programming Interface, page 438*

And the `ucontext_t`, which has stuff like saved register values, so it's never used.

- Can this be used outside of signals to implement user level threads? **NO** because they
  deprecated and removed the functions to manipulate the ucontext.

The `SA_RESTART` flag on `sa_flags` tells you whether a blocking system call should be
restarted after the signal handler or whether it should return `EINTR`

- Use the `signinterrupt(int sig, int flag)` to change the `SA_RESTART` of a given signal
  - `flag = 0`: blocking system calls will restart
  - `flag = 1`: blocking system calls will be interrupted

**Not all blocking system calls are restarted after the signal handler**, even if you use
the `SA_RESTART` flag.

- Therefore, system calls which can return `EINTR` should properly handle it
- note that `sleep()` system call, before and including Linux 2.4, when interrupted,
  returns the remaining number of unslept seconds, not an error

What if we **unblock multiple signals and both are pending** and the handler makes a
**system call**?

- The process starts executing the handler for the first signal **in integer ascending order**
  (not time of arrival)
- When return from system call, the handler for the second signal is invoked
- once the handler for the second signal returns, the first handler will resume execution
- See *The Linux Programming Interface, page 454*

### Realtime Signals

SUSv3 requires at minimum 8 realtime signals numbers, whose value range is defined with

```cpp
#if SIGRTMIN+100 > SIGRTMAX
#error "Not enough realtime signals"
#endif
```

The limit of how many realtime signals can be enqueued on the whole system
should be at minimum `_POSIX_SIGQUEUE_MAX`, which is 32. It can be queried

```cpp
// actual upper limit of number of realtime signals which a process
lim = sysconf(_SC_SIGQUEUE_MAX);
```

**Which on linux returns -1** because

- *Linux 2.6.7 and lower*: Query pseudofiles

  ```cpp
  // max realtime signals to be queried for all processes
  "/proc/sys/kernel/rtsig-max"
  // currently queued realtime signals on the system
  "/proc/sys/kernel/rtsig-nr"
  ```

- *Linux 2.6.8 and higher*: Query **Process Limits**
  (*The Linux Programming Interface, chapter 36*)

  ```cpp
  int getrlimit(RLIMIT_SIGPENDING, struct rlimit* rlim);
  ```

  Note that this is a per-process limit, hence the linux kernel makes sure that we can
  have some realtime signals

A realtime signal is handled as any other signal, but is sent with `sigqueue`

## Win32

## Mobile Devices
