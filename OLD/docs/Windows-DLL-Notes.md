# Windows DLL

## Implicit Loading vs Explicit Loading

1. Implicit (Load Time) Linking: Using an _Import Library_ `.lib` and `__declspec(dllimport)`
2. Explicit (Runtime) Linking: Using `LoadLibrary` and `GetProcAddress`

They differ on how the PE Executable is built, when the DLL is loaded, how symbols are resolved, and how the _DLL
Reference Count_
is managed

### Implicit Linking

- You link against the _Import Library_ `foo.lib`, not the DLL Itself
- The `.lib` does **not** contain code. Only symbol stubs and metadata
- The compiler emits references marked as `__declspec(dllimport)`
- The linker
    - Creates and _Import Directory_ entry in the PE Executable
    - Adds entries to the **Import Address Table** (IAT)

At Runtime:

- The **Windows Loader**
    - Maps the PE Executable file (`.exe`)
    - **Loads** all the required DLLs listed into the import table
      ```shell
      dumpbin /IMPORTS $exeFile
      ```
    - Resolves all imported symbols
    - Patches the IAT with the actual function addresses

After that, calls are indirect calls via the IAT

Each symbol in the IAT contains

- symbol stub
- DLL Name
- Function names and/or **Ordinals**

Meaning that the executable file gains an `IMAGE_IMPORT_DESCRIPTOR` entry for the linked DLL file

- The DLL Must be resolved at process startup (Unless _Delay Loaded_), otherwise error

Effects on the process address space

- DLL Is mapped before `main()`/`WinMain()` executes
- DLL Code Section
    - shared across processes
    - mapped read-only
- DLL Data Section
    - Private per process
- A function call, eg `foo()` is mapped (example with x64) to
  ```txt
  ;; assuming MASM syntax
  call qword ptr [IAT_foo]
  ```

The DLL usage count

- Loader increments the DLL reference count **Once per importing module** (No matter how many threads in the process use
  the library)
- The DLL Stays loaded in the process until
    - the process exits, OR
    - `FreeLibrary` is called **only if you manually loaded it as well** (No matter whether explicit or implicit
      linking)
        - You cannot explicitly unload an implicitly loaded library unless you `LoadLibrary` yourself, which creates an
          extra reference

Pros of Implicit Linking

- Simple syntax
- type-safe at compile type
- fast calls (single indirection)
- Automatic lifetime management

Cons

- DLL must exist at load time
- Hard Dependency (Unless _Delay Loaded_)

### Explicit Linking

The Executable

- Has **No import table entry** for `foo.dll`
- Does not list the DLL as a dependency (`dumpbin` and software like dependency walker can't see it)
- Probably only imports `kernel32.dll` and other windows functionality (at least `LoadLibrary`, `GetProcAddress`)

Result

- Executable can start without resolving and loading `foo.dll`

This means that **The DLL is mapped into process address space only when** `LoadLibrary` is called

1. `LoadLibrary`
    - Locate the DLL
    - Map it to process
    - Increment its usage count for the process
    - call, when first loaded `DllMain(DLL_PROCESS_ATTACH)` into the DLL
2. `GetProcAddress`
    - Looks up the export table
    - Returns a function pointer
3. `FreeLibrary`
    - Decrements usage count
    - Unloads DLL when count reaches zero
    - call, when finally unloaded `DllMain(DLL_PROCESS_DETACH)`

Mapping is identical to implicit loading once the `LoadLibrary` is called, but function calls are compiled to calls
through a function pointer. Example with x64

```txt
call rax ; or similiar indirect call via function pointer
```

Pros

- DLL can bhe optional
- Ideal for plugin
- you control when it loads/unloads
- can choose between multiple versions

Cons

- Boilerplate
- No compile time checking
- Must manually define function signatures
- Easier to crash if mismatched

| Aspect            | Implicit linking         | Explicit linking             |
|-------------------|--------------------------|------------------------------|
| DLL loaded        | At process startup       | When `LoadLibrary` is called |
| PE import table   | Contains DLL + symbols   | No entry for the DLL         |
| Symbol resolution | By loader                | By your code                 |
| Call mechanism    | IAT indirection          | Function pointer             |
| Type safety       | Compile-time             | Manual                       |
| Optional DLL      | No (unless delay-load)   | Yes                          |
| Lifetime control  | Loader-managed           | Explicit                     |
| Reference count   | One per importing module | One per `LoadLibrary`        |

### Delay-Loaded DLLs (Hybrid Model)

Setup identical to Implicit Loading, plus passing to the linker `/DELAYLOAD:foo.dll`

- DLL **Appears** in the **delay-import table**
- Loaded on **First Function call**
- Still uses IAT
- Still type-safe
- Can intercept load failures

## Integrating DLL build with cmake ninja clang into our C# .NET WinRT/WinUI3 application

- don't use ASAN. You'll crash when calling p/invoke
- The strategy here is to create a nuget package with a cmake install script, install it locally and let it be referenced from the application
  - note: Whenever you change something inside the generated nuget package, **delete it from the package cache**
    which is under `$env:userprofile\.nuget\packages`
