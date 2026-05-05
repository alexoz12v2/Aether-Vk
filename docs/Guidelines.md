# AetherVk Architectural & Code Style Guidelines

This document outlines the gathered engineering rules, coding conventions, and architectural constraints discovered and enforced during the development of AetherVk.

## 1. Native C/Rust FFI (cdylib)
- **Stateless & Context-Driven:** FFI functions MUST be stateless. They operate on an explicit `simulation_context_ptr` (and typically a `scene_id` parameter). The simulation engine does not keep a hidden global state.
- **Asynchrony for Heavy Operations:** File I/O, heavy parsing (e.g., GLTF/GLB models, SPK Kernels), and complex engine tasks MUST be processed asynchronously in the Rust logic thread. 
- **Task Polling:** Async FFI functions MUST immediately return a `u64 task_id`. The C# layer is responsible for polling `avkSimulationContext_getTaskStatus` before querying the result.
- **Explicit Result Types:** Functions returning data from tasks use explicit typed retrieval functions (e.g., `avkSimulationContext_getTaskResultU64`, `avkSimulationContext_getTaskResultKinematicState`) to maintain memory safety.
- **No C++ STL Boundaries:** All returned strings or arrays are passed through pointers managed by the caller, or returned as allocated buffers that MUST be explicitly freed by corresponding FFI calls (e.g., `avkSimulationContext_freeBvhNodes`).

## 2. C# UI & Logic Layer Separation (.NET)
- **Strict Project Boundaries:** 
  - `aethervk.ui.logic` (ViewModels, Services, Models) MUST target `netstandard2.0` and contain ZERO dependencies on Avalonia UI or any visual frameworks. 
  - `aethervk.ui.app` contains the `Avalonia` specific Views, UserControls, and bootstrapping.
- **MVVM Pattern & `CommunityToolkit.Mvvm`:** 
  - Leverage `[ObservableProperty]` and `[RelayCommand]` source generators.
  - No `async void` outside of event handlers (and we avoid event handlers in favor of Commands). Commands that await FFI operations MUST return `Task` and use `[RelayCommand]`.
- **Dependency Injection (DI) & Constructor Injections:**
  - **NO ServiceLocator:** The `ServiceLocator` anti-pattern is strictly banned.
  - ViewModels MUST request dependencies via their constructors.
  - Views MUST have parameterless constructors (to allow XAML Designer to instantiate them) and they must receive their data context via binding or assignment at runtime from the `IViewModelFactory` or DI resolution.
- **Decoupled Communication:** ViewModels and Services communicate system-wide events using `WeakReferenceMessenger` (e.g., `EntitySelectedMessage`, `BvhNodeVisibilityChangedMessage`). Models MUST NOT use Services directly.
- **UI Thread Dispatching:** Background logic polling or events originating from Native callbacks MUST dispatch property changes back to the UI thread using an abstracted `IUiThreadDispatcher`, ensuring the Logic layer remains agnostic to Avalonia's `Dispatcher.UIThread`.

## 3. C# Tests
- **No UI Spawning in Tests:** Tests run in memory and MUST mock UI dispatchers and file dialogues using `Moq`.
- **Graceful Shutdown & Leak Prevention:** Native resources allocated in tests MUST be gracefully cleaned up (e.g., `_service.Dispose()`).
- **Disk Artifacts:** C# tests rendering images or exporting scenes must output `.png` and `.json` (scene hierarchy) to a local `render/` directory to avoid dirtying the workspace.

## 4. Subagent Recommendations: Logic Thread Asynchrony
From the recent `@codebase_investigator` architectural review, the following methods in the Logic/Render threads are categorized:

**MUST Be Asynchronous (Thread-Pool or Logic Thread Worker):**
- `loadCometSpk`: Reads memory-mapped BSP files. Spawns tasks and returns `KinematicState`. *(Implemented)*
- `importModel`: Parses GLB/GLTF assets. Memory-intensive. *(Implemented)*
- `loadAlmanacFile`: Heavy text or kernel parsing. *(Implemented)*
- `spawnModelInstance`: Deep copies FFI components. *(Implemented)*
- `raycast`: Needs to traverse BVH trees dynamically. *(Implemented)*
- *Future FFI Methods*: Any operation mutating deep scene graphs, loading textures, or requiring complex computations should return a task ID.

**CAN Remain Synchronous (Immediate return):**
- Simple visibility toggles (`setEntityVisibility`, `setBvhNodeVisibility`).
- Transform getters/setters (`getTransformComponent`, `setTransformComponent`).
- Component property edits (e.g., Camera parameters, Sun properties).
- UI polling and data read operations (`getEntityName`, `getSimulationTime`).