using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.Tests.Mocks;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// Shared fake of the native asset library: <see cref="INativeRuntimeService"/> mock whose
/// asset list / thumbnails are driven by the test, plus a way to fire the
/// <c>AssetImported</c> external state like the native callback thread would.
/// </summary>
internal sealed class FakeAssetRuntime
{
  public readonly Mock<INativeRuntimeService> Runtime = new();
  public readonly TestSchedulerProvider Schedulers = new();
  public readonly Mock<ILocalStorageService> Storage = new();
  public readonly List<ImportedAsset> NativeAssets = [];
  public readonly List<(ulong RequestId, string Path, string CacheDir)> ImportRequests = [];
  public readonly Dictionary<ulong, int> ThumbnailCalls = [];
  public bool LibraryBusy;
  public bool AcceptImports = true;
  public readonly List<(ulong RequestId, ulong AssetId)> RemoveRequests = [];
  private Action<nint>? _assetImported;
  private Action<nint>? _assetRemoved;

  public FakeAssetRuntime()
  {
    Storage.Setup(s => s.GetSessionPath("asset_cache")).Returns("/session/asset_cache");
    Runtime
      .Setup(r => r.RegisterExternalStateListener(ExternalStateType.AssetImported, It.IsAny<Action<nint>>()))
      .Callback<ExternalStateType, Action<nint>>((_, h) => _assetImported = h)
      .Returns(Mock.Of<IDisposable>());
    Runtime
      .Setup(r => r.RegisterExternalStateListener(ExternalStateType.AssetRemoved, It.IsAny<Action<nint>>()))
      .Callback<ExternalStateType, Action<nint>>((_, h) => _assetRemoved = h)
      .Returns(Mock.Of<IDisposable>());
    Runtime
      .Setup(r => r.RemoveAsset(It.IsAny<ulong>(), It.IsAny<ulong>()))
      .Returns<ulong, ulong>((rid, id) =>
      {
        RemoveRequests.Add((rid, id));
        return true;
      });
    Runtime
      .Setup(r => r.ImportAsset(It.IsAny<ulong>(), It.IsAny<string>(), It.IsAny<string>()))
      .Returns<ulong, string, string>((id, path, dir) =>
      {
        ImportRequests.Add((id, path, dir));
        return AcceptImports;
      });
    Runtime.Setup(r => r.GetAssets()).Returns(() => LibraryBusy ? null : NativeAssets.ToArray());
    Runtime
      .Setup(r => r.GetAssetThumbnail(It.IsAny<ulong>()))
      .Returns<ulong>(id =>
      {
        ThumbnailCalls[id] = ThumbnailCalls.GetValueOrDefault(id) + 1;
        return new AssetThumbnail(1, 1, [(byte)id, 0, 0, 255]);
      });
  }

  public AssetLibraryService CreateService() =>
    new(Runtime.Object, Schedulers, Storage.Object);

  public static ImportedAsset Mesh(ulong id, string label, params ulong[] bundled) =>
    new(id, AssetKind.Mesh, label, $"/data/{label}.glb", 100, 300,
      [.. bundled.Concat(Enumerable.Repeat(0UL, 4)).Take(4)], null, null);

  public static ImportedAsset Texture(ulong id, string label, TextureChannel? hint = null) =>
    new(id, AssetKind.Texture, label, $"/data/{label}.png", 64, 64, [], hint, null);

  /// <summary>Fires <c>AssetImported</c> and drains the schedulers.</summary>
  public void CompleteImport(ulong requestId, bool success = true, ulong meshId = 0, uint added = 1)
  {
    Assert.NotNull(_assetImported);
    var dto = new CAssetImportedDTO
    {
      RequestId = requestId,
      MeshId = meshId,
      Success = success ? 1u : 0u,
      AddedCount = added,
    };
    var ptr = Marshal.AllocHGlobal(Marshal.SizeOf<CAssetImportedDTO>());
    try
    {
      Marshal.StructureToPtr(dto, ptr, false);
      _assetImported!(ptr);
    }
    finally
    {
      Marshal.FreeHGlobal(ptr);
    }
    Drain();
  }

  /// <summary>Fires <c>AssetRemoved</c> (removing the asset from the fake list on success).</summary>
  public void CompleteRemoval(ulong requestId, ulong assetId, bool success = true, uint ejected = 0)
  {
    Assert.NotNull(_assetRemoved);
    if (success) NativeAssets.RemoveAll(a => a.Id == assetId);
    var dto = new CAssetRemovedDTO
    {
      RequestId = requestId,
      AssetId = assetId,
      Success = success ? 1u : 0u,
      Ejected = ejected,
    };
    var ptr = Marshal.AllocHGlobal(Marshal.SizeOf<CAssetRemovedDTO>());
    try
    {
      Marshal.StructureToPtr(dto, ptr, false);
      _assetRemoved!(ptr);
    }
    finally
    {
      Marshal.FreeHGlobal(ptr);
    }
    Drain();
  }

  public void Drain()
  {
    Schedulers.Background.Start();
    Schedulers.MainThread.Start();
  }
}

public class AssetLibraryServiceTests
{
  [Fact]
  public void CAssetImportedDto_matches_native_layout()
  {
    Assert.Equal(24, Marshal.SizeOf<CAssetImportedDTO>());
    Assert.Equal(24, Marshal.SizeOf<CAssetRemovedDTO>());
    Assert.Equal(72, Marshal.SizeOf<CometAppearanceDto>());
  }

  [Fact]
  public async Task Import_sends_session_cache_dir_and_completes_on_matching_callback()
  {
    var f = new FakeAssetRuntime();
    using var svc = f.CreateService();
    ImmutableArray<ImportedAsset> meshes = [], textures = [];
    svc.Meshes.Subscribe(m => meshes = m);
    svc.Textures.Subscribe(t => textures = t);

    var task = svc.ImportAsync("/data/comet.glb");
    Assert.False(task.IsCompleted);
    var (requestId, path, cacheDir) = Assert.Single(f.ImportRequests);
    Assert.Equal("/data/comet.glb", path);
    Assert.Equal("/session/asset_cache", cacheDir);
    Assert.True(svc.IsImportingValue);

    // native: one mesh + its bundled albedo, split into two assets
    f.NativeAssets.Add(FakeAssetRuntime.Mesh(1, "comet", 2));
    f.NativeAssets.Add(FakeAssetRuntime.Texture(2, "comet_albedo", TextureChannel.Albedo));

    f.CompleteImport(requestId + 1000); // someone else's request: ignored
    Assert.False(task.IsCompleted);

    f.CompleteImport(requestId, meshId: 1, added: 2);
    var result = await task;
    Assert.True(result.Success);
    Assert.Equal(1UL, result.MeshId);
    Assert.Equal(2u, result.AddedCount);
    Assert.False(svc.IsImportingValue);

    Assert.Equal("comet", Assert.Single(meshes).Label);
    var tex = Assert.Single(textures);
    Assert.Equal("comet_albedo", tex.Label);
    Assert.NotNull(tex.Thumbnail);
    Assert.Equal(2UL, meshes[0].BundledTexture(TextureChannel.Albedo));
  }

  [Fact]
  public async Task Unsupported_extension_is_rejected_without_native_call()
  {
    var f = new FakeAssetRuntime();
    using var svc = f.CreateService();
    var result = await svc.ImportAsync("/data/notes.txt");
    Assert.False(result.Success);
    Assert.Empty(f.ImportRequests);
  }

  [Fact]
  public async Task Queue_rejection_fails_immediately_and_clears_importing()
  {
    var f = new FakeAssetRuntime { AcceptImports = false };
    using var svc = f.CreateService();
    var result = await svc.ImportAsync("/data/rock.png");
    Assert.False(result.Success);
    Assert.False(svc.IsImportingValue);
  }

  [Fact]
  public async Task Failed_import_completes_without_touching_the_list()
  {
    var f = new FakeAssetRuntime();
    using var svc = f.CreateService();
    var task = svc.ImportAsync("/data/broken.glb");
    f.NativeAssets.Add(FakeAssetRuntime.Mesh(9, "should_not_appear"));
    f.CompleteImport(f.ImportRequests[0].RequestId, success: false, added: 0);
    Assert.False((await task).Success);
    Assert.Empty(svc.CurrentMeshes);
    f.Runtime.Verify(r => r.GetAssets(), Times.Never);
  }

  [Fact]
  public void Busy_library_is_retried_on_next_completion_and_thumbnails_are_cached()
  {
    var f = new FakeAssetRuntime();
    using var svc = f.CreateService();
    _ = svc.ImportAsync("/data/a.png");
    _ = svc.ImportAsync("/data/b.png");
    f.NativeAssets.Add(FakeAssetRuntime.Texture(1, "a"));

    f.LibraryBusy = true; // second import still holds the native library
    f.CompleteImport(f.ImportRequests[0].RequestId);
    Assert.Empty(svc.CurrentTextures);

    f.LibraryBusy = false;
    f.NativeAssets.Add(FakeAssetRuntime.Texture(2, "b"));
    f.CompleteImport(f.ImportRequests[1].RequestId);
    Assert.Equal(["a", "b"], svc.CurrentTextures.Select(t => t.Label));

    Assert.True(svc.Refresh());
    Assert.Equal(1, f.ThumbnailCalls[1]);
    Assert.Equal(1, f.ThumbnailCalls[2]);
  }

  [Fact]
  public void Reimport_reports_nothing_added()
  {
    var f = new FakeAssetRuntime();
    using var svc = f.CreateService();
    var task = svc.ImportAsync("/data/a.png");
    f.NativeAssets.Add(FakeAssetRuntime.Texture(1, "a"));
    f.CompleteImport(f.ImportRequests[0].RequestId, added: 0);
    Assert.Equal(0u, task.Result.AddedCount);
    Assert.Single(svc.CurrentTextures);
  }

  [Fact]
  public async Task Remove_completes_on_callback_refreshes_list_and_drops_thumbnail()
  {
    var f = new FakeAssetRuntime();
    using var svc = f.CreateService();
    f.NativeAssets.Add(FakeAssetRuntime.Mesh(1, "comet", 2));
    f.NativeAssets.Add(FakeAssetRuntime.Texture(2, "comet_albedo"));
    svc.Refresh();
    Assert.Single(svc.CurrentMeshes);

    var task = svc.RemoveAsync(1);
    Assert.True(svc.IsBusyValue);
    var (requestId, assetId) = Assert.Single(f.RemoveRequests);
    Assert.Equal(1UL, assetId);
    f.CompleteRemoval(requestId, 1, ejected: 1);

    var result = await task;
    Assert.True(result.Success);
    Assert.Equal(1u, result.Ejected);
    Assert.False(svc.IsBusyValue);
    Assert.Empty(svc.CurrentMeshes);
    Assert.Single(svc.CurrentTextures); // bundled textures are kept

    // a re-import of the same id would fetch a fresh preview
    f.NativeAssets.Add(FakeAssetRuntime.Mesh(1, "comet", 2));
    svc.Refresh();
    Assert.Equal(2, f.ThumbnailCalls[1]);
  }

  [Fact]
  public async Task Refused_removal_keeps_the_asset()
  {
    var f = new FakeAssetRuntime();
    using var svc = f.CreateService();
    f.NativeAssets.Add(FakeAssetRuntime.Texture(5, "rock"));
    svc.Refresh();
    var task = svc.RemoveAsync(5);
    f.CompleteRemoval(f.RemoveRequests[0].RequestId, 5, success: false);
    Assert.False((await task).Success);
    Assert.Single(svc.CurrentTextures);
  }

  [Fact]
  public void Dispose_cancels_pending_imports()
  {
    var f = new FakeAssetRuntime();
    var svc = f.CreateService();
    var task = svc.ImportAsync("/data/a.png");
    svc.Dispose();
    Assert.True(task.IsCanceled);
  }
}

public class ImportsTabViewModelTests
{
  private static (ImportsTabViewModel Vm, FakeAssetRuntime Fake, AssetLibraryService Svc, Mock<IFileDialogService> Dialog, TimelineService Timeline)
    Create()
  {
    var f = new FakeAssetRuntime();
    var svc = f.CreateService();
    var translation = new Mock<ITranslationService>();
    translation.Setup(t => t.CultureChanged).Returns(System.Reactive.Linq.Observable.Empty<System.Globalization.CultureInfo>());
    translation.Setup(t => t.GetString(It.IsAny<string>())).Returns<string>(k => k);
    var sessions = new Mock<ITabStateService<ImportsSession>>();
    sessions.Setup(s => s.ActiveSessionIds).Returns(new System.Collections.ObjectModel.ObservableCollection<SessionId> { new SessionId(typeof(ImportsSession), 1) });
    sessions.Setup(s => s.ObserveSession(It.IsAny<SessionId>())).Returns(System.Reactive.Linq.Observable.Return(new ImportsSession()));
    sessions.Setup(s => s.ObserveSessionList()).Returns(new System.Reactive.Subjects.BehaviorSubject<IReadOnlyList<SessionId>>(new List<SessionId>()));
    f.Storage.Setup(s => s.SessionDirectory).Returns("/nonexistent/sessions/current");
    var dialog = new Mock<IFileDialogService>();
    var timeline = new TimelineService(f.Runtime.Object, f.Schedulers, new CometConfigService(f.Runtime.Object, f.Schedulers), new BreadcrumbService());
    var vm = new ImportsTabViewModel(translation.Object, f.Schedulers, sessions.Object, f.Storage.Object, svc, dialog.Object, timeline);
    f.Drain();
    return (vm, f, svc, dialog, timeline);
  }

  [Fact]
  public async Task Unload_removes_the_card_and_is_locked_while_playing()
  {
    var (vm, f, _, _, timeline) = Create();
    f.NativeAssets.Add(FakeAssetRuntime.Mesh(1, "comet"));
    f.NativeAssets.Add(FakeAssetRuntime.Texture(2, "rock"));
    var import = vm.ImportPathsAsync(["/data/comet.glb"]);
    f.CompleteImport(f.ImportRequests[0].RequestId, meshId: 1, added: 2);
    await import;
    f.Drain();
    var mesh = Assert.Single(vm.MeshAssets);
    Assert.True(vm.RemoveAssetCommand.CanExecute(mesh));

    // playing: locked
    f.Runtime.Setup(r => r.StartSimulation(It.IsAny<int>())).Returns(true);
    f.Runtime.Setup(r => r.SnapshotSceneSync()).Returns(true);
    timeline.Play(1);
    f.Drain();
    Assert.False(vm.CanUnload);
    Assert.False(vm.RemoveAssetCommand.CanExecute(mesh));

    f.Runtime.Setup(r => r.PauseSimulationSync()).Returns(true);
    timeline.Pause();
    f.Drain();
    Assert.True(vm.RemoveAssetCommand.CanExecute(mesh));

    var run = vm.RemoveAssetCommand.ExecuteAsync(mesh);
    f.Drain();
    Assert.True(vm.IsBusy);
    f.CompleteRemoval(f.RemoveRequests[0].RequestId, 1, ejected: 1);
    await run;
    f.Drain();
    Assert.Empty(vm.MeshAssets);
    Assert.Single(vm.TextureAssets);
    Assert.StartsWith("Tabs_Imports_StatusUnloaded", vm.StatusMessage);
  }

  [Fact]
  public async Task Import_command_imports_picked_file_and_fills_both_sections()
  {
    var (vm, f, _, dialog, _) = Create();
    dialog
      .Setup(d => d.ShowOpenFileDialogAsync(It.IsAny<string>(), It.IsAny<string[]?>()))
      .ReturnsAsync("/data/comet.glb");
    Assert.False(vm.HasMeshes);
    Assert.False(vm.HasTextures);

    var run = vm.ImportCommand.ExecuteAsync(null);
    f.Drain();
    Assert.True(vm.IsImporting);
    Assert.False(vm.ImportCommand.CanExecute(null));

    f.NativeAssets.Add(FakeAssetRuntime.Mesh(1, "comet", 2));
    f.NativeAssets.Add(FakeAssetRuntime.Texture(2, "comet_albedo"));
    f.CompleteImport(f.ImportRequests[0].RequestId, meshId: 1, added: 2);
    await run;
    f.Drain();

    Assert.Equal("comet", Assert.Single(vm.MeshAssets).Label);
    Assert.Equal("comet_albedo", Assert.Single(vm.TextureAssets).Label);
    Assert.True(vm.HasMeshes && vm.HasTextures);
    Assert.False(vm.IsImporting);
    Assert.Contains("comet.glb", vm.StatusMessage);
    Assert.StartsWith("Tabs_Imports_StatusImported", vm.StatusMessage);

    // the picker offers exactly the supported formats
    dialog.Verify(d => d.ShowOpenFileDialogAsync(
      It.IsAny<string>(),
      It.Is<string[]?>(x => x != null && x.SequenceEqual(AssetLibraryService.AllExtensions))));
  }

  [Fact]
  public async Task Cancelled_dialog_does_nothing()
  {
    var (vm, f, _, dialog, _) = Create();
    dialog
      .Setup(d => d.ShowOpenFileDialogAsync(It.IsAny<string>(), It.IsAny<string[]?>()))
      .ReturnsAsync((string?)null);
    await vm.ImportCommand.ExecuteAsync(null);
    Assert.Empty(f.ImportRequests);
    Assert.Equal(string.Empty, vm.StatusMessage);
  }

  [Fact]
  public async Task Reimport_reports_already_imported_and_does_not_duplicate()
  {
    var (vm, f, _, _, _) = Create();
    f.NativeAssets.Add(FakeAssetRuntime.Texture(1, "rock"));
    var first = vm.ImportPathsAsync(["/data/rock.png"]);
    f.CompleteImport(f.ImportRequests[0].RequestId, added: 1);
    await first;
    var second = vm.ImportPathsAsync(["/data/rock.png"]);
    f.CompleteImport(f.ImportRequests[1].RequestId, added: 0);
    await second;
    f.Drain();
    Assert.Single(vm.TextureAssets);
    Assert.StartsWith("Tabs_Imports_StatusAlreadyImported", vm.StatusMessage);
  }

  [Fact]
  public async Task Unsupported_file_reports_status()
  {
    var (vm, f, _, _, _) = Create();
    await vm.ImportPathsAsync(["/data/readme.md"]);
    Assert.Empty(f.ImportRequests);
    Assert.StartsWith("Tabs_Imports_StatusUnsupported", vm.StatusMessage);
  }
}

public class CometAppearanceViewModelTests
{
  private readonly FakeAssetRuntime _f = new();
  private readonly AssetLibraryService _svc;
  private readonly TimelineService _timeline;
  private readonly Mock<ITranslationService> _translation = new();
  private readonly List<CometAppearanceDto> _sent = [];
  private CometAppearanceStatus _nextStatus = CometAppearanceStatus.Queued;
  private CometAppearanceDto _native;

  public CometAppearanceViewModelTests()
  {
    _translation.Setup(t => t.CultureChanged).Returns(System.Reactive.Linq.Observable.Empty<System.Globalization.CultureInfo>());
    _translation.Setup(t => t.GetString(It.IsAny<string>())).Returns<string>(k => k);
    _f.Runtime
      .Setup(r => r.SetCometAppearance(It.IsAny<CometAppearanceDto>()))
      .Returns<CometAppearanceDto>(dto =>
      {
        _sent.Add(dto);
        if (_nextStatus == CometAppearanceStatus.Queued) _native = dto;
        return _nextStatus;
      });
    _f.Runtime
      .Setup(r => r.GetCometAppearance(out It.Ref<CometAppearanceDto>.IsAny))
      .Returns(new GetAppearance((out CometAppearanceDto a) => { a = _native; return true; }));

    _svc = _f.CreateService();
    var config = new CometConfigService(_f.Runtime.Object, _f.Schedulers);
    _timeline = new TimelineService(_f.Runtime.Object, _f.Schedulers, config, new BreadcrumbService());
  }

  private delegate bool GetAppearance(out CometAppearanceDto a);

  /// <summary>Library with one mesh (bundling albedo 10 + normal 11) and three textures.</summary>
  private void SeedLibrary()
  {
    _f.NativeAssets.Add(FakeAssetRuntime.Mesh(1, "comet", 10, 11));
    _f.NativeAssets.Add(FakeAssetRuntime.Mesh(2, "boulder"));
    _f.NativeAssets.Add(FakeAssetRuntime.Texture(10, "comet_albedo", TextureChannel.Albedo));
    _f.NativeAssets.Add(FakeAssetRuntime.Texture(11, "comet_normal", TextureChannel.Normal));
    _f.NativeAssets.Add(FakeAssetRuntime.Texture(12, "rock"));
    _svc.Refresh();
  }

  private CometAppearanceViewModel CreateVm()
  {
    var vm = new CometAppearanceViewModel(_f.Runtime.Object, _svc, _timeline, _translation.Object, _f.Schedulers);
    _f.Drain();
    return vm;
  }

  private void SetRunning(bool running)
  {
    if (running)
    {
      _f.Runtime.Setup(r => r.StartSimulation(It.IsAny<int>())).Returns(true);
      _f.Runtime.Setup(r => r.SnapshotSceneSync()).Returns(true);
      _timeline.Play(1);
    }
    else
    {
      _f.Runtime.Setup(r => r.PauseSimulationSync()).Returns(true);
      _timeline.Pause();
    }
    _f.Drain();
  }

  [Fact]
  public void Defaults_to_procedural_sphere_with_none_texture_options()
  {
    SeedLibrary();
    using var vm = CreateVm();
    Assert.True(vm.IsDefaultMode);
    Assert.False(vm.IsCustomMode);
    Assert.Equal(2, vm.MeshOptions.Count);
    Assert.Equal(0UL, vm.TextureOptions[0].Id); // "None" first
    Assert.Equal(4, vm.TextureOptions.Count);
    Assert.Equal(0UL, vm.SelectedAlbedo!.Id);
    Assert.True(vm.IsWiringEnabled);
    Assert.Empty(_sent); // loading state must not echo back to the runtime
  }

  [Fact]
  public void Initial_state_is_read_back_from_runtime()
  {
    SeedLibrary();
    _native = new CometAppearanceDto { Mode = CometDisplayMode.Custom, Mesh = 2, Roughness = 12, Yaw = 15 };
    using var vm = CreateVm();
    Assert.True(vm.IsCustomMode);
    Assert.Equal(2UL, vm.SelectedMesh!.Id);
    Assert.Equal(12UL, vm.SelectedRoughness!.Id);
    Assert.Equal(15, vm.Yaw, 3);
    Assert.Empty(_sent);
  }

  [Fact]
  public void Selecting_a_mesh_prewires_bundled_textures_in_one_dispatch()
  {
    SeedLibrary();
    using var vm = CreateVm();
    vm.IsCustomMode = true;
    Assert.Equal(CometDisplayMode.Custom, _sent.Last().Mode);
    int before = _sent.Count;

    vm.SelectedMesh = vm.MeshOptions.First(m => m.Id == 1);
    Assert.Equal(before + 1, _sent.Count);
    var dto = _sent.Last();
    Assert.Equal(1UL, dto.Mesh);
    Assert.Equal(10UL, dto.Albedo);
    Assert.Equal(11UL, dto.Normal);
    Assert.Equal(0UL, dto.Roughness);
    Assert.Equal(10UL, vm.SelectedAlbedo!.Id);
  }

  [Fact]
  public void Prewiring_never_overrides_an_explicit_choice()
  {
    SeedLibrary();
    using var vm = CreateVm();
    vm.IsCustomMode = true;
    vm.SelectedAlbedo = vm.TextureOptions.First(o => o.Id == 12);
    vm.SelectedMesh = vm.MeshOptions.First(m => m.Id == 1);
    Assert.Equal(12UL, _sent.Last().Albedo);
    Assert.Equal(11UL, _sent.Last().Normal);
  }

  [Fact]
  public void Wiring_is_locked_while_running_and_unlocked_when_paused()
  {
    SeedLibrary();
    using var vm = CreateVm();
    SetRunning(true);
    Assert.False(vm.IsWiringEnabled);
    Assert.True(vm.IsWiringLocked);
    SetRunning(false);
    Assert.True(vm.IsWiringEnabled);
  }

  [Fact]
  public void Wiring_is_locked_during_an_import()
  {
    SeedLibrary();
    using var vm = CreateVm();
    _ = _svc.ImportAsync("/data/new.glb");
    _f.Drain();
    Assert.False(vm.IsWiringEnabled);
    _f.CompleteImport(_f.ImportRequests[0].RequestId);
    Assert.True(vm.IsWiringEnabled);
  }

  [Fact]
  public void Refused_change_reverts_ui_to_last_applied_state()
  {
    SeedLibrary();
    using var vm = CreateVm();
    _nextStatus = CometAppearanceStatus.LockedWhileRunning;
    vm.IsCustomMode = true;
    Assert.False(vm.IsCustomMode, "UI must reflect what the runtime displays");
    Assert.Equal("Tabs_Settings_AppearanceLockedHint", vm.StatusMessage);

    _nextStatus = CometAppearanceStatus.Queued;
    vm.IsCustomMode = true;
    Assert.True(vm.IsCustomMode);
    Assert.Equal(string.Empty, vm.StatusMessage);
  }

  [Fact]
  public void Placement_changes_are_throttled_and_allowed_while_running()
  {
    SeedLibrary();
    using var vm = CreateVm();
    SetRunning(true);
    _sent.Clear();

    vm.Yaw = 10;
    vm.Yaw = 20;
    vm.TranslationX = 0.5;
    Assert.Empty(_sent); // nothing until the throttle window elapses

    _f.Schedulers.Background.AdvanceBy(CometAppearanceViewModel.PlacementThrottle.Ticks + 1);
    _f.Schedulers.MainThread.Start();
    var dto = Assert.Single(_sent);
    Assert.Equal(20f, dto.Yaw);
    Assert.Equal(0.5f, dto.TranslationX);
    Assert.Equal(CometDisplayMode.Default, dto.Mode); // wiring untouched
  }

  [Fact]
  public void Reset_placement_dispatches_identity()
  {
    SeedLibrary();
    _native = new CometAppearanceDto { Mode = CometDisplayMode.Custom, Mesh = 1, Yaw = 30, Pitch = -5, TranslationZ = 1 };
    using var vm = CreateVm();
    vm.ResetPlacementCommand.Execute(null);
    var dto = Assert.Single(_sent);
    Assert.Equal(0f, dto.Yaw);
    Assert.Equal(0f, dto.Pitch);
    Assert.Equal(0f, dto.TranslationZ);
    Assert.Equal(1UL, dto.Mesh);
  }

  [Fact]
  public void Library_refresh_keeps_current_selection()
  {
    SeedLibrary();
    using var vm = CreateVm();
    vm.IsCustomMode = true;
    vm.SelectedMesh = vm.MeshOptions.First(m => m.Id == 2);
    int sent = _sent.Count;

    _f.NativeAssets.Add(FakeAssetRuntime.Texture(13, "ice"));
    _svc.Refresh();
    _f.Drain();

    Assert.Equal(5, vm.TextureOptions.Count);
    Assert.Equal(2UL, vm.SelectedMesh!.Id);
    Assert.Equal(sent, _sent.Count); // rebuilding options must not re-dispatch
  }

  [Fact]
  public void Custom_mode_is_unavailable_without_meshes()
  {
    _f.NativeAssets.Add(FakeAssetRuntime.Texture(12, "rock"));
    _svc.Refresh();
    using var vm = CreateVm();
    Assert.False(vm.HasMeshes);
    Assert.False(vm.IsCustomModeAvailable);

    _f.NativeAssets.Add(FakeAssetRuntime.Mesh(1, "comet"));
    _svc.Refresh();
    _f.Drain();
    Assert.True(vm.IsCustomModeAvailable);

    SetRunning(true);
    Assert.False(vm.IsCustomModeAvailable, "locked while playing");
  }

  [Fact]
  public void Unloading_the_displayed_mesh_switches_ui_back_to_default()
  {
    SeedLibrary();
    using var vm = CreateVm();
    vm.IsCustomMode = true;
    vm.SelectedMesh = vm.MeshOptions.First(m => m.Id == 1);
    int sent = _sent.Count;

    // runtime ejects the comet (RemoveAsset) and the mesh leaves the library
    var task = _svc.RemoveAsync(1);
    _native = new CometAppearanceDto { Yaw = _native.Yaw };
    _f.CompleteRemoval(_f.RemoveRequests[0].RequestId, 1, ejected: 1);
    Assert.True(task.Result.Success);

    Assert.False(vm.IsCustomMode);
    Assert.Null(vm.SelectedMesh);
    Assert.DoesNotContain(vm.MeshOptions, m => m.Id == 1);
    Assert.Equal(sent, _sent.Count); // reflecting the runtime state must not dispatch
  }

  [Fact]
  public void Selecting_none_unwires_a_channel()
  {
    SeedLibrary();
    using var vm = CreateVm();
    vm.IsCustomMode = true;
    vm.SelectedMesh = vm.MeshOptions.First(m => m.Id == 1);
    vm.SelectedNormal = vm.TextureOptions[0];
    Assert.Equal(0UL, _sent.Last().Normal);
    Assert.Equal(10UL, _sent.Last().Albedo);
  }
}
