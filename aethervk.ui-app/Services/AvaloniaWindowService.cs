using System;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;

namespace AetherVk.Services
{
  public class AvaloniaWindowService : IWindowService
  {
    private readonly NativeRuntimeService _runtimeService;
    private readonly BreadcrumbService _breadcrumbService;
    private readonly FileWatcherService _fileWatcherService;
    private readonly ConsoleService _consoleService;
    private readonly IUiThreadDispatcher _uiThreadDispatcher;
    private readonly HorizonJplService _horizonService;
    private readonly SceneStateManager _sceneStateManager;
    private readonly TimelineService _timelineService;

    public AvaloniaWindowService(
      NativeRuntimeService runtimeService,
      BreadcrumbService breadcrumbService,
      FileWatcherService fileWatcherService,
      ConsoleService consoleService,
      IUiThreadDispatcher uiThreadDispatcher,
      HorizonJplService horizonService,
      SceneStateManager sceneStateManager,
      TimelineService timelineService
    )
    {
      _runtimeService = runtimeService;
      _breadcrumbService = breadcrumbService;
      _fileWatcherService = fileWatcherService;
      _consoleService = consoleService;
      _uiThreadDispatcher = uiThreadDispatcher;
      _horizonService = horizonService;
      _sceneStateManager = sceneStateManager;
      _timelineService = timelineService;
    }

    private Window? GetMainWindow()
    {
      if (
        Application.Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop
      )
      {
        return desktop.MainWindow;
      }

      return null;
    }

    public async Task ShowSpawnImageDialogAsync(string imagePath)
    {
      var mainWindow = GetMainWindow();
      if (mainWindow == null)
        return;

      try
      {
        // Parse dimensions
        using var stream = System.IO.File.OpenRead(imagePath);
        var bitmap = new Avalonia.Media.Imaging.Bitmap(stream);
        float width = (float)bitmap.Size.Width;
        float height = (float)bitmap.Size.Height;

        var fileName = System.IO.Path.GetFileNameWithoutExtension(imagePath);

        var spawnDialog = new Views.SpawnImageDialogWindow
        {
          DataContext = new SpawnImageViewModel(fileName + " Billboard", width, height),
        };

        var dlgResult = await spawnDialog.ShowDialog<bool>(mainWindow);
        if (dlgResult && spawnDialog.DataContext is SpawnImageViewModel vm)
        {
          var entity = _runtimeService.SpawnImageBillboard(
            1,
            vm.EntityName,
            vm.IsScreenSpace,
            vm.Width,
            vm.Height
          );

          _fileWatcherService.WatchImageFile(imagePath, entity);
        }
      }
      catch (Exception ex)
      {
        _breadcrumbService.ShowMessageAsync(
          "Import Error",
          $"Failed to load image: {ex.Message}",
          default,
          3
        );
      }
    }

    public async Task ShowManageImportsDialogAsync()
    {
      var mainWindow = GetMainWindow();
      if (mainWindow == null)
        return;

      var window = new Views.ManageImportsWindow { DataContext = mainWindow.DataContext };
      await window.ShowDialog(mainWindow);
    }

    public async Task ShowSettingsDialogAsync()
    {
      var mainWindow = GetMainWindow();
      if (mainWindow == null)
        return;

      var inputRegistry =
        Microsoft.Extensions.DependencyInjection.ServiceProviderServiceExtensions.GetRequiredService<AetherVk.Logic.Input.InputRegistry>(
          App.Host!.Services
        );

      var settingsWindow = new Views.SettingsWindow
      {
        DataContext = new SettingsViewModel(inputRegistry),
      };

      await settingsWindow.ShowDialog(mainWindow);
    }

    public Task OpenMeshViewerAsync(string meshId)
    {
      var mainWindow = GetMainWindow();
      if (mainWindow == null)
        return Task.CompletedTask;

      var isLightTheme =
        Application.Current?.ActualThemeVariant == Avalonia.Styling.ThemeVariant.Light;
      var vm = mainWindow.DataContext as MainWindowViewModel;
      var model =
        vm != null
          ? System.Linq.Enumerable.FirstOrDefault(vm.ImportedModels, m => m.Id.ToString() == meshId)
          : null;

      if (model != null)
      {
        var meshViewer = new Views.MeshViewerWindow
        {
          DataContext = new MeshViewerViewModel(
            model.Id,
            model.FullPath,
            model.Name,
            isLightTheme,
            _runtimeService,
            _consoleService,
            _uiThreadDispatcher
          ),
        };
        var inputRegistry =
          Microsoft.Extensions.DependencyInjection.ServiceProviderServiceExtensions.GetRequiredService<AetherVk.Logic.Input.InputRegistry>(
            App.Host!.Services
          );
        _ = new AetherVk.Input.GlobalInputRouter(meshViewer, inputRegistry);
        meshViewer.Show(mainWindow);
      }

      return Task.CompletedTask;
    }

    public async Task<ulong> ShowSpawnMeshDialogAsync(string modelId, string modelName)
    {
      var mainWindow = GetMainWindow();
      if (mainWindow == null)
        return 0;

      var dialog = new Views.SpawnMeshDialogWindow
      {
        DataContext = new SpawnMeshViewModel(modelName + " Instance"),
      };

      var result = await dialog.ShowDialog<bool>(mainWindow);
      if (result && dialog.DataContext is SpawnMeshViewModel vm)
      {
        return await _runtimeService.SpawnModelInstanceAsync(
          1,
          ulong.Parse(modelId),
          vm.EntityName,
          vm.PosX,
          vm.PosY,
          vm.PosZ
        );
      }

      return 0;
    }

    public async Task<ulong> ShowSpawnCometDialogAsync(
      System.Collections.Generic.IEnumerable<AetherVk.Logic.ViewModels.ImportedModelItem> models,
      ulong? preselectedModelId = null
    )
    {
      var mainWindow = GetMainWindow();
      if (mainWindow == null)
        return 0;

      if (_sceneStateManager.HasComet(1))
      {
        _breadcrumbService.ShowMessageAsync(
          "Cannot Spawn Comet",
          "The scene already contains a comet. Only 1 comet is allowed.",
          default,
          5
        );
        return 0;
      }

      var dialog = new Views.SpawnCometWindow
      {
        DataContext = new SpawnCometViewModel(
          models,
          _horizonService,
          _timelineService,
          _breadcrumbService,
          preselectedModelId
        ),
      };
      var result = await dialog.ShowDialog<SpawnCometResult?>(mainWindow);
      if (result == null)
        return 0;

      uint physicsTypeIdx = result.PhysicsType switch
      {
        "Kinematic" => 1,
        "Dynamic" => 2,
        _ => 0, // Static
      };

      // ── Determine spawn position ─────────────────────────────────────────────
      float spawnPx = result.PosX,
        spawnPy = result.PosY,
        spawnPz = result.PosZ;
      float spawnVx = 0f,
        spawnVy = 0f,
        spawnVz = 0f;

      if (result.PhysicsType == "Dynamic" && result.OrbitData != null)
      {
        // Derive initial Cartesian state from EPA osculating elements.
        bool ok = NativeRuntimeService.KeplerToCartesian(
          result.OrbitData.SemiMajorAxisAu,
          result.OrbitData.Eccentricity,
          result.OrbitData.Inclination,
          result.OrbitData.AscendingNodeLongitude,
          result.OrbitData.ArgumentOfPerifocus,
          result.OrbitData.MeanAnomaly,
          out double kPx,
          out double kPy,
          out double kPz,
          out double kVx,
          out double kVy,
          out double kVz
        );
        if (ok)
        {
          spawnPx = (float)kPx;
          spawnPy = (float)kPy;
          spawnPz = (float)kPz;
          spawnVx = (float)kVx;
          spawnVy = (float)kVy;
          spawnVz = (float)kVz;
          _consoleService.Log(
            $"[Spawn] Dynamic initial pos=({spawnPx:F4},{spawnPy:F4},{spawnPz:F4}) AU, "
              + $"vel=({spawnVx:F4},{spawnVy:F4},{spawnVz:F4}) km/s"
          );
        }
        else
        {
          _consoleService.Log("[Spawn] KeplerToCartesian failed (e>=1). Using zero position.");
        }
      }

      // ── SpawnComet ───────────────────────────────────────────────────────────
      var (_, id) = _runtimeService.SpawnComet(
        sceneId: 1,
        modelId: result.Model.Id,
        name: result.EntityName,
        posX: spawnPx,
        posY: spawnPy,
        posZ: spawnPz,
        rotW: result.RotW,
        rotX: result.RotX,
        rotY: result.RotY,
        rotZ: result.RotZ,
        radiusKm: result.CometRadiusKm,
        massKg: result.MassKg,
        physicsType: physicsTypeIdx
      );
      ulong cometId = id;

      if (cometId > 0)
      {
        _sceneStateManager.SetComet(1, cometId);
      }
      else
      {
        await _breadcrumbService.ShowMessageAsync(
          "Spawn Error",
          "SpawnComet returned entity id 0.",
          default,
          3
        );
        return 0;
      }

      // ── Physics-mode post-spawn wiring ───────────────────────────────────────
      if (result.PhysicsType == "Kinematic" && result.SpkNaifId != 0)
      {
        // 1. Download and load the .bsp ephemeris file from JPL Horizons.
        _consoleService.Log($"[Spawn] Kinematic: downloading SPK {result.SpkNaifId}…");
        var spkLoadMsg = _breadcrumbService.ShowLoadingMessage(
          "Kinematic Setup",
          "Downloading SPK ephemeris…"
        );
        try
        {
          var savePath = _horizonService.GetSpkSavePath(result.SpkNaifId);
          var startStr = DateTime.UtcNow.AddYears(-5).ToString("yyyy-MM-dd");
          var stopStr = DateTime.UtcNow.AddYears(+5).ToString("yyyy-MM-dd");

          var spkPath = await _horizonService.DownloadSpkByIdAsync(
            result.CometDesignation ?? result.SpkNaifId.ToString(),
            result.SpkRecordId ?? result.SpkNaifId.ToString(),
            savePath,
            startStr,
            stopStr
          );

          if (spkPath != null)
          {
            // 2. Load .bsp into Rust AlmanacPackedData.
            NativeInterop.avkSimulationContext_loadAlmanacFile(
              _runtimeService.SimulationContext,
              spkPath
            );
            _consoleService.Log($"[Spawn] Kinematic: loaded almanac from {spkPath}.");

            // Allow the async almanac-load task a moment to register before patching the id.
            await Task.Delay(200);

            // 3. Patch the comet entity's AlmanacPlanet.naif_id with the real NAIF id.
            bool patched = _runtimeService.SetAlmanacPlanetNaifId(1, cometId, result.SpkNaifId);
            _consoleService.Log(
              $"[Spawn] Kinematic: SetAlmanacPlanetNaifId({result.SpkNaifId}) → {patched}"
            );
          }
          else
          {
            await _breadcrumbService.ShowMessageAsync(
              "SPK Download Failed",
              $"Could not download SPK for record {result.SpkRecordId}. Kinematic comet will not move.",
              default,
              4
            );
          }
        }
        catch (Exception ex)
        {
          _consoleService.Log($"[Spawn] Kinematic SPK setup error: {ex.Message}");
        }
        finally
        {
          _breadcrumbService.RemoveMessage(spkLoadMsg);
        }
      }
      else if (result.PhysicsType == "Dynamic")
      {
        // Inject initial orbital velocity.
        bool velSet = _runtimeService.SetKinematicVelocity(1, cometId, spawnVx, spawnVy, spawnVz);
        _consoleService.Log(
          $"[Spawn] Dynamic: SetKinematicVelocity({spawnVx:F4},{spawnVy:F4},{spawnVz:F4} km/s) → {velSet}"
        );
      }

      // ── Orbit trajectory (hidden by default, named orbit_{name}, child of root) ──
      if (result.OrbitData != null)
      {
        var orbitName = $"orbit_{result.EntityName}";
        ulong orbitId = _runtimeService.SpawnTrajectoryFromElements(
          1,
          orbitName,
          result.OrbitData.SemiMajorAxisAu,
          result.OrbitData.Eccentricity,
          result.OrbitData.Inclination,
          result.OrbitData.AscendingNodeLongitude,
          result.OrbitData.ArgumentOfPerifocus
        );
        if (orbitId > 0)
        {
          // Hide immediately via the synced wrapper so C# Entity.IsVisible and
          // the Rust HiddenComponent are set atomically; the outline eye-icon will
          // correctly reflect the hidden state and toggle on first click.
          _runtimeService.SetEntityHidden(1, orbitId, visible: false);
          _consoleService.Log(
            $"[Spawn] Orbit trajectory '{orbitName}' spawned (hidden, id={orbitId})."
          );
        }
      }

      return cometId;
    }

    public async Task ShowSpawnBillboardDialogAsync()
    {
      var mainWindow = GetMainWindow();
      if (mainWindow == null)
        return;

      var dialog = new Avalonia.Controls.OpenFileDialog
      {
        Title = "Select Billboard Image",
        AllowMultiple = false,
      };

      var result = await dialog.ShowAsync(mainWindow);
      if (result != null && result.Length > 0)
      {
        string imagePath = result[0];
        _runtimeService.SpawnBillboard(1, imagePath, 0.5f, 0.5f, 1.0f, 1.0f, 0); // Center, default scale/opacity, unscoped viewport
      }
    }

    public async Task<(double X, double Y, double Z)?> ShowSnapObserverDialogAsync()
    {
      var mainWindow = GetMainWindow();
      if (mainWindow == null)
        return null;

      var dialog = new Views.SnapObserverWindow();
      var vm = new AetherVk.Logic.ViewModels.SnapObserverViewModel();
      dialog.DataContext = vm;

      var result = await dialog.ShowDialog<(double, double, double)?>(mainWindow);
      return result;
    }
  }
}
