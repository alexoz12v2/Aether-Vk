using System;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using AetherVk.Logic.Utils;

namespace AetherVk.Logic.ViewModels.Debug;

#if DEBUG
/// <summary>
/// Exposes a RenderDoc frame-capture trigger to the View layer.
///
/// Available in DEBUG builds only.
/// </summary>
public sealed partial class RenderDocCaptureViewModel : ObservableObject, IDisposable
{
  private readonly INativeRuntimeService _runtime;
  private readonly CompositeDisposable _disposables = [];

  /// <summary>
  /// <c>true</c> when the process was launched under RenderDoc and the
  /// in-app capture API is available. Drives the <c>IsVisible</c> binding
  /// on the capture button so the button is completely absent when irrelevant.
  /// </summary>
  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(CaptureCommand))]
  private bool _isRenderDocAvailable;

  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(CaptureCommand))]
  private bool _isSimulationRunning;

  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(CaptureCommand))]
  private bool _hasParticleSystems;

  /// <summary>Triggers a single-frame capture on the next present.</summary>
  public IRelayCommand CaptureCommand { get; }

  public RenderDocCaptureViewModel(
    INativeRuntimeService runtime,
    TimelineService timelineService,
    ITabStateService<ModelSession> modelSessionService,
    ISchedulerProvider schedulerProvider)
  {
    _runtime = runtime;

    // Probe once at construction time — the underlying Rust Once cell ensures
    // this is a cheap cached lookup on every subsequent call.
    _isRenderDocAvailable = runtime.IsRenderDocAvailable();

    CaptureCommand = new RelayCommand(
      execute: TriggerCapture,
      canExecute: () => _isRenderDocAvailable && IsSimulationRunning && HasParticleSystems
    );

    timelineService.IsSimulationRunning
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(running => IsSimulationRunning = running)
      .AddDisposableTo(_disposables);

    var modelSession = modelSessionService.GetSession(new SessionId(typeof(ModelSession), 1));
    if (modelSession != null)
    {
      // Observe Jets collection count
      Observable.FromEventPattern<System.Collections.Specialized.NotifyCollectionChangedEventHandler, System.Collections.Specialized.NotifyCollectionChangedEventArgs>(
          h => modelSession.Jets.CollectionChanged += h,
          h => modelSession.Jets.CollectionChanged -= h)
        .Select(_ => modelSession.Jets.Count > 0)
        .StartWith(modelSession.Jets.Count > 0)
        .ObserveOn(schedulerProvider.MainThread)
        .Subscribe(hasJets => HasParticleSystems = hasJets)
        .AddDisposableTo(_disposables);
    }
  }

  private void TriggerCapture()
  {
    var peId = _runtime.PresentationEngineId;
    if (peId.HasValue)
      _runtime.StartScopedRenderDocCapture(peId.Value);
    else
      _runtime.TriggerRenderDocCapture(); // fallback: PE not yet registered
  }

  public void Dispose()
  {
    _disposables.Dispose();
  }
}
#endif
