using System;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using System.Threading;
using System.Threading.Tasks;

namespace AetherVk.Logic.Services;

/// <summary>
/// Orchestrates the full comet SPK commitment lifecycle.
///
/// <para><b>Responsibilities:</b></para>
/// <list type="bullet">
///   <item>Listens to <c>ExternalState::AlmanacImported</c> (state_id = 3) from the native runtime.</item>
///   <item>After the callback fires, calls <c>ReconfigureComet(ATTACH)</c> to trigger
///     <c>InitComet</c> in the Rust logic thread (attach <c>AlmanacPlanet</c>, force-reposition,
///     queue trajectory).</item>
///   <item>Exposes <see cref="IsAlmanacCommitted"/> and <see cref="CommittedSpkId"/> observables
///     for the UI layer to react to.</item>
///   <item>Owns <see cref="DecommitComet"/> (DETACH path) and
///     <see cref="SetRotationalModel"/> (live push to ECS).</item>
/// </list>
///
/// <para>- Part of the "Companion Runtime Service" group.</para>
/// </summary>
/// <seealso cref="CameraService" />
/// <seealso cref="CometPositionTrackerService" />
/// <seealso cref="TimelineService" />
/// <seealso cref="ImportedModelsTrackerService" />
public sealed class CometConfigService : IDisposable
{
  private readonly INativeRuntimeService _runtimeService;
  private readonly ISchedulerProvider _schedulerProvider;

  // ── State subjects ────────────────────────────────────────────────────────

  private readonly BehaviorSubject<bool> _isCommittedSubject = new(false);
  private readonly BehaviorSubject<int?> _committedSpkIdSubject = new(null);

  // ── Pending commit state ──────────────────────────────────────────────────
  // When CommitCometAsync is in flight, these fields track what we are committing.

  // Note: instead of using `volatile` on these 3 fields, we prefer explicitly using
  // Volatile.Read/Volatile.Write hwen thread safety is needed
  // Note: instead of 3 Volatile.Write, we group these in a record, so that we can atomically write
  // all 3 fields together
  private record class PendingState(
    TaskCompletionSource<bool>? CommitTcs,
    int SpkId,
    string? FilePath
  );

  private PendingState? _pendingState = new(null, 0, null);

  // Last committed values (used for decommit)
  private int _lastCommittedSpkId;
  private string? _lastCommittedFilePath;

  // ── Listener token ────────────────────────────────────────────────────────

  private readonly IDisposable _almanacListenerToken;

  // ── Constructor ───────────────────────────────────────────────────────────

  public CometConfigService(
    INativeRuntimeService runtimeService,
    ISchedulerProvider schedulerProvider
  )
  {
    _runtimeService = runtimeService;
    _schedulerProvider = schedulerProvider;

    // Register permanent listener for AlmanacImported (state_id = 3).
    // This fires every time the Rust logic thread finishes loading any almanac file.
    _almanacListenerToken = runtimeService.RegisterExternalStateListener(
      ExternalStateType.AlmanacImported,
      HandleAlmanacImportedCallback
    );
  }

  // ── Observables ───────────────────────────────────────────────────────────

  /// <summary>
  /// <c>true</c> once the comet <c>AlmanacPlanet</c> is attached to the native entity
  /// and <c>force_reposition</c> has completed. Observed on the main-thread scheduler.
  /// </summary>
  public IObservable<bool> IsAlmanacCommitted =>
    _isCommittedSubject.ObserveOn(_schedulerProvider.MainThread);

  /// <summary>
  /// Synchronous read of the committed state (for callers that cannot subscribe to an observable).
  /// </summary>
  public bool IsAlmanacCommittedValue => _isCommittedSubject.Value;

  /// <summary>
  /// The NAIF SPK id of the currently committed comet, or <c>null</c>.
  /// Observed on the main-thread scheduler.
  /// </summary>
  public IObservable<int?> CommittedSpkId =>
    _committedSpkIdSubject.ObserveOn(_schedulerProvider.MainThread);

  // ── Commands ──────────────────────────────────────────────────────────────

  /// <summary>
  /// Full commit pipeline:
  /// <list type="number">
  ///   <item>Load the SPK file into the native almanac (async — waits for
  ///     <c>ExternalState::AlmanacImported</c> callback).</item>
  ///   <item>Call <c>ReconfigureComet(ATTACH=0x1, naifId)</c> to trigger
  ///     <c>InitComet</c> in the Rust logic thread.</item>
  ///   <item>Emit <c>IsAlmanacCommitted = true</c>.</item>
  /// </list>
  /// Returns <c>false</c> on failure (enqueue error or 30 s timeout).
  /// </summary>
  public async Task<bool> CommitCometAsync(
    string spkFilePath,
    int naifId,
    CancellationToken ct = default
  )
  {
    // Allow only one concurrent commit.
    var tcs = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
    Volatile.Write(ref _pendingState, new PendingState(tcs, naifId, spkFilePath));

    try
    {
      // Link: caller token + service shutdown token + 30 s wall-clock timeout.
      // The shutdown token fires immediately when NativeRuntimeService.Dispose() is called,
      // so an in-flight commit doesn't hang until the timeout when the app is closing.
      using var cts = CancellationTokenSource.CreateLinkedTokenSource(
        ct,
        _runtimeService.ShutdownToken
      );
      cts.CancelAfter(TimeSpan.FromSeconds(30));
      using (cts.Token.Register(() => tcs.TrySetResult(false)))
      {
        // LoadAlmanacFileAsync enqueues the load to the Rust logic thread and registers
        // its own one-shot listener. Our permanent listener (HandleAlmanacImportedCallback)
        // will also fire — we use the permanent listener to drive the commit sequence.
        _ = _runtimeService.LoadAlmanacFileAsync(spkFilePath);
        return await tcs.Task;
      }
    }
    catch
    {
      tcs.TrySetResult(false);
      return false;
    }
    finally
    {
      UpdateTcs(null);
    }
  }

  /// <summary>
  /// Detaches the current comet almanac:
  /// <list type="number">
  ///   <item>Calls <c>ReconfigureComet(DETACH=0x2)</c> → Rust <c>CleanupComet</c>
  ///     (removes <c>AlmanacPlanet</c>, resets subtree to 1 AU +X).</item>
  ///   <item>Calls <c>UnloadAlmanacFile</c> to free the SPK from the almanac store.</item>
  ///   <item>Emits <c>IsAlmanacCommitted = false</c>.</item>
  /// </list>
  /// Returns <c>true</c> if both native calls succeeded.
  /// </summary>
  public bool DecommitComet()
  {
    if (!_isCommittedSubject.Value)
      return true; // already clean

    bool detachOk = _runtimeService.ReconfigureComet(
      commandFlags: 0x2,
      spkId: _lastCommittedSpkId,
      out _
    );

    bool unloadOk = true;
    if (_lastCommittedFilePath is not null)
      unloadOk = _runtimeService.UnloadAlmanacFile(_lastCommittedFilePath);

    _lastCommittedSpkId = 0;
    _lastCommittedFilePath = null;
    _isCommittedSubject.OnNext(false);
    _committedSpkIdSubject.OnNext(null);

    return detachOk && unloadOk;
  }

  /// <summary>
  /// Synchronously pushes updated IAU rotational model parameters to the native ECS.
  /// No-op unless the almanac is currently committed.
  /// The logic thread picks up the new <c>BodyRotationalModel</c> on its next tick (~16 ms).
  /// </summary>
  public void SetRotationalModel(BodyRotationalModelDto dto)
  {
    if (!_isCommittedSubject.Value)
      return;

    // CometEntityId is populated at startup from CStartupReturn.CometPlanetEntity.
    if (_runtimeService.CometEntityId is ulong cometBodyId)
      _runtimeService.SetBodyRotationalModel(cometBodyId, dto);
  }

  // ── Internal callback handling ────────────────────────────────────────────

  // Invoked on the native callback thread — must not block, must not throw.
  private unsafe void HandleAlmanacImportedCallback(nint dataPtr)
  {
    var pending = PendingTcs;
    if (pending is null)
      return; // no commit in flight — ignore

    // Only act on a successful load event (operation = 1).
    // Unload events (operation = 2) and load-failure events (operation = 0) are ignored here;
    // the pending-null check above already handles the unload path safely.
    var dto = *(CAlmanacImportedDTO*)dataPtr;
    if (dto.Operation != 1)
      return;

    int spkId = PendingSpkId;
    string? filePath = PendingFilePath;

    try
    {
      // Step 2 — tell the Rust logic thread to attach AlmanacPlanet + force-reposition.
      bool attached = _runtimeService.ReconfigureComet(commandFlags: 0x1, spkId: spkId, out _);

      if (attached)
      {
        _lastCommittedSpkId = spkId;
        _lastCommittedFilePath = filePath;
        _isCommittedSubject.OnNext(true);
        _committedSpkIdSubject.OnNext(spkId);
        pending.TrySetResult(true);
      }
      else
      {
        pending.TrySetResult(false);
      }
    }
    catch
    {
      pending.TrySetResult(false);
    }
  }

  // Update Pending state utility
  private void UpdateState(Func<PendingState?, PendingState> mutate)
  {
    while (true)
    {
      var current = Volatile.Read(ref _pendingState);
      var next = mutate(current);

      if (Interlocked.CompareExchange(ref _pendingState, next, current) == current)
        break;
    }
  }

  private void UpdateTcs(TaskCompletionSource<bool>? newTcs) =>
    UpdateState(c => c == null ? new PendingState(newTcs, 0, null) : c with { CommitTcs = newTcs });

  private TaskCompletionSource<bool>? PendingTcs => Volatile.Read(ref _pendingState)?.CommitTcs;

  private int PendingSpkId => Volatile.Read(ref _pendingState)?.SpkId ?? 0; // Defaulting to 0 if null

  private string? PendingFilePath => Volatile.Read(ref _pendingState)?.FilePath;

  // ── IDisposable ───────────────────────────────────────────────────────────

  public void Dispose()
  {
    _almanacListenerToken.Dispose();
    _isCommittedSubject.Dispose();
    _committedSpkIdSubject.Dispose();
  }
}
