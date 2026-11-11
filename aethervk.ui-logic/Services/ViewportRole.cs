using System.Collections.Generic;

namespace AetherVk.Logic.Services;

/// <summary>What a live viewport is for. The runtime attaches at most one viewport per role.</summary>
public enum ViewportRole
{
  /// <summary>The main 3-D viewport: camera hotkeys, operators, export, captures, saves.</summary>
  Main,

  /// <summary>
  /// The small nucleus view next to the main viewport: a presentation engine and its camera,
  /// nothing else. No input routing, no camera registration (so mode bar, settings, billboards
  /// and saves never see it); allowed only while a <see cref="Main"/> viewport is attached.
  /// </summary>
  NucleusMini,
}

/// <summary>
/// The role bookkeeping of the attached viewports (pure, thread-safe): which roles may attach,
/// the ids of each one and the order they are removed in at shutdown.
/// <list type="bullet">
///   <item><see cref="ViewportRole.Main"/>: once (a second main viewport crashed the render thread
///     at shutdown and stole the camera hotkeys from the first one).</item>
///   <item><see cref="ViewportRole.NucleusMini"/>: once, and only while a main viewport exists.</item>
/// </list>
/// </summary>
public sealed class ViewportRoleTable
{
  private readonly object _gate = new();
  private ulong? _mainPe, _mainCamera, _miniPe, _miniCamera;

  public ulong? MainPresentationEngineId { get { lock (_gate) return _mainPe; } }
  public ulong? MainCameraEntityId { get { lock (_gate) return _mainCamera; } }
  public ulong? NucleusMiniPresentationEngineId { get { lock (_gate) return _miniPe; } }
  public ulong? NucleusMiniCameraEntityId { get { lock (_gate) return _miniCamera; } }

  /// <summary>Whether a viewport of <paramref name="role"/> may attach now; <paramref name="refusal"/>
  /// says why not.</summary>
  public bool CanAdd(ViewportRole role, out string? refusal)
  {
    lock (_gate)
    {
      refusal = role switch
      {
        ViewportRole.Main when _mainPe is not null =>
          $"the main viewport PE={_mainPe} is already attached",
        ViewportRole.NucleusMini when _mainPe is null =>
          "a nucleus mini-view needs the main viewport attached first",
        ViewportRole.NucleusMini when _miniPe is not null =>
          $"the nucleus mini-view PE={_miniPe} is already attached",
        ViewportRole.Main or ViewportRole.NucleusMini => null,
        _ => $"unknown viewport role {role}",
      };
      return refusal is null;
    }
  }

  /// <summary>Records an attached viewport (after a successful <see cref="CanAdd"/> and native add);
  /// false, recording nothing, when the role was taken in between (the caller removes the engine).</summary>
  public bool TryAdd(ViewportRole role, ulong presentationEngineId, ulong cameraEntityId)
  {
    lock (_gate)
    {
      if (!CanAdd(role, out _))
        return false;
      if (role == ViewportRole.Main)
      {
        _mainPe = presentationEngineId;
        _mainCamera = cameraEntityId;
      }
      else
      {
        _miniPe = presentationEngineId;
        _miniCamera = cameraEntityId;
      }
      return true;
    }
  }

  /// <summary>Forgets the viewport of <paramref name="presentationEngineId"/>; its role, or null
  /// when no attached viewport has that engine.</summary>
  public ViewportRole? Removed(ulong presentationEngineId)
  {
    lock (_gate)
    {
      if (presentationEngineId != 0 && _miniPe == presentationEngineId)
      {
        _miniPe = null;
        _miniCamera = null;
        return ViewportRole.NucleusMini;
      }
      if (presentationEngineId != 0 && _mainPe == presentationEngineId)
      {
        _mainPe = null;
        _mainCamera = null;
        return ViewportRole.Main;
      }
      return null;
    }
  }

  /// <summary>The attached engines in removal order: the mini-view first, the main viewport last
  /// (none may present to a closed window, and the mini exists only beside a main one).</summary>
  public IReadOnlyList<ulong> ShutdownOrder()
  {
    lock (_gate)
    {
      var order = new List<ulong>(2);
      if (_miniPe is ulong mini)
        order.Add(mini);
      if (_mainPe is ulong main)
        order.Add(main);
      return order;
    }
  }
}

public static class ViewportShutdown
{
  /// <summary>
  /// Removes every attached viewport, the nucleus mini-view first, then the main one, while their
  /// windows still exist (presenting to a destroyed window deadlocked the render thread at
  /// shutdown). Idempotent.
  /// </summary>
  public static void RemoveAllViewports(this INativeRuntimeService runtime)
  {
    if (runtime.NucleusMiniPresentationEngineId is ulong mini)
      runtime.RemoveViewport(mini);
    if (runtime.PresentationEngineId is ulong main)
      runtime.RemoveViewport(main);
  }
}
