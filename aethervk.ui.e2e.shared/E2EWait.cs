// Source-linked into aethervk.ui.e2e (Windows) and aethervk.ui.e2e.mac.
using System;
using System.Threading;

namespace AetherVk.UI.E2E.Shared;

/// <summary>Polling with a deadline, driver-agnostic: a driver error inside the probe is retried.</summary>
public static class E2EWait
{
  public static readonly TimeSpan DefaultTimeout = TimeSpan.FromSeconds(20);

  /// <summary>Exceptions a probe may not swallow: the app is gone, or an assertion failed.</summary>
  public static Func<Exception, bool> IsFatal { get; set; } = e =>
    e.GetType().Name.Contains("AppGone", StringComparison.Ordinal)
    || e is Xunit.Sdk.XunitException
    || e is OutOfMemoryException;

  public static T For<T>(Func<T?> probe, TimeSpan? timeout = null, string? what = null) where T : class
  {
    var deadline = DateTime.UtcNow + (timeout ?? DefaultTimeout);
    Exception? last = null;
    while (DateTime.UtcNow < deadline)
    {
      try
      {
        if (probe() is { } value)
          return value;
      }
      catch (Exception e) when (!IsFatal(e))
      {
        last = e;
      }
      Thread.Sleep(250);
    }
    throw new TimeoutException($"timed out waiting for {what ?? "condition"}", last);
  }

  public static void Until(Func<bool> condition, TimeSpan? timeout = null, string? what = null) =>
    For(() => condition() ? "ok" : null, timeout, what);
}
