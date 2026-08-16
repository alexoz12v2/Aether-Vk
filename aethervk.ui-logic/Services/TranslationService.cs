using System;
using System.Collections.Generic;
using System.Globalization;
using System.Resources;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using AetherVk.Logic.Utils;

namespace AetherVk.Logic.Services;

public interface ITranslationService
{
  /// <summary>Gets or sets current culture</summary>
  CultureInfo CurrentCulture { get; set; }

  /// <summary>Fires whenever the culture changes. For UI subscriptions (View Model), `ObserveOn` UI
  /// Thread is mandatory</summary>
  IObservable<CultureInfo> CultureChanged { get; }

  /// <summary>Gets a string for the current culture immediately</summary>
  string GetString(string key);

  /// <summary>Gets an observable stream that emits the translated string initially
  /// and every time the culture changes. For UI subscriptions (View Model), `ObserveOn` UI Thread
  /// is mandatory</summary>
  IObservable<string> GetStringObservable(string key);

  // ── Named interpolation (AOT-safe, zero-reflection) ───────────────────────

  /// <summary>
  /// Immediately fetches the template for <paramref name="key"/> from the current
  /// culture and formats it with <paramref name="args"/> using
  /// <see cref="NamedFormatter"/>.
  /// </summary>
  /// <param name="key">Resource key.</param>
  /// <param name="args">
  /// Pre-rendered named values (e.g. <c>count.ToString("N0", culture)</c>).
  /// </param>
  string GetFormattedString(string key, IReadOnlyDictionary<string, string> args);

  /// <summary>
  /// Observable that emits a formatted string on subscribe and re-emits whenever
  /// the <b>culture</b> changes. Args are snapshotted at call time — suitable for
  /// values that do not change after the ViewModel is created.
  /// For UI subscriptions (View Model), <c>ObserveOn</c> UI Thread is mandatory.
  /// </summary>
  IObservable<string> GetFormattedObservable(
    string key,
    IReadOnlyDictionary<string, string> args);

  /// <summary>
  /// Observable that re-emits a formatted string whenever <b>either</b> the
  /// culture <b>or</b> <paramref name="argsStream"/> emits a new value (via
  /// <c>CombineLatest</c>). Suitable for runtime-changing values such as live
  /// simulation counters updated from <c>WeakReferenceMessenger</c> callbacks.
  /// For UI subscriptions (View Model), <c>ObserveOn</c> UI Thread is mandatory.
  /// </summary>
  IObservable<string> GetFormattedObservable(
    string key,
    IObservable<IReadOnlyDictionary<string, string>> argsStream);
}

public partial class TranslationService(ResourceManager resourceManager, CultureInfo? culture = null) : ITranslationService
{
  private readonly ResourceManager _resourceManager = resourceManager;
  private readonly BehaviorSubject<CultureInfo> _cultureSubject = new BehaviorSubject<CultureInfo>(culture ?? CultureInfo.CurrentUICulture);

  public CultureInfo CurrentCulture
  {
    get => _cultureSubject.Value;
    set
    {
      if (!Equals(_cultureSubject.Value, value))
      {
        // Note: Set also thread local cultures if used elsewhere (shouldn't be)
        // - Sets formatting (dates, numbers, currency)
        CultureInfo.CurrentCulture = value;
        // - sets localization for UI Strings (ResourceManager)
        CultureInfo.CurrentUICulture = value;

        // Broadcast culture changed
        _cultureSubject.OnNext(value);
      }
    }
  }

  // should be subscribed with main thread if in view model
  public IObservable<CultureInfo> CultureChanged => _cultureSubject.AsObservable();

  public string GetString(string key)
  {
    if (string.IsNullOrWhiteSpace(key)) return string.Empty;

    // fetch string from .resx using current reactive culture
    var translated = _resourceManager.GetString(key, CurrentCulture);

    // fallback to the key itself if translation is missing
    return translated ?? $"[{key}]";
  }

  public IObservable<string> GetStringObservable(string key)
  {
    // every time culture changes, fetch new string for this key
    return CultureChanged.Select(_ => GetString(key));
  }

  // ── Named interpolation ───────────────────────────────────────────────────

  public string GetFormattedString(string key, IReadOnlyDictionary<string, string> args)
  {
    var template = GetString(key); // returns "[key]" on miss, never throws
    return NamedFormatter.Format(template, args);
  }

  public IObservable<string> GetFormattedObservable(
    string key,
    IReadOnlyDictionary<string, string> args)
  {
    // Re-format with the same frozen args each time the culture changes
    return CultureChanged.Select(_ => GetFormattedString(key, args));
  }

  public IObservable<string> GetFormattedObservable(
    string key,
    IObservable<IReadOnlyDictionary<string, string>> argsStream)
  {
    // Re-format whenever culture OR args change
    return CultureChanged
      .CombineLatest(argsStream, (_, args) => GetFormattedString(key, args));
  }
}

