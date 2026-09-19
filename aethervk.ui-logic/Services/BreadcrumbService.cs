using System;
using System.Reactive.Subjects;
using System.Threading.Tasks;

namespace AetherVk.Logic.Services;

/// <summary>
/// Discriminated union emitted by <see cref="BreadcrumbService.Events"/>.
/// </summary>
public abstract record BreadcrumbEvent
{
  public sealed record Added(BreadcrumbMessage Message)   : BreadcrumbEvent;
  public sealed record Removed(BreadcrumbMessage Message) : BreadcrumbEvent;
}

public class BreadcrumbMessage
{
  public int    Status    { get; set; } // 0=Info, 1=Success, 2=Warning, 3=Error
  public string Title     { get; set; } = string.Empty;
  public string Content   { get; set; } = string.Empty;
  public bool   IsLoading { get; set; }
}

/// <summary>
/// Publishes breadcrumb lifecycle events via an Rx subject.
/// Callers use Show*/Dismiss to emit events; only
/// <see cref="AetherVk.Logic.ViewModels.BreadcrumbWindowViewModel"/> subscribes to display them.
/// </summary>
public sealed class BreadcrumbService : IDisposable
{
  private readonly Subject<BreadcrumbEvent> _subject = new();

  /// <summary>
  /// Observable stream of add/remove events. Subscribe with
  /// <c>.ObserveOn(schedulerProvider.MainThread)</c> before mutating UI collections.
  /// </summary>
  public IObservable<BreadcrumbEvent> Events => _subject;

  public BreadcrumbMessage ShowLoadingMessage(string title, string content)
  {
    var msg = new BreadcrumbMessage { Title = title, Content = content, IsLoading = true };
    _subject.OnNext(new BreadcrumbEvent.Added(msg));
    return msg;
  }

  public void RemoveMessage(BreadcrumbMessage msg)
    => _subject.OnNext(new BreadcrumbEvent.Removed(msg));

  public async Task ShowMessageAsync(
    string title,
    string content,
    TimeSpan duration = default,
    int status = 0)
  {
    if (duration == default) duration = TimeSpan.FromSeconds(3);
    var msg = new BreadcrumbMessage { Title = title, Content = content, Status = status };
    _subject.OnNext(new BreadcrumbEvent.Added(msg));
    await Task.Delay(duration);
    _subject.OnNext(new BreadcrumbEvent.Removed(msg));
  }

  public void ShowErrorMessage(string title, string content)
    => _ = ShowMessageAsync(title, content, TimeSpan.FromSeconds(6), status: 3);

  public void Dispose() => _subject.Dispose();
}
