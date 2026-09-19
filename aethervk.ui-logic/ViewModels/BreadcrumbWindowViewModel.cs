using System;
using System.Collections.ObjectModel;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// The sole UI recipient of <see cref="BreadcrumbService"/> events.
/// Subscribes to <see cref="BreadcrumbService.Events"/> on the main thread and
/// builds the <see cref="Messages"/> collection that <c>BreadcrumbWindow</c> binds to.
/// </summary>
public partial class BreadcrumbWindowViewModel : ObservableObject, IDisposable
{
  private readonly CompositeDisposable _disposables = [];

  /// <summary>The View binds <c>ItemsSource</c> directly to this.</summary>
  public ObservableCollection<BreadcrumbMessage> Messages { get; } = [];

  public BreadcrumbWindowViewModel(
    BreadcrumbService breadcrumbService,
    ISchedulerProvider schedulerProvider)
  {
    breadcrumbService.Events
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(ev =>
      {
        switch (ev)
        {
          case BreadcrumbEvent.Added   a: Messages.Add(a.Message);    break;
          case BreadcrumbEvent.Removed r: Messages.Remove(r.Message); break;
        }
      })
      .AddDisposableTo(_disposables);
  }

  public void Dispose() => _disposables.Dispose();
}
