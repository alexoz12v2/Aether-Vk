using System;
using System.Collections.ObjectModel;
using System.Threading.Tasks;

namespace AetherVk.Logic.Services;

public class BreadcrumbMessage
{
  public int Status { get; set; } // 0=Info, 1=Success, 2=Warning, 3=Error
  public string Title { get; set; } = string.Empty;
  public string Content { get; set; } = string.Empty;
}

public class BreadcrumbService
{
  private readonly IUiThreadDispatcher _dispatcher;
  public ObservableCollection<BreadcrumbMessage> Messages { get; } = new();

  public BreadcrumbService(IUiThreadDispatcher dispatcher)
  {
    _dispatcher = dispatcher;
  }

  public async Task ShowMessageAsync(
    string title,
    string content,
    TimeSpan duration = default,
    int status = 0
  )
  {
    if (duration == default)
    {
      duration = TimeSpan.FromSeconds(3);
    }

    var msg = new BreadcrumbMessage
    {
      Title = title,
      Content = content,
      Status = status,
    };

    _dispatcher.Dispatch(() => Messages.Add(msg));

    await Task.Delay(duration);

    _dispatcher.Dispatch(() => Messages.Remove(msg));
  }
}
