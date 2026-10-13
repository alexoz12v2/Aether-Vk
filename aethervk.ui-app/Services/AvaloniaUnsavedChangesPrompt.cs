using System.Threading.Tasks;
using AetherVk.Logic.Services;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Layout;

namespace AetherVk.Services;

/// <summary>A small modal dialog: Save / Don't save / Cancel.</summary>
public sealed class AvaloniaUnsavedChangesPrompt : IUnsavedChangesPrompt
{
  public async Task<UnsavedChoice> AskAsync(string saveName)
  {
    if (Application.Current?.ApplicationLifetime is not IClassicDesktopStyleApplicationLifetime { MainWindow: { } main })
      return UnsavedChoice.Cancel;
    var dialog = BuildDialog(saveName);
    var result = await dialog.ShowDialog<UnsavedChoice?>(main);
    return result ?? UnsavedChoice.Cancel;
  }

  /// <summary>The dialog (buttons named Save, Discard, Cancel); closing it otherwise cancels.</summary>
  public static Window BuildDialog(string saveName)
  {
    var dialog = new Window
    {
      Title = "Unsaved changes",
      SizeToContent = SizeToContent.WidthAndHeight,
      CanResize = false,
      WindowStartupLocation = WindowStartupLocation.CenterOwner,
      ShowInTaskbar = false,
    };
    Button Make(string name, string text, UnsavedChoice choice)
    {
      var b = new Button { Name = name, Content = text, MinWidth = 90 };
      b.Click += (_, _) => dialog.Close(choice);
      return b;
    }
    var save = Make("Save", "Save", UnsavedChoice.Save);
    save.IsDefault = true;
    var cancel = Make("Cancel", "Cancel", UnsavedChoice.Cancel);
    cancel.IsCancel = true;
    dialog.Content = new StackPanel
    {
      Margin = new Thickness(20),
      Spacing = 16,
      Children =
      {
        new TextBlock { Text = $"Save the changes to \"{saveName}\"?", MaxWidth = 420, TextWrapping = Avalonia.Media.TextWrapping.Wrap },
        new StackPanel
        {
          Orientation = Orientation.Horizontal,
          HorizontalAlignment = HorizontalAlignment.Right,
          Spacing = 8,
          Children = { save, Make("Discard", "Don't save", UnsavedChoice.Discard), cancel },
        },
      },
    };
    return dialog;
  }
}
