using System.Collections.Specialized;
using System.Linq;
using AetherVk.Logic.ViewModels;
using Avalonia.Controls;

namespace AetherVk.Views;

public partial class ConsoleView : UserControl
{
  public ConsoleView()
  {
    InitializeComponent();
    this.DataContextChanged += OnDataContextChanged;
  }

  private void OnDataContextChanged(object? sender, System.EventArgs e)
  {
    if (DataContext is ConsoleViewModel vm)
    {
      vm.Messages.CollectionChanged += OnMessagesCollectionChanged;
    }
  }

  private void OnMessagesCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
  {
    // Scroll to the end when a new message is added
    if (e.Action == NotifyCollectionChangedAction.Add)
    {
      var textBox = this.FindControl<TextBox>("ConsoleTextBox");
      if (textBox != null)
      {
        textBox.CaretIndex = textBox.Text?.Length ?? 0;
      }
    }
  }

  private void OnConsolePointerWheelChanged(object? sender, Avalonia.Input.PointerWheelEventArgs e)
  {
    if (e.KeyModifiers.HasFlag(Avalonia.Input.KeyModifiers.Control))
    {
      if (DataContext is ConsoleViewModel vm)
      {
        if (e.Delta.Y > 0)
        {
          vm.ZoomInCommand.Execute(null);
        }
        else if (e.Delta.Y < 0)
        {
          vm.ZoomOutCommand.Execute(null);
        }
        e.Handled = true;
      }
    }
  }

  private async void OnExportLogsClicked(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
  {
    if (DataContext is ConsoleViewModel vm)
    {
      var topLevel = TopLevel.GetTopLevel(this);
      if (topLevel == null)
        return;

      var file = await topLevel.StorageProvider.SaveFilePickerAsync(
        new Avalonia.Platform.Storage.FilePickerSaveOptions
        {
          Title = "Export Console Logs",
          DefaultExtension = "txt",
          SuggestedFileName = "AetherVk_ConsoleLogs",
          FileTypeChoices = new[]
          {
            new Avalonia.Platform.Storage.FilePickerFileType("Text files")
            {
              Patterns = new[] { "*.txt" },
            },
          },
        }
      );

      if (file != null)
      {
        try
        {
          var text = string.Join(System.Environment.NewLine, vm.Messages);
          await System.IO.File.WriteAllTextAsync(file.Path.LocalPath, text);
        }
        catch (System.Exception)
        {
          // Optionally log the error or show a dialog
        }
      }
    }
  }
}
