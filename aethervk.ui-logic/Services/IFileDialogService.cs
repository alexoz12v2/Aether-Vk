using System.Threading.Tasks;

namespace AetherVk.Logic.Services
{
  public interface IFileDialogService
  {
    Task<string?> ShowOpenFileDialogAsync(string title, string[]? filters = null);
    Task<string?> ShowSaveFileDialogAsync(
      string title,
      string defaultExtension,
      string[]? filters = null
    );

    /// <summary>Picks (or creates, where the platform dialog allows) a folder: the save folder of
    /// <i>File › Save As…</i>. Null when cancelled.</summary>
    Task<string?> PickFolderAsync(string title);
  }
}
