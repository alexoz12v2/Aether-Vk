using System.Threading.Tasks;

namespace AetherVk.Logic.Services
{
    public interface IFileDialogService
    {
        Task<string?> ShowOpenFileDialogAsync(string title, string[]? filters = null);
        Task<string?> ShowSaveFileDialogAsync(string title, string defaultExtension, string[]? filters = null);
    }
}
