using System.Threading.Tasks;

namespace AetherVk.Logic.Services
{
  public interface IWindowService
  {
    Task ShowSpawnImageDialogAsync(string imagePath);
    Task ShowManageImportsDialogAsync();
    Task OpenMeshViewerAsync(string meshId);
    Task<ulong> ShowSpawnMeshDialogAsync(string modelId, string modelName);
  }
}
