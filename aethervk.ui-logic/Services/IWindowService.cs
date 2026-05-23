using System.Threading.Tasks;

namespace AetherVk.Logic.Services
{
  public interface IWindowService
  {
    Task ShowSpawnImageDialogAsync(string imagePath);
    Task ShowManageImportsDialogAsync();
    Task ShowSettingsDialogAsync();
    Task OpenMeshViewerAsync(string meshId);
    Task<ulong> ShowSpawnMeshDialogAsync(string modelId, string modelName);
    Task<ulong> ShowSpawnCometDialogAsync(
      System.Collections.Generic.IEnumerable<ViewModels.ImportedModelItem> models
    );
    Task ShowSpawnBillboardDialogAsync();
  }
}
