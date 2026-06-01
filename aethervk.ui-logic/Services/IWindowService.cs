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
      System.Collections.Generic.IEnumerable<ViewModels.ImportedModelItem> models,
      ulong? preselectedModelId = null
    );
    Task ShowSpawnBillboardDialogAsync();
    Task<(double X, double Y, double Z)?> ShowSnapObserverDialogAsync();
  }
}
