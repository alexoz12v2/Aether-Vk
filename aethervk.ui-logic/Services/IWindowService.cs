using System.Threading.Tasks;

namespace AetherVk.Logic.Services
{
  public interface IWindowService
  {
    Task ShowSpawnImageDialogAsync(string imagePath);
    Task ShowManageImportsDialogAsync();
    Task ShowSettingsDialogAsync();
    Task<ulong> ShowSpawnMeshDialogAsync(string modelId, string modelName);
    Task ShowSpawnBillboardDialogAsync();
    Task<(double X, double Y, double Z)?> ShowSnapObserverDialogAsync();
  }
}
