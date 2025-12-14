using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.Logging;

namespace AetherVk.Core.ViewModels
{
    public sealed partial class EditorPageSplashScreenViewModel(ILogger<Index> logger) : ObservableObject
    {

        [RelayCommand]
        public void OnLog()
        {
            logger.LogInformation("Beautiful Logging");
        }
    }
}
