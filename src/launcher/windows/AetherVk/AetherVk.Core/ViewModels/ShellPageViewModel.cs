
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

// reference: https://www.youtube.com/watch?v=83UVWrfYreU

namespace AetherVk.Core.ViewModels
{

    // TODO Internal strongly named assemblies with friend assembly declarations

    public partial class ShellPageViewModel : ObservableObject
    {
        [ObservableProperty]
        public partial int PanelCount { get; set; }

    }
}
