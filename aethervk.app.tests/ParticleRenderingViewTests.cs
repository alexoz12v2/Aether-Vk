#if DEBUG
using System.Linq;
using AetherVk.Logic.ViewModels;
using AetherVk.Views;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.LogicalTree;
using Avalonia.Threading;
using Xunit;

namespace AetherVk.AppTests;

/// <summary>The Settings "Particle rendering" section: the volume detail picker (16, 8, 4, 2, 1).</summary>
public class ParticleRenderingViewTests
{
  [AvaloniaFact]
  public void VolumeDetailPicker_ShowsTheFiveSteps_AndFollowsTheViewModel()
  {
    var vm = new DesignSettingsTabViewModel();
    var view = new SettingsTabView { DataContext = vm };
    var window = new Window { Content = view, Width = 600, Height = 1600 };
    window.Show();
    view.GetLogicalDescendants().OfType<Expander>().Single(e => e.Name == "ParticleRenderingSection").IsExpanded = true;
    Dispatcher.UIThread.RunJobs();

    var picker = view.GetLogicalDescendants().OfType<ComboBox>().Single(c => c.Name == "VolumeDetailPicker");
    Assert.Equal(5, picker.ItemCount);
    Assert.Equal(0, picker.SelectedIndex);
    Assert.Equal(16, vm.ParticleRendering.VolumeDetail);

    picker.SelectedIndex = 2;
    Dispatcher.UIThread.RunJobs();
    Assert.Equal(4, vm.ParticleRendering.VolumeDetail);

    vm.ParticleRendering.VolumeDetail = 1;
    Dispatcher.UIThread.RunJobs();
    Assert.Equal(4, picker.SelectedIndex);
    window.Close();
  }
}
#endif
