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

/// <summary>The Settings "Particle rendering" section: the volume detail picker (16, 8, 4, 2, 1)
/// and the temporal reuse tolerance picker (Off, 1, 2, 4, 8, 16 px).</summary>
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
    Assert.Equal(4, picker.SelectedIndex);
    Assert.Equal(1, vm.ParticleRendering.VolumeDetail);

    picker.SelectedIndex = 2;
    Dispatcher.UIThread.RunJobs();
    Assert.Equal(4, vm.ParticleRendering.VolumeDetail);

    vm.ParticleRendering.VolumeDetail = 16;
    Dispatcher.UIThread.RunJobs();
    Assert.Equal(0, picker.SelectedIndex);
    window.Close();
  }

  [AvaloniaFact]
  public void HazeDetailPicker_ShowsTheFiveSteps_AndFollowsTheViewModel()
  {
    var vm = new DesignSettingsTabViewModel();
    var view = new SettingsTabView { DataContext = vm };
    var window = new Window { Content = view, Width = 600, Height = 1600 };
    window.Show();
    view.GetLogicalDescendants().OfType<Expander>().Single(e => e.Name == "ParticleRenderingSection").IsExpanded = true;
    Dispatcher.UIThread.RunJobs();

    var picker = view.GetLogicalDescendants().OfType<ComboBox>().Single(c => c.Name == "HazeDetailPicker");
    Assert.Equal(5, picker.ItemCount);
    Assert.Equal(4, picker.SelectedIndex);
    Assert.Equal(1, vm.ParticleRendering.HazeDetail);

    picker.SelectedIndex = 2;
    Dispatcher.UIThread.RunJobs();
    Assert.Equal(4, vm.ParticleRendering.HazeDetail);
    Assert.Equal(1, vm.ParticleRendering.VolumeDetail);

    vm.ParticleRendering.HazeDetail = 16;
    Dispatcher.UIThread.RunJobs();
    Assert.Equal(0, picker.SelectedIndex);
    window.Close();
  }

  [AvaloniaFact]
  public void PropagateDetailPicker_ShowsTheThreeSteps_AndFollowsTheViewModel()
  {
    var vm = new DesignSettingsTabViewModel();
    var view = new SettingsTabView { DataContext = vm };
    var window = new Window { Content = view, Width = 600, Height = 1600 };
    window.Show();
    var section = view.GetLogicalDescendants().OfType<Expander>().Single(e => e.Name == "ParticleRenderingSection");
    section.IsExpanded = true;
    Dispatcher.UIThread.RunJobs();

    var picker = section.GetLogicalDescendants().OfType<ComboBox>().Single(c => c.Name == "PropagateDetailPicker");
    Assert.Equal(3, picker.ItemCount);
    Assert.Equal(2, picker.SelectedIndex);
    Assert.Equal(5, vm.ParticleRendering.PropagateDetail);

    picker.SelectedIndex = 1;
    Dispatcher.UIThread.RunJobs();
    Assert.Equal(9, vm.ParticleRendering.PropagateDetail);
    Assert.Equal(1, vm.ParticleRendering.VolumeDetail);

    vm.ParticleRendering.PropagateDetail = 17;
    Dispatcher.UIThread.RunJobs();
    Assert.Equal(0, picker.SelectedIndex);
    window.Close();
  }

  [AvaloniaFact]
  public void ReuseTolerancePicker_ShowsTheSixSteps_AndFollowsTheViewModel()
  {
    var vm = new DesignSettingsTabViewModel();
    var view = new SettingsTabView { DataContext = vm };
    var window = new Window { Content = view, Width = 600, Height = 1600 };
    window.Show();
    var section = view.GetLogicalDescendants().OfType<Expander>().Single(e => e.Name == "ParticleRenderingSection");
    section.IsExpanded = true;
    Dispatcher.UIThread.RunJobs();

    var picker = section.GetLogicalDescendants().OfType<ComboBox>().Single(c => c.Name == "ReuseTolerancePicker");
    Assert.Equal(6, picker.ItemCount);
    Assert.Equal(5, picker.SelectedIndex); // the default, 16 px
    Assert.Equal("Off", picker.Items[0]);
    Assert.Equal("16 px", picker.Items[5]);
    Assert.Equal(16, vm.ParticleRendering.ReuseTolerance);

    picker.SelectedIndex = 3;
    Dispatcher.UIThread.RunJobs();
    Assert.Equal(4, vm.ParticleRendering.ReuseTolerance);
    Assert.Equal(1, vm.ParticleRendering.VolumeDetail);

    vm.ParticleRendering.ReuseTolerance = 8;
    Dispatcher.UIThread.RunJobs();
    Assert.Equal(4, picker.SelectedIndex);

    // a culture change relabels the off item and keeps the selection
    vm.ParticleRendering.ReuseToleranceOffLabel = "Disattivato";
    Dispatcher.UIThread.RunJobs();
    Assert.Equal("Disattivato", picker.Items[0]);
    Assert.Equal(4, picker.SelectedIndex);
    Assert.Equal(8, vm.ParticleRendering.ReuseTolerance);
    window.Close();
  }

  /// <summary>The quality slider: five snapped levels, starting at the lowest; moving it moves the
  /// pickers, and a hand-edited picker reads "Custom".</summary>
  [AvaloniaFact]
  public void QualitySlider_StartsLowest_DrivesThePickers_AndShowsCustom()
  {
    var vm = new DesignSettingsTabViewModel();
    var view = new SettingsTabView { DataContext = vm };
    var window = new Window { Content = view, Width = 600, Height = 1600 };
    window.Show();
    var section = view.GetLogicalDescendants().OfType<Expander>().Single(e => e.Name == "ParticleRenderingSection");
    section.IsExpanded = true;
    Dispatcher.UIThread.RunJobs();

    var slider = section.GetLogicalDescendants().OfType<Slider>().Single(c => c.Name == "ParticleQualitySlider");
    var label = section.GetLogicalDescendants().OfType<TextBlock>().Single(c => c.Name == "ParticleQualityLabel");
    var volume = section.GetLogicalDescendants().OfType<ComboBox>().Single(c => c.Name == "VolumeDetailPicker");
    Assert.Equal(1, slider.Minimum);
    Assert.Equal(5, slider.Maximum);
    Assert.True(slider.IsSnapToTickEnabled);
    Assert.Equal(1, slider.Value);
    Assert.Equal("Lowest", label.Text);

    slider.Value = 5;
    Dispatcher.UIThread.RunJobs();
    Assert.Equal("Ultra", label.Text);
    Assert.Equal(0, volume.SelectedIndex); // 16

    volume.SelectedIndex = 1; // 8: no preset has 8 with the ultra rest
    Dispatcher.UIThread.RunJobs();
    Assert.Equal("Custom", label.Text);
    Assert.Equal(5, slider.Value);
    window.Close();
  }
}
#endif
