#if DEBUG
using System.Linq;
using AetherVk.Controls;
using AetherVk.Converters;
using AetherVk.Logic.Models;
using AetherVk.Logic.ViewModels;
using AetherVk.Views;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.LogicalTree;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Xunit;

namespace AetherVk.AppTests;

public class ImportsTabViewTests
{
  private static (Window Window, ImportsTabView View) Show(double width)
  {
    var view = new ImportsTabView { DataContext = new DesignImportsTabViewModel() };
    var window = new Window { Content = view, Width = width, Height = 600 };
    window.Show();
    Dispatcher.UIThread.RunJobs();
    return (window, view);
  }

  [AvaloniaFact]
  public void Shows_meshes_and_textures_in_separate_sections()
  {
    var (window, view) = Show(900);
    var mesh = view.FindControl<Border>("MeshSection")!;
    var tex = view.FindControl<Border>("TextureSection")!;
    var meshLabels = mesh.GetLogicalDescendants().OfType<TextBlock>().Select(t => t.Text).ToList();
    var texLabels = tex.GetLogicalDescendants().OfType<TextBlock>().Select(t => t.Text).ToList();
    Assert.Contains("67P_churyumov", meshLabels);
    Assert.Contains("bennu", meshLabels);
    Assert.DoesNotContain("regolith_albedo", meshLabels);
    Assert.Contains("regolith_albedo", texLabels);
    Assert.Contains("regolith_normal", texLabels);
    window.Close();
  }

  [AvaloniaFact]
  public void Sections_are_side_by_side_when_wide_and_stacked_when_narrow()
  {
    var (wide, wideView) = Show(1000);
    Assert.True(wideView.IsSideBySide);
    Assert.Equal(2, Grid.GetColumn(wideView.FindControl<Border>("TextureSection")!));
    wide.Close();

    var (narrow, narrowView) = Show(ImportsTabView.StackedLayoutThreshold - 100);
    Assert.False(narrowView.IsSideBySide);
    var tex = narrowView.FindControl<Border>("TextureSection")!;
    Assert.Equal(0, Grid.GetColumn(tex));
    Assert.Equal(2, Grid.GetRow(tex));
    narrow.Close();
  }
}

public class ImportsTabUnloadButtonTests
{
  [AvaloniaFact]
  public void Every_card_has_an_unload_button_bound_to_the_command()
  {
    var vm = new DesignImportsTabViewModel();
    var view = new ImportsTabView { DataContext = vm };
    var window = new Window { Content = view, Width = 900, Height = 600 };
    window.Show();
    Dispatcher.UIThread.RunJobs();
    var buttons = view.GetVisualDescendants().OfType<Button>().Where(b => b.Classes.Contains("asset-unload")).ToList();
    Assert.Equal(vm.MeshAssets.Count + vm.TextureAssets.Count, buttons.Count);
    Assert.All(buttons, b => Assert.Same(vm.RemoveAssetCommand, b.Command));
    Assert.Contains(buttons, b => ReferenceEquals(b.CommandParameter, vm.MeshAssets[0]));
    window.Close();
  }
}

public class ThumbnailConverterTests
{
  [AvaloniaFact]
  public void Converts_rgba_thumbnail_to_bitmap_and_caches_it()
  {
    var thumb = new AssetThumbnail(3, 2, new byte[3 * 2 * 4]);
    var c = ThumbnailToBitmapConverter.Instance;
    var bmp = Assert.IsAssignableFrom<Bitmap>(c.Convert(thumb, typeof(object), null, System.Globalization.CultureInfo.InvariantCulture));
    Assert.Equal(3, bmp.PixelSize.Width);
    Assert.Equal(2, bmp.PixelSize.Height);
    Assert.Same(bmp, c.Convert(thumb, typeof(object), null, System.Globalization.CultureInfo.InvariantCulture));
  }

  [AvaloniaFact]
  public void Rejects_missing_or_truncated_thumbnails()
  {
    var c = ThumbnailToBitmapConverter.Instance;
    var ci = System.Globalization.CultureInfo.InvariantCulture;
    Assert.Null(c.Convert(null, typeof(object), null, ci));
    Assert.Null(c.Convert(new AssetThumbnail(4, 4, new byte[8]), typeof(object), null, ci));
  }
}

public class CometAppearanceViewTests
{
  [AvaloniaFact]
  public void Custom_mesh_section_follows_display_mode_and_wiring_locks_while_running()
  {
    var vm = new DesignSettingsTabViewModel();
    var view = new SettingsTabView { DataContext = vm };
    var window = new Window { Content = view, Width = 500, Height = 1200 };
    window.Show();
    foreach (var e in view.GetVisualDescendants().OfType<Expander>()) e.IsExpanded = true;
    Dispatcher.UIThread.RunJobs();

    var appearance = vm.CometAppearance;
    var customExpander = view.GetLogicalDescendants().OfType<Expander>()
      .Single(e => Equals(e.Header, vm.StrCustomMesh));
    Assert.True(customExpander.IsVisible);

    var meshCombo = view.GetLogicalDescendants().OfType<ComboBox>()
      .Single(c => ReferenceEquals(c.ItemsSource, appearance.MeshOptions));
    var yawSlider = view.GetLogicalDescendants().OfType<UnboundedSlider>().First();
    Assert.True(meshCombo.IsEffectivelyEnabled);

    // playing: mesh / texture wiring disabled, placement still editable
    appearance.IsSimulationRunning = true;
    Dispatcher.UIThread.RunJobs();
    Assert.False(meshCombo.IsEffectivelyEnabled);
    Assert.True(yawSlider.IsEffectivelyEnabled);

    appearance.IsSimulationRunning = false;
    appearance.IsCustomMode = false;
    Dispatcher.UIThread.RunJobs();
    Assert.False(customExpander.IsVisible);
    window.Close();
  }

  [AvaloniaFact]
  public void Custom_radio_is_disabled_without_imported_meshes()
  {
    var vm = new DesignSettingsTabViewModel();
    var view = new SettingsTabView { DataContext = vm };
    var window = new Window { Content = view, Width = 500, Height = 1200 };
    window.Show();
    foreach (var e in view.GetVisualDescendants().OfType<Expander>()) e.IsExpanded = true;
    Dispatcher.UIThread.RunJobs();

    var customRadio = view.GetLogicalDescendants().OfType<RadioButton>()
      .Single(r => Equals(r.Content, vm.StrDisplayModeCustom));
    Assert.True(customRadio.IsEffectivelyEnabled);

    vm.CometAppearance.IsCustomMode = false;
    vm.CometAppearance.HasMeshes = false;
    Dispatcher.UIThread.RunJobs();
    Assert.False(customRadio.IsEffectivelyEnabled);
    window.Close();
  }
}
#endif
