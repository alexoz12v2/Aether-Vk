#if DEBUG
using System;
using System.Linq;
using AetherVk.Logic.Models;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using AetherVk.UI;
using AetherVk.Views;
using Avalonia;
using Avalonia.Automation;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Input;
using Avalonia.LogicalTree;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace AetherVk.AppTests;

/// <summary>Desktop billboards in the headless UI (plan B-8, B-9).</summary>
public class DesktopBillboardViewTests
{
  /// B-8: the Settings tab has the Billboards section; Create adds an editable row whose edits
  /// reach the service.
  [AvaloniaFact]
  public void SettingsSection_CreatesAndEditsBillboards()
  {
    var vm = new DesignSettingsTabViewModel();
    var view = new SettingsTabView { DataContext = vm };
    var window = new Window { Content = view, Width = 520, Height = 1600 };
    window.Show();
    var section = view.GetLogicalDescendants().OfType<Expander>().Single(e => Equals(e.Header, vm.StrBillboards));
    section.IsExpanded = true;
    Dispatcher.UIThread.RunJobs();

    var picker = view.FindControl<ComboBox>("BillboardTexturePicker")!;
    Assert.Equal(1, picker.ItemCount);
    var create = view.FindControl<Button>("CreateBillboardButton")!;
    Assert.Same(vm.Billboards.CreateCommand, create.Command);
    create.Command!.Execute(null);
    Dispatcher.UIThread.RunJobs();

    var list = view.FindControl<ItemsControl>("BillboardList")!;
    Assert.Equal(1, list.ItemCount);
    var row = vm.Billboards.Items.Single();
    Assert.Contains(view.GetVisualDescendants().OfType<TextBlock>(), t => t.Text == row.DisplayLabel);

    // edit through the controls of the row
    var numbers = list.GetVisualDescendants().OfType<NumericUpDown>().ToList();
    Assert.Equal(4, numbers.Count); // x, y, w, h
    numbers[0].Value = 333;
    var slider = list.GetVisualDescendants().OfType<Slider>().Single();
    slider.Value = 0.3;
    Dispatcher.UIThread.RunJobs();
    var b = vm.Billboards.Service.Current.Single();
    Assert.Equal(333, b.CenterX, 6);
    Assert.Equal(0.3, b.Alpha, 6);

    var delete = list.GetVisualDescendants().OfType<Button>().Single(x => Equals(x.Content, vm.StrBillboardDelete));
    delete.Command!.Execute(null);
    Dispatcher.UIThread.RunJobs();
    Assert.Equal(0, list.ItemCount);
    Assert.Empty(vm.Billboards.Service.Current);
    window.Close();
  }

  /// A FITS billboard row has the "FITS image" expander: the key facts, the header cards in a
  /// DataGrid (one row per card), the stretch controls and the purpose buttons.
  [AvaloniaFact]
  public void FitsRow_ShowsTheFactsAndTheHeaderGrid()
  {
    var vm = new DesignSettingsTabViewModel();
    var view = new SettingsTabView { DataContext = vm };
    var window = new Window { Content = view, Width = 520, Height = 2400 };
    window.Show();
    var section = view.GetLogicalDescendants().OfType<Expander>().Single(e => Equals(e.Header, vm.StrBillboards));
    section.IsExpanded = true;
    vm.Billboards.CreateCommand.Execute(null);
    var row = vm.Billboards.Items.Single();
    var meta = Logic.Models.FitsMetadata.Parse(DesignFits.SampleJson)!;
    row.ShowFits(meta, 5);
    Dispatcher.UIThread.RunJobs();
    window.UpdateLayout();
    Dispatcher.UIThread.RunJobs();

    var fitsSection = view.GetVisualDescendants().OfType<Expander>()
      .Single(e => AutomationProperties.GetAutomationId(e) == "BillboardFitsSection");
    fitsSection.IsExpanded = true;
    Dispatcher.UIThread.RunJobs();
    window.UpdateLayout();
    Dispatcher.UIThread.RunJobs();

    Assert.Contains(fitsSection.GetVisualDescendants().OfType<SelectableTextBlock>(), t => t.Text == "Liverpool Telescope / IO:O");
    var grid = fitsSection.GetVisualDescendants().OfType<DataGrid>().Single();
    var rows = grid.GetVisualDescendants().OfType<DataGridRow>().ToList();
    Assert.Equal(meta.Cards.Count, rows.Count);
    Assert.Contains(grid.GetVisualDescendants().OfType<TextBlock>(), t => t.Text == "OBJECT");
    Assert.Contains(grid.GetVisualDescendants().OfType<TextBlock>(), t => t.Text == "[degrees West] Telescope longitude");
    Assert.InRange(grid.Bounds.Height, 239, 241); // its own scroll
    var buttons = fitsSection.GetVisualDescendants().OfType<Button>().ToList();
    Assert.Contains(buttons, b => Equals(b.Content, "Apply stretch"));
    Assert.Contains(buttons, b => Equals(b.Content, "Set up comparison"));
    // the plate scale came from the header
    Assert.Equal(0.30368, vm.Billboards.Service.Current.Single().ImageArcsecPerPx);
    window.Close();
  }

  private static (DesktopBillboardWindow Window, Func<DesktopBillboard?> Last) ShowBillboard(DesktopBillboard b)
  {
    DesktopBillboard? last = null;
    var w = new DesktopBillboardWindow(b, e => last = e);
    var bmp = new WriteableBitmap(new PixelSize(4, 2), new Vector(96, 96), PixelFormat.Rgba8888, AlphaFormat.Unpremul);
    using (var fb = bmp.Lock())
    {
      var red = new byte[] { 255, 0, 0, 255 };
      for (int y = 0; y < 2; y++)
        for (int x = 0; x < 4; x++)
          System.Runtime.InteropServices.Marshal.Copy(red, 0, fb.Address + y * fb.RowBytes + x * 4, 4);
    }
    w.SetBitmap(bmp);
    w.Apply(b);
    w.Show();
    w.Apply(b);
    Dispatcher.UIThread.RunJobs();
    return (w, () => last);
  }

  private static Point Local(DesktopBillboardWindow w, double dx, double dy)
  {
    // window-local DIP of the image centre offset by (dx, dy) desktop px (scaling 1 headless)
    var r = DesktopBillboardGeometry.WindowRect(w.Billboard);
    return new Point(w.Billboard.CenterX - r.X + dx, w.Billboard.CenterY - r.Y + dy);
  }

  /// B-9: left drag moves, right click cycles the gizmo, clicks on the transparent margin are
  /// ignored.
  [AvaloniaFact]
  public void Window_PointerRules()
  {
    var b = new DesktopBillboard(Guid.NewGuid(), "k", 400, 300, 200, 100, 0.5);
    var (w, last) = ShowBillboard(b);

    var p0 = Local(w, -30, 10);
    w.MouseDown(p0, MouseButton.Left);
    w.MouseMove(p0 + new Point(25, -15));
    // the window followed the pointer: the same desktop point is back at p0 in window coordinates
    w.MouseUp(Local(w, -30, 10), MouseButton.Left);
    Dispatcher.UIThread.RunJobs();
    Assert.NotNull(last());
    Assert.Equal(425, last()!.CenterX, 6);
    Assert.Equal(285, last()!.CenterY, 6);

    var p1 = Local(w, -30, 10);
    w.MouseDown(p1, MouseButton.Right);
    w.MouseUp(p1, MouseButton.Right);
    Dispatcher.UIThread.RunJobs();
    Assert.Equal(BillboardGizmoMode.Translate, last()!.GizmoMode);
    Assert.Equal(BillboardGizmoMode.Translate, w.Billboard.GizmoMode);

    // the margin (window corner) takes no presses
    var before = last();
    w.MouseDown(new Point(2, 2), MouseButton.Left);
    w.MouseMove(new Point(40, 40));
    w.MouseUp(new Point(40, 40), MouseButton.Left);
    Dispatcher.UIThread.RunJobs();
    Assert.Same(before, last());
    w.Close();
  }

  /// B-9: the margin is fully transparent, the image is drawn at the billboard's opacity.
  [AvaloniaFact]
  public void Window_RendersTheImageAtItsOpacity_OverATransparentMargin()
  {
    var b = new DesktopBillboard(Guid.NewGuid(), "k", 400, 300, 200, 100, 0.5);
    var (w, _) = ShowBillboard(b);
    var frame = w.CaptureRenderedFrame();
    Assert.NotNull(frame);
    using var fb = ((WriteableBitmap)frame!).Lock();
    byte[] Px(int x, int y)
    {
      var px = new byte[4];
      System.Runtime.InteropServices.Marshal.Copy(fb.Address + y * fb.RowBytes + x * 4, px, 0, 4);
      return px;
    }
    var centre = Local(w, 0, 0);
    var c = Px((int)centre.X, (int)centre.Y);
    var corner = Px(1, 1);
    Assert.Equal(0, corner[3]);
    Assert.InRange(c[3], 120, 136); // 0.5 · 255
    w.Close();
  }
}

/// <summary>The application's DI container wires the billboard pieces (the app's real host).</summary>
public class DesktopBillboardWiringTests
{
  [AvaloniaFact]
  public void Host_ResolvesTheBillboardServicesAndTheSettingsSection()
  {
    var host = App.Host ?? throw new InvalidOperationException("DI host not initialised");
    var svc = host.Services.GetRequiredService<Logic.Services.DesktopBillboardService>();
    Assert.NotNull(host.Services.GetRequiredService<DesktopBillboardWindowManager>());
    using var scope = host.Services.CreateScope();
    var settings = scope.ServiceProvider.GetRequiredService<SettingsTabViewModel>();
    // the section edits the application's single billboard list
    Assert.Same(svc, settings.Billboards.Service);
  }
}
#endif
