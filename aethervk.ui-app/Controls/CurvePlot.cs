using System;
using System.Globalization;
using AetherVk.Logic.Utils;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;

namespace AetherVk.Controls;

/// <summary>
/// A small read-only line plot of a <see cref="CurvePlotModel"/> (series, log or linear axes,
/// ticks, an optional vertical marker). The math lives in the logic layer
/// (<see cref="DustCurves"/>); this only maps data to pixels. Colours come from brush properties
/// set to theme tokens in AXAML. On a log y axis a value ≤ 0 is drawn on the bottom edge.
/// </summary>
public sealed class CurvePlot : Control
{
  public static readonly StyledProperty<CurvePlotModel?> ModelProperty =
    AvaloniaProperty.Register<CurvePlot, CurvePlotModel?>(nameof(Model));

  public static readonly StyledProperty<IBrush?> PrimaryBrushProperty =
    AvaloniaProperty.Register<CurvePlot, IBrush?>(nameof(PrimaryBrush), Brushes.CornflowerBlue);

  public static readonly StyledProperty<IBrush?> SecondaryBrushProperty =
    AvaloniaProperty.Register<CurvePlot, IBrush?>(nameof(SecondaryBrush), Brushes.Orange);

  public static readonly StyledProperty<IBrush?> MarkerBrushProperty =
    AvaloniaProperty.Register<CurvePlot, IBrush?>(nameof(MarkerBrush), Brushes.Goldenrod);

  public static readonly StyledProperty<IBrush?> AxisBrushProperty =
    AvaloniaProperty.Register<CurvePlot, IBrush?>(nameof(AxisBrush), Brushes.Gray);

  public static readonly StyledProperty<IBrush?> TextBrushProperty =
    AvaloniaProperty.Register<CurvePlot, IBrush?>(nameof(TextBrush), Brushes.Gray);

  public static readonly StyledProperty<IBrush?> PlotBackgroundProperty =
    AvaloniaProperty.Register<CurvePlot, IBrush?>(nameof(PlotBackground));

  static CurvePlot()
  {
    AffectsRender<CurvePlot>(ModelProperty, PrimaryBrushProperty, SecondaryBrushProperty,
      MarkerBrushProperty, AxisBrushProperty, TextBrushProperty, PlotBackgroundProperty);
    HeightProperty.OverrideDefaultValue<CurvePlot>(140);
  }

  public CurvePlotModel? Model { get => GetValue(ModelProperty); set => SetValue(ModelProperty, value); }
  public IBrush? PrimaryBrush { get => GetValue(PrimaryBrushProperty); set => SetValue(PrimaryBrushProperty, value); }
  public IBrush? SecondaryBrush { get => GetValue(SecondaryBrushProperty); set => SetValue(SecondaryBrushProperty, value); }
  public IBrush? MarkerBrush { get => GetValue(MarkerBrushProperty); set => SetValue(MarkerBrushProperty, value); }
  public IBrush? AxisBrush { get => GetValue(AxisBrushProperty); set => SetValue(AxisBrushProperty, value); }
  public IBrush? TextBrush { get => GetValue(TextBrushProperty); set => SetValue(TextBrushProperty, value); }
  public IBrush? PlotBackground { get => GetValue(PlotBackgroundProperty); set => SetValue(PlotBackgroundProperty, value); }

  private const double FontSize = 9.5;
  private const double MarginLeft = 38, MarginRight = 8, MarginTop = 14, MarginBottom = 26;

  private FormattedText Text(string s, IBrush? brush) =>
    new(s, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, Typeface.Default, FontSize, brush ?? Brushes.Gray);

  private static double Map(double v, double min, double max, bool log)
  {
    if (log)
    {
      if (!(v > 0)) return double.NegativeInfinity;
      return (Math.Log10(v) - Math.Log10(min)) / (Math.Log10(max) - Math.Log10(min));
    }
    return (v - min) / (max - min);
  }

  public override void Render(DrawingContext ctx)
  {
    base.Render(ctx);
    var m = Model;
    var b = Bounds;
    if (m is null || b.Width < MarginLeft + MarginRight + 10 || b.Height < MarginTop + MarginBottom + 10)
      return;
    if (!(m.XMax > m.XMin) || !(m.YMax > m.YMin) || (m.XLog && !(m.XMin > 0)) || (m.YLog && !(m.YMin > 0)))
      return;

    var plot = new Rect(MarginLeft, MarginTop, b.Width - MarginLeft - MarginRight, b.Height - MarginTop - MarginBottom);
    if (PlotBackground is { } bg)
      ctx.FillRectangle(bg, plot);

    double Px(double x) => plot.Left + Map(x, m.XMin, m.XMax, m.XLog) * plot.Width;
    double Py(double y)
    {
      double t = Map(y, m.YMin, m.YMax, m.YLog);
      if (double.IsNegativeInfinity(t)) t = 0; // ≤ 0 on a log axis: the bottom edge
      return plot.Bottom - t * plot.Height;
    }

    var axisPen = new Pen(AxisBrush, 1);
    var gridPen = new Pen(AxisBrush, 0.5, new DashStyle(new double[] { 2, 3 }, 0));

    // grid + tick labels
    foreach (var t in m.XTicks)
    {
      double x = Px(t.Value);
      if (x < plot.Left - 0.5 || x > plot.Right + 0.5) continue;
      ctx.DrawLine(gridPen, new Point(x, plot.Top), new Point(x, plot.Bottom));
      var ft = Text(t.Label, TextBrush);
      double tx = Math.Max(0, Math.Min(b.Width - ft.Width, x - ft.Width / 2));
      ctx.DrawText(ft, new Point(tx, plot.Bottom + 2));
    }
    foreach (var t in m.YTicks)
    {
      double y = Py(t.Value);
      if (y < plot.Top - 0.5 || y > plot.Bottom + 0.5) continue;
      ctx.DrawLine(gridPen, new Point(plot.Left, y), new Point(plot.Right, y));
      var ft = Text(t.Label, TextBrush);
      ctx.DrawText(ft, new Point(Math.Max(0, plot.Left - 3 - ft.Width), y - ft.Height / 2));
    }
    ctx.DrawRectangle(null, axisPen, plot);

    // axis labels: x under the tick labels (right aligned), y above the plot (left)
    var xl = Text(m.XLabel, TextBrush);
    ctx.DrawText(xl, new Point(plot.Right - xl.Width, b.Height - xl.Height));
    var yl = Text(m.YLabel, TextBrush);
    ctx.DrawText(yl, new Point(0, 0));

    using (ctx.PushClip(plot.Inflate(1)))
    {
      foreach (var s in m.Series)
      {
        if (s.Points.Count < 2) continue;
        var pen = s.StyleIndex == 0
          ? new Pen(PrimaryBrush, 1.6)
          : new Pen(SecondaryBrush, 1.6, new DashStyle(new double[] { 4, 2 }, 0));
        var g = new StreamGeometry();
        using (var c = g.Open())
        {
          bool started = false;
          foreach (var p in s.Points)
          {
            if (double.IsNaN(p.X) || double.IsNaN(p.Y)) continue;
            var pt = new Point(Px(p.X), Py(p.Y));
            if (!started) { c.BeginFigure(pt, false); started = true; }
            else c.LineTo(pt);
          }
          if (started) c.EndFigure(false);
        }
        ctx.DrawGeometry(null, pen, g);
      }

      if (!double.IsNaN(m.MarkerX))
      {
        double x = Px(m.MarkerX);
        var mp = new Pen(MarkerBrush, 1.2, new DashStyle(new double[] { 3, 2 }, 0));
        ctx.DrawLine(mp, new Point(x, plot.Top), new Point(x, plot.Bottom));
        if (!string.IsNullOrEmpty(m.MarkerLabel))
        {
          var ft = Text(m.MarkerLabel, MarkerBrush);
          double tx = x + 3 + ft.Width > plot.Right ? x - 3 - ft.Width : x + 3;
          ctx.DrawText(ft, new Point(tx, plot.Bottom - ft.Height - 2));
        }
      }
    }

    // legend (top right, inside the plot) when there is more than one series
    if (m.Series.Count > 1)
    {
      double y = plot.Top + 2;
      foreach (var s in m.Series)
      {
        var ft = Text(s.Label, s.StyleIndex == 0 ? PrimaryBrush : SecondaryBrush);
        double x = plot.Right - ft.Width - 4;
        var pen = s.StyleIndex == 0
          ? new Pen(PrimaryBrush, 1.6)
          : new Pen(SecondaryBrush, 1.6, new DashStyle(new double[] { 4, 2 }, 0));
        ctx.DrawLine(pen, new Point(x - 16, y + ft.Height / 2), new Point(x - 3, y + ft.Height / 2));
        ctx.DrawText(ft, new Point(x, y));
        y += ft.Height + 1;
      }
    }
  }
}
