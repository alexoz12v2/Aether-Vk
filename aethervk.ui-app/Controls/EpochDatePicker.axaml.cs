using System;
using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Layout;
using Avalonia.Media;

namespace AetherVk.Controls;

public partial class EpochDatePicker : UserControl
{
  public static readonly StyledProperty<DateTimeOffset> SelectedEpochProperty = AvaloniaProperty.Register<
    EpochDatePicker,
    DateTimeOffset
  >(nameof(SelectedEpoch), DateTimeOffset.UtcNow, defaultBindingMode: Avalonia.Data.BindingMode.TwoWay);

  public DateTimeOffset SelectedEpoch
  {
    get => GetValue(SelectedEpochProperty);
    set => SetValue(SelectedEpochProperty, value);
  }

  public static readonly StyledProperty<string> LabelProperty = AvaloniaProperty.Register<
    EpochDatePicker,
    string
  >(nameof(Label), "Epoch");

  public string Label
  {
    get => GetValue(LabelProperty);
    set => SetValue(LabelProperty, value);
  }

  public static readonly StyledProperty<bool> IsValidProperty = AvaloniaProperty.Register<
    EpochDatePicker,
    bool
  >(nameof(IsValid), true);

  public bool IsValid
  {
    get => GetValue(IsValidProperty);
    set => SetValue(IsValidProperty, value);
  }

  private int _displayMonth;
  private int _displayYear;
  private bool _suppressSync;

  public EpochDatePicker()
  {
    InitializeComponent();

    var now = DateTimeOffset.UtcNow;
    _displayMonth = now.Month;
    _displayYear = now.Year;
  }

  protected override void OnLoaded(RoutedEventArgs e)
  {
    base.OnLoaded(e);
    SyncTextFromEpoch();
    RegenerateDays();
    UpdateMonthYearLabels();
    SyncTimeFields();
  }

  static EpochDatePicker()
  {
    SelectedEpochProperty.Changed.AddClassHandler<EpochDatePicker>((picker, _) =>
    {
      if (!picker._suppressSync)
      {
        picker.SyncTextFromEpoch();
        picker.SyncTimeFields();
        picker.RegenerateDays();
      }
    });
  }

  // ────────────────── Epoch text commit ──────────────────

  private void OnEpochTextCommit()
  {
    var text = EpochTextBox.Text;
    if (string.IsNullOrWhiteSpace(text))
    {
      IsValid = false;
      ValidationBlock.IsVisible = true;
      return;
    }

    if (DateTimeOffset.TryParse(text, CultureInfo.InvariantCulture,
          DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var dt))
    {
      _suppressSync = true;
      SelectedEpoch = dt;
      IsValid = true;
      ValidationBlock.IsVisible = false;
      _displayMonth = dt.Month;
      _displayYear = dt.Year;
      UpdateMonthYearLabels();
      SyncTimeFields();
      RegenerateDays();
      _suppressSync = false;
    }
    else
    {
      IsValid = false;
      ValidationBlock.IsVisible = true;
    }
  }

  private void OnEpochTextKeyDown(object? sender, KeyEventArgs e)
  {
    if (e.Key == Key.Enter)
    {
      OnEpochTextCommit();
      TopLevel.GetTopLevel(this)?.FocusManager?.ClearFocus();
    }
    else if (e.Key == Key.Escape)
    {
      SyncTextFromEpoch();
      TopLevel.GetTopLevel(this)?.FocusManager?.ClearFocus();
    }
  }

  private void OnEpochTextLostFocus(object? sender, RoutedEventArgs e)
  {
    OnEpochTextCommit();
  }

  private void SyncTextFromEpoch()
  {
    EpochTextBox.Text = SelectedEpoch.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fff", CultureInfo.InvariantCulture) + " UTC";
  }

  // ────────────────── Calendar button ──────────────────

  private void OnCalendarButtonClick(object? sender, RoutedEventArgs e)
  {
    _displayMonth = SelectedEpoch.Month;
    _displayYear = SelectedEpoch.Year;
    UpdateMonthYearLabels();
    SyncTimeFields();
    RegenerateDays();
  }

  // ────────────────── Month / Year navigation ──────────────────

  private void OnPrevMonthClick(object? sender, RoutedEventArgs e)
  {
    _displayMonth--;
    if (_displayMonth < 1)
    {
      _displayMonth = 12;
      _displayYear--;
    }
    UpdateMonthYearLabels();
    RegenerateDays();
  }

  private void OnNextMonthClick(object? sender, RoutedEventArgs e)
  {
    _displayMonth++;
    if (_displayMonth > 12)
    {
      _displayMonth = 1;
      _displayYear++;
    }
    UpdateMonthYearLabels();
    RegenerateDays();
  }

  private void UpdateMonthYearLabels()
  {
    MonthLabel.Text = new DateTimeOffset(_displayYear, _displayMonth, 1, 0, 0, 0, TimeSpan.Zero)
      .ToString("MMMM", CultureInfo.InvariantCulture);
    YearLabel.Text = _displayYear.ToString();
  }

  // ────────────────── Month / Year inline edit ──────────────────

  private void OnMonthLabelClick(object? sender, PointerPressedEventArgs e)
  {
    MonthLabel.IsVisible = false;
    MonthEditor.IsVisible = true;
    MonthEditor.Text = MonthLabel.Text;
    MonthEditor.Focus();
    MonthEditor.SelectAll();
  }

  private void CommitMonthEdit()
  {
    if (DateTime.TryParseExact(MonthEditor.Text, "MMMM", CultureInfo.InvariantCulture,
          DateTimeStyles.None, out var parsed))
    {
      _displayMonth = parsed.Month;
    }
    else if (int.TryParse(MonthEditor.Text, out int m) && m >= 1 && m <= 12)
    {
      _displayMonth = m;
    }
    MonthEditor.IsVisible = false;
    MonthLabel.IsVisible = true;
    UpdateMonthYearLabels();
    RegenerateDays();
  }

  private void OnMonthEditorKeyDown(object? sender, KeyEventArgs e)
  {
    if (e.Key == Key.Enter)
    {
      CommitMonthEdit();
    }
    else if (e.Key == Key.Escape)
    {
      MonthEditor.IsVisible = false;
      MonthLabel.IsVisible = true;
    }
  }

  private void OnMonthEditorLostFocus(object? sender, RoutedEventArgs e)
  {
    if (MonthEditor.IsVisible)
      CommitMonthEdit();
  }

  private void OnYearLabelClick(object? sender, PointerPressedEventArgs e)
  {
    YearLabel.IsVisible = false;
    YearEditor.IsVisible = true;
    YearEditor.Text = _displayYear.ToString();
    YearEditor.Focus();
    YearEditor.SelectAll();
  }

  private void CommitYearEdit()
  {
    if (int.TryParse(YearEditor.Text, out int y) && y >= 1 && y <= 9999)
    {
      _displayYear = y;
    }
    YearEditor.IsVisible = false;
    YearLabel.IsVisible = true;
    UpdateMonthYearLabels();
    RegenerateDays();
  }

  private void OnYearEditorKeyDown(object? sender, KeyEventArgs e)
  {
    if (e.Key == Key.Enter)
    {
      CommitYearEdit();
    }
    else if (e.Key == Key.Escape)
    {
      YearEditor.IsVisible = false;
      YearLabel.IsVisible = true;
    }
  }

  private void OnYearEditorLostFocus(object? sender, RoutedEventArgs e)
  {
    if (YearEditor.IsVisible)
      CommitYearEdit();
  }

  // ────────────────── Day grid generation ──────────────────

  private void RegenerateDays()
  {
    DaysGrid.Children.Clear();

    var firstDay = new DateTimeOffset(_displayYear, _displayMonth, 1, 0, 0, 0, TimeSpan.Zero);
    int startDow = ((int)firstDay.DayOfWeek + 6) % 7; // Monday=0
    int daysInMonth = DateTime.DaysInMonth(_displayYear, _displayMonth);
    int selectedDay = SelectedEpoch.Month == _displayMonth && SelectedEpoch.Year == _displayYear
      ? SelectedEpoch.Day
      : -1;

    for (int cell = 0; cell < 42; cell++)
    {
      int dayNum = cell - startDow + 1;
      if (dayNum >= 1 && dayNum <= daysInMonth)
      {
        var accentBg = this.TryFindResource("Accent.Primary", out var abr) && abr is IBrush ab
          ? ab : Brushes.DodgerBlue;
        var accentFg = this.TryFindResource("Text.On-Accent", out var oar) && oar is IBrush oa
          ? oa : Brushes.White;
        var normalFg = this.TryFindResource("Text.Primary", out var tpr) && tpr is IBrush tp
          ? tp : Brushes.White;

        var btn = new Button
        {
          Content = dayNum.ToString(),
          HorizontalAlignment = HorizontalAlignment.Stretch,
          HorizontalContentAlignment = HorizontalAlignment.Center,
          Padding = new Thickness(0, 4),
          FontSize = 12,
          Tag = dayNum,
          Background = dayNum == selectedDay ? accentBg : Brushes.Transparent,
          Foreground = dayNum == selectedDay ? accentFg : normalFg,
          BorderThickness = new Thickness(0),
          CornerRadius = new CornerRadius(4),
        };
        btn.Click += OnDayButtonClick;
        DaysGrid.Children.Add(btn);
      }
      else
      {
        DaysGrid.Children.Add(new Border()); // empty placeholder
      }
    }
  }

  private void OnDayButtonClick(object? sender, RoutedEventArgs e)
  {
    if (sender is Button btn && btn.Tag is int day)
    {
      var current = SelectedEpoch;
      _suppressSync = true;
      SelectedEpoch = new DateTimeOffset(
        _displayYear, _displayMonth, day,
        current.Hour, current.Minute, current.Second, current.Millisecond,
        TimeSpan.Zero);
      IsValid = true;
      ValidationBlock.IsVisible = false;
      SyncTextFromEpoch();
      RegenerateDays();
      _suppressSync = false;
      CalendarButton.Flyout?.Hide();
    }
  }

  // ────────────────── Time fields ──────────────────

  private void SyncTimeFields()
  {
    var utc = SelectedEpoch.ToUniversalTime();
    HourBox.Text = utc.Hour.ToString("D2");
    MinuteBox.Text = utc.Minute.ToString("D2");
    SecondBox.Text = utc.Second.ToString("D2");
    MillisecondBox.Text = utc.Millisecond.ToString("D3");
  }

  private void CommitTimeFields()
  {
    if (!int.TryParse(HourBox.Text, out int h)) h = 0;
    if (!int.TryParse(MinuteBox.Text, out int m)) m = 0;
    if (!int.TryParse(SecondBox.Text, out int s)) s = 0;
    if (!int.TryParse(MillisecondBox.Text, out int ms)) ms = 0;

    h = Math.Clamp(h, 0, 23);
    m = Math.Clamp(m, 0, 59);
    s = Math.Clamp(s, 0, 59);
    ms = Math.Clamp(ms, 0, 999);

    var current = SelectedEpoch.ToUniversalTime();
    _suppressSync = true;
    SelectedEpoch = new DateTimeOffset(
      current.Year, current.Month, current.Day,
      h, m, s, ms, TimeSpan.Zero);
    SyncTextFromEpoch();
    SyncTimeFields();
    _suppressSync = false;
  }

  private void OnTimeFieldLostFocus(object? sender, RoutedEventArgs e)
  {
    CommitTimeFields();
  }

  private void OnTimeFieldKeyDown(object? sender, KeyEventArgs e)
  {
    if (e.Key == Key.Enter)
    {
      CommitTimeFields();
      TopLevel.GetTopLevel(this)?.FocusManager?.ClearFocus();
    }
  }
}
