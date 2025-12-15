using AetherVk.Core.Types;
using AetherVk.Core.ViewModels;
using Microsoft.UI.Composition;
using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Data;
using Microsoft.UI.Xaml.Hosting;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Markup;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Animation;
using Microsoft.UI.Xaml.Media.Imaging;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Reflection;

namespace AetherVk.Pages
{
    internal sealed partial class EditorToIconConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, string language)
        {
            if (value is not EditorDescriptor ed)
            {
                return null!;
            }
            if (!string.IsNullOrWhiteSpace(ed.Glyph))
            {
                return new FontIcon { Glyph = ed.Glyph };
            }
            if (!string.IsNullOrWhiteSpace(ed.ImagePath))
            {
                return new ImageIcon { Source = new BitmapImage(new Uri(ed.ImagePath, UriKind.RelativeOrAbsolute)) };
            }
            if (!string.IsNullOrWhiteSpace(ed.VectorData))
            {
                // https://stackoverflow.com/questions/34880793/is-there-a-way-to-parse-a-vector-path-geometry-in-uwp-in-code-behind
                return new PathIcon { Data = (Geometry)XamlBindingHelper.ConvertValue(typeof(Geometry), ed.VectorData) };
            }
            return null!;
        }

        public object ConvertBack(object value, Type targetType, object parameter, string language)
        {
            throw new NotSupportedException(nameof(ConvertBack));
        }
    }

    // TODO put internal and frient assembly core (needed to use it as type parameter for interfaces coming from core)
    // I tried, it didnt' work (from <AssemblyAttribute> in csproj. I have MSBuild)
    public sealed partial class PanelHostPage : Page
    {
        private static readonly Dictionary<EditorType, Type> _EditorsMap = new()
        {
            { EditorType.SplashScreen, typeof(EditorPageSplashScreen) },
            { EditorType.Console, typeof(EditorPageConsole) },
        };

        // View Model
        public PanelHostPageViewModel ViewModel => (PanelHostPageViewModel)DataContext;

        public PanelHostPage()
        {
            InitializeComponent();

            // visual changes once XAML template loaded
            OuterBorder.Loaded += OuterBorder_Loaded;

            // once the whole page has been loaded, it's safe to act on the template, in particular,
            // we can navigate to our initial page
            Loaded += PanelHostPage_OnLoaded;
        }

        private void PanelHostPage_OnLoaded(object sender, RoutedEventArgs e)
        {
            if (EditorFrame.Content == null)
            {
                _ = EditorFrame.Navigate(_EditorsMap.GetValueOrDefault(ViewModel.SelectedEditor, typeof(EditorPageSplashScreen)));
            }

            ViewModel.PropertyChanged += ViewModel_PropertyChanged;
        }

        private void ViewModel_PropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(ViewModel.SelectedEditor))
            {
                _ = EditorFrame.Navigate(_EditorsMap.GetValueOrDefault(ViewModel.SelectedEditor, typeof(EditorPageSplashScreen)));
            }
        }

        private void OuterBorder_Loaded(object sender, RoutedEventArgs e)
        {
            InitializeBorderVisual(OuterBorder);
        }

        private void HeaderFlyout_Opened(object sender, object e)
        {
            Storyboard sb = (Storyboard)Resources["ChevronRotateUp"];
            sb.Begin();
        }

        private void HeaderFlyout_Closed(object sender, object e)
        {
            Storyboard sb = (Storyboard)Resources["ChevronRotateDown"];
            sb.Begin();
        }

        private bool _IsDragging = false;
        private uint _ActivePointerId = uint.MaxValue;
        private bool IsDragging => _IsDragging && _ActivePointerId != uint.MaxValue;

        private void OuterBorder_PointerMoved(object sender, PointerRoutedEventArgs e)
        {
            if (!IsDragging)
            {

                Border border = (Border)sender;
                Windows.Foundation.Point pos = e.GetCurrentPoint(border).Position;

                bool withinBorder = IsWithingBorder(pos, border);
                if (withinBorder && !IsHovering)
                {
                    StartHoverAnimation();
                }
                else if (!withinBorder && IsHovering)
                {
                    StopHoverAnimation();
                }
            }
            else
            {
                PointerPoint point = e.GetCurrentPoint((UIElement)Parent);
                if (point.PointerId != _ActivePointerId) { return; }
                ViewModel.UpdateSplitSession(ToCorePoint(point.Position));
            }
        }

        // TODO: When Resizing, you shouldn't track borders (dep prop)
        private void OuterBorder_PointerExited(object sender, PointerRoutedEventArgs e)
        {
            if (IsHovering)
            {
                StopHoverAnimation();
            }
        }

        private void OuterBorder_PointerPressed(object sender, PointerRoutedEventArgs e)
        {
            if (!IsHovering) { return; }

            PointerPoint point = e.GetCurrentPoint((UIElement)Parent);
            if (!point.Properties.IsLeftButtonPressed) { return; }

            Border border = (Border)sender;
            _IsDragging = true;
            _ActivePointerId = point.PointerId;

            _ = border.CapturePointer(e.Pointer);
            StopHoverAnimation();

            ViewModel.BeginSplitSession(ToCorePoint(point.Position), ToCoreBounds(point.Position, ActualSize));
        }

        private void OuterBorder_PointerReleased(object sender, PointerRoutedEventArgs e)
        {
            if (!IsDragging) { return; }
            PointerPoint point = e.GetCurrentPoint((UIElement)Parent);
            if (point.PointerId != _ActivePointerId) { return; }
            // TODO Customize cancel action
            bool cancel = point.Properties.IsRightButtonPressed;

            Border border = (Border)sender;
            border.ReleasePointerCapture(e.Pointer);
            _IsDragging = false;
            _ActivePointerId = uint.MaxValue;

            ViewModel.EndSplitSession(ToCorePoint(point.Position), cancel);
        }

        private static Core.Types.Point ToCorePoint(Windows.Foundation.Point position)
        {
            return new Point { X = position.X, Y = position.Y };
        }

        private RectD ToCoreBounds(Windows.Foundation.Point position, Vector2 actualSize)
        {
            return new RectD(X: position.X, Y: position.Y, Width: actualSize.X, Height: actualSize.Y);
        }

        private static bool IsWithingBorder(Windows.Foundation.Point pos, Border border)
        {
            return pos.X <= border.BorderThickness.Left ||
                pos.Y <= border.BorderThickness.Top ||
                pos.X >= (border.ActualWidth - border.BorderThickness.Right) ||
                pos.Y >= (border.ActualHeight - border.BorderThickness.Bottom);
        }

        private void InitializeBorderVisual(Border border)
        {
            _Compositor = CompositionTarget.GetCompositorForCurrentThread();


            // create sprite visual if needed
            if (_BorderVisual == null)
            {
                _BorderVisual = _Compositor.CreateSpriteVisual();
                // set initial size to actual size (will update on size changed)
                _BorderVisual.Size = new System.Numerics.Vector2((float)border.ActualWidth, (float)border.ActualHeight);

                // initial brush
                _BorderVisual.Brush = _Compositor.CreateColorBrush(((SolidColorBrush)Resources["TabColor"]).Color);

                // attach the visual to the Border
                ElementCompositionPreview.SetElementChildVisual(border, _BorderVisual);

                // keep the visual in sync when the border is resized
                border.SizeChanged += (s, e) =>
                {
                    _BorderVisual.Size = new System.Numerics.Vector2((float)border.ActualWidth, (float)border.ActualHeight);
                };
            }


        }

        private void StartHoverAnimation()
        {
            if (_HsvAnimation != null || _Compositor == null)
            {
                return; // already animating
            }

            // animate value as a function of time
            CompositionPropertySet clock = _Compositor.CreatePropertySet();
            clock.InsertScalar("Time", 0);
            clock.InsertScalar("Alpha", ((SolidColorBrush)Resources["TabColor"]).Color.A);
            System.Diagnostics.Stopwatch stopwatch = new();

            float frequency = 0.5f * 2f * float.Pi;
            float target = 64;
            string sinVal = $"Clamp((Sin(Clock.Time * {frequency}) + 1) * {target / 2}, 0, {target})";
            string expr = $"ColorRGB(Clock.Alpha, {sinVal}, {sinVal}, {sinVal})";

            _HsvAnimation = _Compositor.CreateExpressionAnimation(expr);
            _HsvAnimation.SetReferenceParameter("Clock", clock);

            // Bind the brush
            if (_BorderVisual?.Brush is CompositionColorBrush colorBrush)
            {
                colorBrush.StartAnimation(nameof(CompositionColorBrush.Color), _HsvAnimation);
                stopwatch.Start();
                // tick the clock
                _RenderingHandler = (sender, e) =>
                {
                    float seconds = (float)stopwatch.ElapsedMilliseconds / 1000;
                    clock.InsertScalar("Time", seconds);
                };
                CompositionTarget.Rendering += _RenderingHandler;
            }
        }

        private void StopHoverAnimation()
        {
            if (!IsHovering)
            {
                return;
            }

            if (_RenderingHandler != null)
            {
                CompositionTarget.Rendering -= _RenderingHandler;
                _RenderingHandler = null;
            }

            // Cleanup animation
            if (_BorderVisual?.Brush is CompositionColorBrush colorBrush)
            {
                colorBrush.StopAnimation(nameof(CompositionColorBrush.Color));
                colorBrush.Color = ((SolidColorBrush)Resources["TabColor"]).Color;
            }

            _HsvAnimation = null;
        }

        // Border Hovering Event Tracking 
        private bool IsHovering => _HsvAnimation != null;

        // Fields for Animation
        private Compositor? _Compositor;
        private SpriteVisual? _BorderVisual;
        private ExpressionAnimation? _HsvAnimation = null;
        private EventHandler<object>? _RenderingHandler;

        private void OuterBorder_PointerCanceled(object sender, PointerRoutedEventArgs e)
        {
            if (!IsDragging) { return; }
            Border border = (Border)sender;
            border.ReleasePointerCapture(e.Pointer);

            _IsDragging = false;
            _ActivePointerId = uint.MaxValue;

            ViewModel.EndSplitSession(default, true);
        }
    }
}
