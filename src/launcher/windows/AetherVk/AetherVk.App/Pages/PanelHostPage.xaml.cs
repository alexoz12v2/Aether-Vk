using AetherVk.Core.Types;
using AetherVk.Core.ViewModels;
using AetherVk.UserControls.Shared;
using Microsoft.UI.Composition;
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
using System.Windows.Input;

namespace AetherVk.Pages
{
    internal sealed class EditorToIconConverter : IValueConverter
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

    internal sealed partial class PanelHostPage : Page
    {
        private static readonly Dictionary<EditorType, Type> _EditorsMap = new()
        {
            { EditorType.SplashScreen, typeof(EditorPageSplashScreen) },
            { EditorType.Console, typeof(EditorPageConsole) },
        };

        // View Model
        public PanelHostPageViewModel ViewModel { get; }

        public PanelHostPage(PanelHostPageViewModel viewModel)
        {
            ViewModel = viewModel;
            InitializeComponent();

            // the attached property
            _ = RegisterPropertyChangedCallback(SplitActions.RequestSplitDependencyProperty, (self, dp) =>
            {
                // Retrieve the attached command
                ICommand cmd = SplitActions.GetRequestSplit(this);

                if (cmd != null)
                {
                    Debug.WriteLine("Attached command.");
                }
                else
                {
                    Debug.WriteLine("Attached command cleared or not set yet.");
                }
            });

            // Optional: initial check in case parent already set it
            ICommand initialCmd = SplitActions.GetRequestSplit(this);
            if (initialCmd != null)
            {
                Debug.WriteLine("Attached command.");
            }

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

        private void OuterBorder_PointerMoved(object sender, PointerRoutedEventArgs e)
        {
            Border border = (Border)sender;
            Windows.Foundation.Point pos = e.GetCurrentPoint(border).Position;

            bool withinBorder = IsWithingBorder(pos, border);
            if (withinBorder && !IsHovering)
            {
                StartHoverAnimation();
                Debug.WriteLine("Do Something OuterBorder_PointerMoved");
            }
            else if (!withinBorder && IsHovering)
            {
                StopHoverAnimation();
                Debug.WriteLine("Do Something Else OuterBorder_PointerMoved");
            }
        }

        // TODO: When Resizing, you shouldn't track borders (dep prop)
        private void OuterBorder_PointerExited(object sender, PointerRoutedEventArgs e)
        {
            Border border = (Border)sender;
            _ = e.GetCurrentPoint(border).Position;
            if (IsHovering)
            {
                StopHoverAnimation();
                Debug.WriteLine("Do Something Else OuterBorder_PointerMoved");
            }
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
    }
}
