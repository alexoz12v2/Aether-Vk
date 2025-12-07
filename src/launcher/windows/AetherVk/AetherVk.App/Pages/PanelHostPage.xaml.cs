using Microsoft.UI.Composition;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Hosting;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Animation;
using System;
using System.Diagnostics;

namespace AetherVk.Pages
{
    internal sealed partial class PanelHostPage : Page
    {
        public PanelHostPage()
        {
            InitializeComponent();

            OuterBorder.Loaded += (sender, e) => { InitializeBorderVisual(OuterBorder); };
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
            Windows.Foundation.Point pos = e.GetCurrentPoint(border).Position;
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

                // initial gray brush
                _BorderVisual.Brush = _Compositor.CreateColorBrush(Microsoft.UI.Colors.Gray);

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
            System.Diagnostics.Stopwatch stopwatch = new();

            // expression:
            // Hue = fmod( Clock.Time * 60.0, 360.0 )         -> cycles hue at 60 deg/sec
            // Sat = 1.0                                      -> full saturation
            // Lightness = 0.5 + 0.25 * sin(Clock.Time*2.0)   -> gentle pulsation between 0.25..0.75
            // Note: we use fmod (not %) and sin(), and reference Clock.Time
            string expr = "ColorHsl(Mod(Clock.Time * 60.0, 360.0), 1.0, 0.5 + 0.25 * Sin(Clock.Time * 2.0))";
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
                    clock.InsertScalar("Time", stopwatch.ElapsedMilliseconds);
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
                colorBrush.Color = Microsoft.UI.Colors.Gray;
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
