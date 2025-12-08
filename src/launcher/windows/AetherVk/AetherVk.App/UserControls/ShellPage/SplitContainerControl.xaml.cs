using AetherVk.Core.ViewModels;
using AetherVk.Pages;
using CommunityToolkit.WinUI.Controls;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Markup;
using Microsoft.UI.Xaml.Media;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

// the RegisterPropertyChangedCallback method.
// This enables application code to register for change notifications when the specified dependency property is changed

// To reset a value to be the default again,
// and also to enable other participants in precedence that might override the default but not a local value, call the ClearValue method

namespace AetherVk.UserControls
{
    public sealed partial class SplitContainerControl : UserControl
    {
        // an input or output is a dependency property, which you can set with GeneratedDependencyGenerator
        public SplitContainerControl()
        {
            InitializeComponent();

            // initialize dependency properties
            Loaded += SplitContainerControl_OnLoaded;
        }

        private void SplitContainerControl_OnLoaded(object sender, RoutedEventArgs e)
        {
            RebuildColumns(SplitContainerControlViewModel.ColumnDefinitions);
            RebuildRows(SplitContainerControlViewModel.RowDefinitions);
            // we start with one page, useless to rebuild
            if (SplitContainerControlViewModel.Pages.Count != 1 && Container.Children.Count != 0) { throw new InvalidOperationException("dead"); }
            Container.Children.Add(new PanelHostPage());
            ContainerPages.Add(SplitContainerControlViewModel.Pages[0].Id, Container.Children[0]);

            SplitContainerControlViewModel.PropertyChanged += (s, ev) =>
            {
                if (ev.PropertyName == nameof(SplitContainerControlViewModel.ColumnDefinitions))
                {
                    RebuildColumns(SplitContainerControlViewModel.ColumnDefinitions);
                }
                if (ev.PropertyName == nameof(SplitContainerControlViewModel.RowDefinitions))
                {
                    RebuildRows(SplitContainerControlViewModel.RowDefinitions);
                }
                if (ev.PropertyName == nameof(SplitContainerControlViewModel.Pages))
                {
                    RebuildPages(SplitContainerControlViewModel.Pages);
                }
                if (ev.PropertyName == nameof(SplitContainerControlViewModel.Splitters))
                {
                    RebuildSplitters(SplitContainerControlViewModel.Splitters);
                }
            };

            // TODO REMOVE: Insert a new panel after ~5 of runtime, and remove it after 5 seconds. Loop this
            PeriodicTimer timer = new(TimeSpan.FromSeconds(5));
            GridElementViewModel first = SplitContainerControlViewModel.Pages[0];
            _ = Task.Run(async () =>
            {
                while (await timer.WaitForNextTickAsync())
                {
                    _ = DispatcherQueue.TryEnqueue(() =>
                    {
                        SplitContainerControlViewModel.SplitCommand.Execute(new SplitCommandData(
                            page: first, ratio: 0.5f, orientation: Core.Types.Orientation.Vertical));
                    });
                    return;
                }
            });
        }

        private SplitContainerControlViewModel SplitContainerControlViewModel => (SplitContainerControlViewModel)Resources["ViewModel"];

        #region GridDefinitions
        private readonly Dictionary<Guid, UIElement> ContainerPages = [];
        private readonly Dictionary<Guid, GridSplitter> ContainerSplitters = [];

        // Grid.ColumnDefinitions and Grid.RowDefinitions are not DepenencyProperty, meaning we can't bind to them to
        // modify them dynamically directly.
        // Instead, since our ViewModel stores the current layout, we can react to changes in the layout inside our view model,
        // and manually modify the Grid's properties as we need them
        private void RebuildColumns(IReadOnlyList<GridDefinitionViewModel> cols)
        {
            Container.ColumnDefinitions.Clear();
            foreach (GridDefinitionViewModel c in cols)
            {
                Container.ColumnDefinitions.Add(new ColumnDefinition
                {
                    Width = c.IsSplitter ? new GridLength(8) : new GridLength(1, GridUnitType.Star),
                    MinWidth = c.IsSplitter ? 0 : 128
                });
            }
        }

        private void RebuildRows(IReadOnlyList<GridDefinitionViewModel> rows)
        {
            Container.RowDefinitions.Clear();
            foreach (GridDefinitionViewModel r in rows)
            {
                Container.RowDefinitions.Add(new RowDefinition
                {
                    Height = r.IsSplitter ? new GridLength(8) : new GridLength(1, GridUnitType.Star),
                    MinHeight = r.IsSplitter ? 0 : 128
                });
            }
        }
        #endregion
        #region GridContent
        private void RebuildPages(IReadOnlyList<GridElementViewModel> pages)
        {
            // 1) Remove visuals for pages that no longer exist
            HashSet<Guid> pageIdsToKeep = [.. pages.Select(p => p.Id)];
            List<Guid> keysToRemove = [.. ContainerPages.Keys.Where(k => !pageIdsToKeep.Contains(k))];

            foreach (Guid id in keysToRemove)
            {
                if (ContainerPages.TryGetValue(id, out UIElement? element))
                {
                    _ = Container.Children.Remove(element);
                    _ = ContainerPages.Remove(id);
                }
            }

            // 2) Add new pages or update existing pages (row/col/span)
            foreach (GridElementViewModel pageVm in pages)
            {
                if (!ContainerPages.TryGetValue(pageVm.Id, out UIElement? pageElement))
                {
                    // create and add
                    PanelHostPage thePage = new();
                    thePage.SetValue(Grid.RowProperty, pageVm.Row);
                    thePage.SetValue(Grid.ColumnProperty, pageVm.Column);
                    thePage.SetValue(Grid.RowSpanProperty, pageVm.RowSpan);
                    thePage.SetValue(Grid.ColumnSpanProperty, pageVm.ColumnSpan);

                    ContainerPages.Add(pageVm.Id, thePage);
                    Container.Children.Add(thePage);
                }
                else
                {
                    // update the existing element's Grid attached properties if they changed
                    if ((int)pageElement.GetValue(Grid.RowProperty) != pageVm.Row)
                    {
                        pageElement.SetValue(Grid.RowProperty, pageVm.Row);
                    }

                    if ((int)pageElement.GetValue(Grid.ColumnProperty) != pageVm.Column)
                    {
                        pageElement.SetValue(Grid.ColumnProperty, pageVm.Column);
                    }

                    if ((int)pageElement.GetValue(Grid.RowSpanProperty) != pageVm.RowSpan)
                    {
                        pageElement.SetValue(Grid.RowSpanProperty, pageVm.RowSpan);
                    }

                    if ((int)pageElement.GetValue(Grid.ColumnSpanProperty) != pageVm.ColumnSpan)
                    {
                        pageElement.SetValue(Grid.ColumnSpanProperty, pageVm.ColumnSpan);
                    }
                }
            }
        }

        private void RebuildSplitters(IReadOnlyList<GridElementViewModel> splitters)
        {
            // 1) Remove splitters that are no longer present
            HashSet<Guid> splitterIdsToKeep = [.. splitters.Select(s => s.Id)];
            List<Guid> splitterKeysToRemove = [.. ContainerSplitters.Keys.Where(k => !splitterIdsToKeep.Contains(k))];

            foreach (Guid id in splitterKeysToRemove)
            {
                if (ContainerSplitters.TryGetValue(id, out GridSplitter? splitter))
                {
                    _ = Container.Children.Remove(splitter);
                    _ = ContainerSplitters.Remove(id);
                }
            }

            // 2) Add new splitters or update existing ones
            foreach (GridElementViewModel sVm in splitters)
            {
                if (!ContainerSplitters.TryGetValue(sVm.Id, out GridSplitter? existingSplitter))
                {
                    // Why XAML: https://stackoverflow.com/questions/5755455/how-to-set-control-template-in-code
                    // Basically, Templated Controls have no template by default, hence you give it to them
                    string splitterXaml =
                        "<cu:GridSplitter xmlns='http://schemas.microsoft.com/winfx/2006/xaml/presentation'"
                        + " xmlns:x='http://schemas.microsoft.com/winfx/2006/xaml'"
                        + " xmlns:cu='using:CommunityToolkit.WinUI.Controls'"
                        + " ResizeBehavior='BasedOnAlignment'"
                        + " ResizeDirection='Columns'"
                        + " Background='Red'"
                        + " HorizontalAlignment='Stretch'"
                        + " VerticalAlignment='Stretch'"
                        + " Width='8'"
                        + " Height='8' />";

                    GridSplitter splitter = (GridSplitter)XamlReader.LoadWithInitialTemplateValidation(splitterXaml);

                    // Set Grid attached properties dynamically
                    splitter.SetValue(Grid.RowProperty, sVm.Row);
                    splitter.SetValue(Grid.ColumnProperty, sVm.Column);
                    splitter.SetValue(Grid.RowSpanProperty, sVm.RowSpan);
                    splitter.SetValue(Grid.ColumnSpanProperty, sVm.ColumnSpan);

                    ContainerSplitters.Add(sVm.Id, splitter);
                    Container.Children.Add(splitter);
                }
                else
                {
                    // update existing splitter attached properties if needed
                    if ((int)existingSplitter.GetValue(Grid.RowProperty) != sVm.Row)
                    {
                        existingSplitter.SetValue(Grid.RowProperty, sVm.Row);
                    }

                    if ((int)existingSplitter.GetValue(Grid.ColumnProperty) != sVm.Column)
                    {
                        existingSplitter.SetValue(Grid.ColumnProperty, sVm.Column);
                    }

                    if ((int)existingSplitter.GetValue(Grid.RowSpanProperty) != sVm.RowSpan)
                    {
                        existingSplitter.SetValue(Grid.RowSpanProperty, sVm.RowSpan);
                    }

                    if ((int)existingSplitter.GetValue(Grid.ColumnSpanProperty) != sVm.ColumnSpan)
                    {
                        existingSplitter.SetValue(Grid.ColumnSpanProperty, sVm.ColumnSpan);
                    }

                    // if orientation changed, update ResizeDirection
                    GridSplitter.GridResizeDirection newDirection = sVm.Orientation == AetherVk.Core.Types.Orientation.Horizontal
                        ? GridSplitter.GridResizeDirection.Rows
                        : GridSplitter.GridResizeDirection.Columns;

                    if (existingSplitter.ResizeDirection != newDirection)
                    {
                        existingSplitter.ResizeDirection = newDirection;
                    }
                }
            }
        }
        #endregion
    }
}
