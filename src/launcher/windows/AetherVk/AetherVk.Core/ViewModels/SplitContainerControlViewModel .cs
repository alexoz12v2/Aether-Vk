
using AetherVk.Core.Private;
using AetherVk.Core.Types;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Core.ViewModels
{
    public sealed class GridElementViewModel(Guid id, int row, int col, int rowSpan, int colSpan)
    {
        public Guid Id { get; init; } = id;
        public int Row { get; init; } = row;
        public int Column { get; init; } = col;
        public int RowSpan { get; init; } = rowSpan;
        public int ColumnSpan { get; init; } = colSpan;
        public bool IsSplitter { get; init; } = false;
        // has meaning only if splitter
        public Orientation Orientation { get; init; } = Orientation.Horizontal;
    }

    public sealed class GridDefinitionViewModel
    {
        public bool IsSplitter { get; init; } = false;
    }

    public sealed class SplitCommandData(GridElementViewModel page, float ratio, Orientation orientation)
    {
        public GridElementViewModel Page { get; } = page ?? throw new ArgumentNullException(nameof(page));
        public float Ratio { get; } = ratio is <= 1 and >= 0 ? ratio : throw new ArgumentOutOfRangeException(nameof(ratio));
        public Orientation Orientation { get; } = orientation;
    }

    public sealed partial class SplitContainerControlViewModel : ObservableObject
    {
        private readonly LayoutTree _LayoutTree;

        [ObservableProperty]
        public partial IReadOnlyList<GridDefinitionViewModel> RowDefinitions { get; set; } = [];
        [ObservableProperty]
        public partial IReadOnlyList<GridDefinitionViewModel> ColumnDefinitions { get; set; } = [];
        [ObservableProperty]
        public partial IReadOnlyList<GridElementViewModel> Pages { get; set; } = [];
        [ObservableProperty]
        public partial IReadOnlyList<GridElementViewModel> Splitters { get; set; } = [];

        public SplitContainerControlViewModel()
        {
            // initial layout with single leaf
            _LayoutTree = new LayoutTree(new LeafNode());

            // register for changes 
            _LayoutTree.TreeChanged += OnTreeChanged;

            // manually recompute the initial layout
            RecomputeLayout();
        }

        // TODO add more customization to command
        [RelayCommand]
        private void Split(SplitCommandData data)
        {
            if (data is null) { throw new ArgumentNullException(nameof(data)); }
            _ = _LayoutTree.FindNode(n => (n is LeafNode) && n.Id.Id == data.Page.Id)
                is LeafNode leaf
                ? _LayoutTree.SplitLeaf(leaf, data.Orientation, data.Ratio, new())
                : throw new InvalidOperationException("Couldn't find requested Node");
        }

        private void OnTreeChanged()
        {
            RecomputeLayout();
        }

        private void RecomputeLayout()
        {
            Layout layout = _LayoutTree.ComputeLayout();

            RowDefinitions = [.. layout.RowsDef.Select(r => new GridDefinitionViewModel { IsSplitter = r.IsSplitter })];
            ColumnDefinitions = [.. layout.ColumnsDef.Select(c => new GridDefinitionViewModel { IsSplitter = c.IsSplitter })];
            Pages = [.. layout.Pages.Select(p => new GridElementViewModel(
                id: p.Id, row: p.Row, col: p.Column, rowSpan: p.RowSpan, colSpan: p.ColumnSpan))];
            Splitters = [.. layout.Splitters.Select(s => new GridElementViewModel(
                id: s.Id, row: s.Row, col: s.Column, rowSpan: s.RowSpan, colSpan: s.ColumnSpan) { IsSplitter = true, Orientation = s.Orientation })];
        }
    }
}
