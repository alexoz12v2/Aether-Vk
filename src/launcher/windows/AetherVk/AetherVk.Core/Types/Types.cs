using System.Windows.Input;

namespace AetherVk.Core.Types
{
    #region SplitContainerControlViewModel
    public enum Orientation
    {
        Horizontal, // children side-by-side (left/right)
        Vertical
    }

    public sealed class GridElementData(Guid id, int row, int col, int rowSpan, int colSpan)
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

    public sealed class GridDefinitionData
    {
        public bool IsSplitter { get; init; } = false;
    }

    public sealed class SplitCommandData(GridElementData page, float ratio, Orientation orientation)
    {
        public GridElementData Page { get; } = page ?? throw new ArgumentNullException(nameof(page));
        public float Ratio { get; } = ratio is <= 1 and >= 0 ? ratio : throw new ArgumentOutOfRangeException(nameof(ratio));
        public Orientation Orientation { get; } = orientation;
    }
    #endregion

    #region PanelHostPageViewModel
    public enum EditorType
    {
        SplashScreen,
        Console
    }

    public class EditorDescriptor(string label, EditorType pageType, ICommand command)
    {
        public string Label { get; } = !string.IsNullOrWhiteSpace(label) ? label : throw new ArgumentNullException(nameof(Label));

        // should be child of Page (Cannot be checked from core, but you can check wtith typeof().IsAssignableFrom
        public EditorType PageType { get; } = pageType;
        public ICommand Command { get; } = command;

        // only one of these should be set on the init. not all of them can be unset
        public string? Glyph { get; set; }
        public string? ImagePath { get; set; }
        public string? VectorData { get; set; }

        // to be Called after init block
        public EditorDescriptor EnsureValid()
        {
            Validate();
            return this;
        }

        private void Validate()
        {
            int setCount = (Glyph is not null ? 1 : 0) + (ImagePath is not null ? 1 : 0) + (VectorData is not null ? 1 : 0);
            if (setCount == 0)
            {
                throw new InvalidOperationException("One of Glyph, ImagePath, VectorData must be specified");
            }
            if (setCount > 1)
            {
                throw new InvalidOperationException("Only one of Glyph, ImagePath, VectorData should be specified");
            }
        }
    }
    #endregion
    #region MessagesModel

    public readonly record struct Point(double X, double Y);

    public enum SplitPreviewKind
    {
        None,
        ValidSplit,
        Coalesce,
        Invalid
    }

    public readonly record struct SplitPreview(
        SplitPreviewKind Kind,
        Orientation Orientation,
        float Ratio
    );

    public record SplitPreviewChanged(object Sender, SplitPreview Preview);
    public record SplitSessionEnded(object Sender);

    #endregion
}