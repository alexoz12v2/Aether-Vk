using System.Windows.Input;

namespace AetherVk.Core.Types
{
    // Used by SplitContainerControlViewModel
    public enum Orientation
    {
        Horizontal, // children side-by-side (left/right)
        Vertical
    }

    // Used by PanelHostPageViewModel
    public enum EditorType
    {
        SplashScreen
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
}