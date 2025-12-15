using AetherVk.Core.Interfaces;
using AetherVk.Core.Types;
using AetherVk.Pages;
using System;
using System.Collections.Generic;

namespace AetherVk.Services
{
    internal sealed class EditorDictionaryService : IEditorDictionaryService
    {
        private sealed class EditorType
        {
            // names should be unique
            public static readonly EditorType SplashScreen = new("SplashScreen");
            public static readonly EditorType Console = new("Console");
            public static readonly EditorType Settings = new("Settings");

            public static implicit operator string(EditorType v) { return v.Value; }

            private EditorType(string value)
            {
                Value = value;
            }

            public string Value { get; private set; }
        }

        // editors, descriptors and editor types should be kept in sync and in the correct order
        private static readonly List<string> _Editors = [
            EditorType.SplashScreen,
            EditorType.Console,
            EditorType.Settings
        ];

        // TODO: Labels should be localized
        private static readonly List<EditorInfo> _Descriptors = [
            new EditorInfo("Splash Screen", EditorType.SplashScreen) {
                Glyph = "\uE80F" // Home glyph
            }.EnsureValid(),
            new EditorInfo("Console", EditorType.Console) {
                Glyph = "\uE756" // commandPrompt glyph
            }.EnsureValid(),
            new EditorInfo("Settings", EditorType.Settings) {
                Glyph = "\uE713" // settings glyph
            }.EnsureValid()
        ];

        private static readonly Dictionary<string, Type> _EditorTypes = new()
        {
            { EditorType.SplashScreen, typeof(EditorPageSplashScreen)},
            { EditorType.Console, typeof(EditorPageConsole)},
            { EditorType.Settings, typeof(EditorPageSettings) }
        };

        public IReadOnlyCollection<string> GetEditorNames()
        {
            return _Editors;
        }

        public IReadOnlyCollection<EditorInfo> GetEditors()
        {
            return _Descriptors;
        }

        public IReadOnlyDictionary<string, Type> GetEditorTypes()
        {
            return _EditorTypes;
        }
    }
}
