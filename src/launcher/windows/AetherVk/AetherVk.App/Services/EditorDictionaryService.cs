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
            public static readonly EditorType SplashScreen = new("SplashScreen");
            public static readonly EditorType Console = new("Console");

            public static implicit operator string(EditorType v) { return v.Value; }

            private EditorType(string value)
            {
                Value = value;
            }

            public string Value { get; private set; }
        }

        private static readonly List<string> _Editors = [
            EditorType.SplashScreen,
            EditorType.Console];

        private static readonly List<EditorInfo> _Descriptors = [
            new EditorInfo("Splash Screen", EditorType.SplashScreen) {
                Glyph = "\uE80F" // Home glyph
            }.EnsureValid(),
            new EditorInfo("Console", EditorType.Console) {
                Glyph = "\uE756" // commandPrompt glyph
            }.EnsureValid()
        ];

        private static readonly Dictionary<string, Type> _EditorTypes = new()
        {
            { EditorType.SplashScreen, typeof(EditorPageSplashScreen)},
            { EditorType.Console, typeof(EditorPageConsole)}
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
