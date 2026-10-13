using System.Threading.Tasks;

namespace AetherVk.Logic.Services;

public enum UnsavedChoice
{
  Save,
  Discard,
  Cancel,
}

/// <summary>Asks what to do with unsaved changes before New, Open or closing the application.</summary>
public interface IUnsavedChangesPrompt
{
  Task<UnsavedChoice> AskAsync(string saveName);
}
