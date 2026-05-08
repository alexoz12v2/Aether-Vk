using System.Collections.Generic;

namespace AetherVk.Logic.Input;

public record InputChord(string? Key = null, bool Shift = false, bool Ctrl = false, bool Alt = false, string? Pointer = null)
{
    public string DisplayText 
    {
        get 
        {
            List<string> parts = new List<string>();
            if (Ctrl) parts.Add("Ctrl");
            if (Alt) parts.Add("Alt");
            if (Shift) parts.Add("Shift");
            if (Key is not null) parts.Add(Key);
            if (Pointer is not null) parts.Add(Pointer.Replace("ButtonPressed", " Click"));
            return string.Join(" + ", parts);
        }
    }
}
