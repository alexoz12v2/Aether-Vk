namespace AetherVk.Logic.Utils;

public static class CursorWrapHelper
{
    public static bool TryWrapCursor(int currentX, int boundsX, int boundsRight, int margin, int offset, out int newX)
    {
        if (currentX <= boundsX + margin)
        {
            newX = boundsRight - offset;
            return true;
        }
        else if (currentX >= boundsRight - margin)
        {
            newX = boundsX + offset;
            return true;
        }
        
        newX = currentX;
        return false;
    }
}
