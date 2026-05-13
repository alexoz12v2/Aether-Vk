using System;

namespace AetherVk.Logic.Services
{
  public interface IViewportRenderer
  {
    void UpdateFrame(IntPtr buffer, nuint bufferSize);
  }
}