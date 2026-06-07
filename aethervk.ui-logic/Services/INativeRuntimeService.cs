using System;
using System.Threading.Tasks;
using AetherVk.Logic.Models;

namespace AetherVk.Logic.Services
{
  public interface INativeRuntimeService
  {
    bool IsInitialized { get; }
    void InitializeSimulationContext(
      string backend,
      string? assetOverride = null,
      bool populateDefault = true
    );
    void PlaySound(AvkSoundEvent soundEvent, AvkAudioParams audioParams);
    // We can add more as needed, but this is enough for Audio2DService
  }
}
