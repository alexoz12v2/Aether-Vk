using System;
using System.Threading.Tasks;
using AetherVk.Logic.Models;

namespace AetherVk.Logic.Services
{
    public interface IAudio2DService
    {
        Task PlayClickAsync();
        Task PlayDropAsync();
        Task PlayGrabAsync();
        Task PlaySoundAsync(AvkSoundEvent soundEvent, float volume = 1.0f, float pitch = 1.0f, float pan = 0.0f);
    }

    public class Audio2DService : IAudio2DService
    {
        private readonly INativeRuntimeService _nativeRuntime;

        public Audio2DService(INativeRuntimeService nativeRuntime)
        {
            _nativeRuntime = nativeRuntime;
        }

        public Task PlayClickAsync()
        {
            return PlaySoundAsync(AvkSoundEvent.UiClick);
        }

        public Task PlayDropAsync()
        {
            return PlaySoundAsync(AvkSoundEvent.UiDrop);
        }

        public Task PlayGrabAsync()
        {
            return PlaySoundAsync(AvkSoundEvent.UiGrab);
        }

        public Task PlaySoundAsync(AvkSoundEvent soundEvent, float volume = 1.0f, float pitch = 1.0f, float pan = 0.0f)
        {
            // We dispatch to the native core which natively pushes to the WASAPI/CoreAudio thread.
            var audioParams = new AvkAudioParams
            {
                Volume = volume,
                Pitch = pitch,
                Pan = pan,
                Mode = AvkAudioPlaybackMode.StereoDirect
            };

            // Using Task.Run to fire and forget into the native API asynchronously
            // so we don't block the Avalonia UI dispatcher thread if the FFI blocks slightly.
            return Task.Run(() => 
            {
                _nativeRuntime.PlaySound(soundEvent, audioParams);
            });
        }
    }
}
