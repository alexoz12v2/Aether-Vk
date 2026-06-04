using System;
using System.Threading.Tasks;
using Xunit;
using Moq;
using AetherVk.Logic.Services;
using AetherVk.Logic.Models;

namespace AetherVk.Logic.Tests
{
    public class AudioIntegrationTests
    {
        [Fact]
        public async Task Audio2DService_PlayClick_DispatchesToNative()
        {
            // Arrange
            var mockNative = new Mock<INativeRuntimeService>();
            mockNative.SetupGet(n => n.IsInitialized).Returns(true);
            
            var audioService = new Audio2DService(mockNative.Object);

            // Act
            await audioService.PlayClickAsync();

            // Assert
            mockNative.Verify(n => n.PlaySound(
                AvkSoundEvent.UiClick,
                It.Is<AvkAudioParams>(p => p.Volume == 1.0f && p.Pitch == 1.0f && p.Pan == 0.0f)
            ), Times.Once);
        }

        [Fact]
        public async Task Audio2DService_PlayGrab_DispatchesToNative()
        {
            // Arrange
            var mockNative = new Mock<INativeRuntimeService>();
            mockNative.SetupGet(n => n.IsInitialized).Returns(true);
            
            var audioService = new Audio2DService(mockNative.Object);

            // Act
            await audioService.PlayGrabAsync();

            // Assert
            mockNative.Verify(n => n.PlaySound(
                AvkSoundEvent.UiGrab,
                It.Is<AvkAudioParams>(p => p.Volume == 1.0f && p.Pitch == 1.0f && p.Pan == 0.0f)
            ), Times.Once);
        }

        [Fact]
        public async Task Audio2DService_PlayDrop_DispatchesToNative()
        {
            // Arrange
            var mockNative = new Mock<INativeRuntimeService>();
            mockNative.SetupGet(n => n.IsInitialized).Returns(true);
            
            var audioService = new Audio2DService(mockNative.Object);

            // Act
            await audioService.PlayDropAsync();

            // Assert
            mockNative.Verify(n => n.PlaySound(
                AvkSoundEvent.UiDrop,
                It.Is<AvkAudioParams>(p => p.Volume == 1.0f && p.Pitch == 1.0f && p.Pan == 0.0f)
            ), Times.Once);
        }
    }
}
