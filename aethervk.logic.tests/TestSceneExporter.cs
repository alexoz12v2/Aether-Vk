using System;
using System.IO;
using System.Text.Json;
using System.Linq;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace AetherVk.Logic.Tests
{
    public static class TestSceneExporter
    {
        public static void EnsureRenderDirectory()
        {
            if (!Directory.Exists("render"))
            {
                Directory.CreateDirectory("render");
            }
        }

        public static void ExportScene(ulong sceneId, SceneStateManager stateManager, string testName)
        {
            EnsureRenderDirectory();
            var scene = stateManager.GetOrCreateScene(sceneId);
            
            var options = new JsonSerializerOptions { WriteIndented = true };
            
            // Map the hierarchy into a clean structure for JSON
            object MapEntity(Entity e)
            {
                return new 
                {
                    Id = e.Id,
                    Name = e.Name,
                    Components = e.Components.Select(c => c.Name).ToList(),
                    Children = e.Children.Select(MapEntity).ToList()
                };
            }

            var root = scene.RootEntities.FirstOrDefault();
            object rootData = root != null ? MapEntity(root) : null;

            var json = JsonSerializer.Serialize(rootData, options);
            File.WriteAllText($"render/{testName}_scene.json", json);
        }

        public static void ExportPng(byte[] bgraPixels, int width, int height, string testName)
        {
            EnsureRenderDirectory();
            using var image = Image.LoadPixelData<Bgra32>(bgraPixels, width, height);
            image.SaveAsPng($"render/{testName}.png");
        }
    }
}
