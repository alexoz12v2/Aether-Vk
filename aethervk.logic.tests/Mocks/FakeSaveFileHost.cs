using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;

namespace AetherVk.Logic.Tests.Mocks;

/// <summary>
/// An in-memory application. Asset content hashes are the FNV-1a of the file bytes; a mesh file
/// "MESH|&lt;texture text&gt;" also brings an embedded texture with that content (as a GLB does).
/// The compatibility JSON is derived from the committed comet and the jets.
/// </summary>
internal sealed class FakeSaveFileHost(string root) : ISaveFileHost
{
  public readonly string SpkSource = Path.Combine(root, "downloads", "spk_67P_90000703.bsp");
  public TimelineDto Timeline = new() { ProposedStart = "2025-10-01T00:00:00Z", ProposedEnd = "2025-11-11T00:00:00Z" };
  public SpkDto? Comet;
  public ModelDto Model = new();
  public List<JetDto> Jets = [];
  public List<ViewportDto> Viewports = [new ViewportDto()];
  public List<LiveAsset> Assets = [];
  public readonly Dictionary<ulong, byte[]> AssetBytes = [];
  public CometAppearanceDto Appearance;
  public List<BillboardDto> Billboards = [];
  public EpochParts? Clock;
  public uint DumpVersion = 8;
  public bool Running;
  public bool DumpFails;
  public readonly List<string> Calls = [];
  public readonly List<string> Notes = [];
  private ulong _nextId = 1;

  public static ulong Fnv(byte[] bytes)
  {
    ulong h = 0xcbf29ce484222325;
    foreach (var b in bytes) h = unchecked((h ^ b) * 0x100000001b3);
    return h;
  }

  public LiveAsset AddAsset(AssetKind kind, string key, string label, string content)
  {
    var bytes = Encoding.UTF8.GetBytes(content);
    var a = new LiveAsset(_nextId++, kind, key, label, Fnv(bytes), null);
    Assets.Add(a);
    AssetBytes[a.Id] = bytes;
    if (kind == AssetKind.Mesh && content.StartsWith("MESH|", StringComparison.Ordinal))
      AddAsset(AssetKind.Texture, key + "#image0", label + "_albedo", content["MESH|".Length..]);
    return a;
  }

  public SaveSnapshot Capture() => new(
    Clone(Timeline),
    Comet is null ? null : new LiveComet(SpkSource, Comet),
    Model,
    [.. Jets],
    [.. Viewports],
    [.. Assets],
    SaveIndexMapping.ToDto(Appearance, id => Assets.FirstOrDefault(a => a.Id == id)?.Key),
    [.. Billboards]);

  private static TimelineDto Clone(TimelineDto t) => new()
  {
    CommittedStart = t.CommittedStart, CommittedEnd = t.CommittedEnd, ProposedStart = t.ProposedStart,
    ProposedEnd = t.ProposedEnd, CurrentEpoch = t.CurrentEpoch, StartPaused = t.StartPaused,
  };

  public string? GetCompatibilityJson() =>
    Comet is null
      ? "{\"comet\":null}"
      : $"{{\"comet\":{Comet.NaifId},\"radius\":{Comet.NucleusRadiusKm},\"jets\":[{string.Join(",", Jets.Select(j => j.LatitudeDeg))}]}}";

  public uint SceneDumpVersion => DumpVersion;

  public Task<bool> DumpSceneAsync(string file)
  {
    Calls.Add("dump");
    if (DumpFails) return Task.FromResult(false);
    File.WriteAllText(file, GetCompatibilityJson() + "|" + Timeline.CurrentEpoch?.Nanoseconds);
    return Task.FromResult(true);
  }

  public ExportedAssetFile? ExportAsset(ulong assetId, string destDir)
  {
    var a = Assets.First(x => x.Id == assetId);
    Directory.CreateDirectory(destDir);
    var name = $"{SaveIndexMapping.HashHex(a.ContentHash)}_{a.Label}.{(a.Kind == AssetKind.Mesh ? "glb" : "png")}";
    var path = Path.Combine(destDir, name);
    bool reused = File.Exists(path);
    if (!reused) File.WriteAllBytes(path, AssetBytes[a.Id]);
    return new ExportedAssetFile(name, a.ContentHash, reused);
  }

  public Task ResetAsync()
  {
    Calls.Add("reset");
    Timeline = new TimelineDto();
    Comet = null;
    Model = new ModelDto();
    Jets = [];
    Viewports = [new ViewportDto()];
    Assets = [];
    AssetBytes.Clear();
    Appearance = default;
    Billboards = [];
    return Task.CompletedTask;
  }

  public Task<bool> ImportAssetAsync(string path)
  {
    Calls.Add("import " + Path.GetFileName(path));
    var content = File.ReadAllText(path);
    var kind = path.EndsWith(".glb", StringComparison.Ordinal) ? AssetKind.Mesh : AssetKind.Texture;
    var hash = Fnv(Encoding.UTF8.GetBytes(content));
    if (Assets.Any(a => a.ContentHash == hash)) return Task.FromResult(true); // dedup
    AddAsset(kind, path, Path.GetFileNameWithoutExtension(path).Split('_').Last(), content);
    return Task.FromResult(true);
  }

  public IReadOnlyList<LiveAsset> ListAssets() => [.. Assets];

  public void ApplyModel(ModelDto model, IReadOnlyList<JetDto> jets)
  {
    Model = model;
    Jets = [.. jets];
  }

  public void ApplyTimeline(TimelineDto timeline, bool committed)
  {
    Timeline.ProposedStart = timeline.ProposedStart;
    Timeline.ProposedEnd = timeline.ProposedEnd;
    if (committed)
    {
      Timeline.CommittedStart = timeline.CommittedStart;
      Timeline.CommittedEnd = timeline.CommittedEnd;
    }
  }

  public Task<bool> CommitCometAsync(string spkPath, SpkDto spk, TimeRange range)
  {
    Calls.Add("commit " + Path.GetRelativePath(root, spkPath));
    Comet = spk;
    Timeline.CurrentEpoch = new EpochParts(range.StartCenturies, range.StartNs);
    return Task.FromResult(true);
  }

  public bool ApplyAppearance(CometAppearanceDto appearance)
  {
    Appearance = appearance;
    return true;
  }

  public Task SettleAsync() => Task.CompletedTask;

  public Task<bool> RestoreSceneAsync(string file)
  {
    Calls.Add("restore " + Path.GetRelativePath(root, file));
    var parts = File.ReadAllText(file).Split('|');
    if (parts[0] != GetCompatibilityJson()) return Task.FromResult(false);
    Timeline.CurrentEpoch = new EpochParts(0, ulong.Parse(parts[1]));
    return Task.FromResult(true);
  }

  public Task<bool> SeekAsync(EpochParts epoch)
  {
    Calls.Add("seek " + epoch.Nanoseconds);
    Timeline.CurrentEpoch = epoch;
    return Task.FromResult(true);
  }

  public void ApplyViewports(IReadOnlyList<ViewportDto> viewports) => Viewports = [.. viewports];

  public void ApplyBillboards(IReadOnlyList<BillboardDto> billboards) => Billboards = [.. billboards];

  public int ClockSyncs { get; private set; }
  public void SyncClock() => ClockSyncs++;

  public int DustFeedbackRestarts { get; private set; }
  public void RestartDustFeedbackWhenSettled() => DustFeedbackRestarts++;

  public IObservable<System.Reactive.Unit> ConfigurationChanged => System.Reactive.Linq.Observable.Never<System.Reactive.Unit>();

  public bool IsSimulationRunning => Running;

  public void Notify(string title, string message, bool error) => Notes.Add(message);
}
