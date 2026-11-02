using System.Buffers.Binary;
using System.Drawing;
using System.IO.Compression;
using System.Text.Json;

namespace AetherVk.UI.E2E.Manual;

/// <summary>A control found for a shot, in absolute desktop px (before it is made relative to the crop).</summary>
internal sealed record RawCallout(string Id, string Label, Rectangle Rect, string? Type = null, string? HelpId = null);

/// <summary>One line of <c>&lt;shot&gt;.callouts.json</c>: a numbered control, relative to the crop.</summary>
internal sealed record CalloutOut(int N, string Id, string Label, int X, int Y, int W, int H, string? Type, string? HelpId);

internal sealed record ShotRecord(
  string Key, string State, string File, string Status, string? Reason,
  int X, int Y, int W, int H, bool Clipped, int Callouts, int[]? Calibration);

/// <summary>
/// Pixel work for the manual's screenshots without an image library (System.Drawing.Common is not
/// referenced): a PNG writer with the 150 dpi <c>pHYs</c> chunk, the numbered round markers, and the
/// json files.
/// </summary>
internal static class ShotTools
{
  /// <summary>Written into every PNG: all UI figures share one scale (1 UI px = 0.169 mm).</summary>
  public const int Dpi = 150;

  public const int MarkerRadius = 11;

  private static readonly JsonSerializerOptions JsonOptions = new()
  {
    WriteIndented = true,
    PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    DictionaryKeyPolicy = JsonNamingPolicy.SnakeCaseLower,
  };

  public static string ToJson(object value) => JsonSerializer.Serialize(value, JsonOptions);

  /// <summary>
  /// The callouts of a crop: those whose centre lies inside it, at most <paramref name="cap"/>,
  /// numbered top to bottom (then left to right). When more qualify, the interactive controls win.
  /// </summary>
  public static List<CalloutOut> ToRelative(IEnumerable<RawCallout> raw, Rectangle crop, int cap = 15)
  {
    var inside = raw
      .Where(c => c.Rect.Width > 0 && c.Rect.Height > 0 && crop.Contains(Center(c.Rect)))
      .ToList();
    if (inside.Count > cap)
      inside = inside.OrderBy(c => Interactive(c.Type) ? 0 : 1).ThenBy(c => c.Rect.Y).Take(cap).ToList();
    var ordered = inside.OrderBy(c => c.Rect.Y).ThenBy(c => c.Rect.X).ToList();
    var result = new List<CalloutOut>(ordered.Count);
    for (int i = 0; i < ordered.Count; i++)
    {
      var c = ordered[i];
      var r = Rectangle.Intersect(c.Rect, crop);
      result.Add(new CalloutOut(i + 1, c.Id, c.Label, r.X - crop.X, r.Y - crop.Y, r.Width, r.Height, c.Type, c.HelpId));
    }
    return result;
  }

  private static Point Center(Rectangle r) => new(r.X + r.Width / 2, r.Y + r.Height / 2);

  private static bool Interactive(string? type) =>
    type is "Button" or "CheckBox" or "RadioButton" or "ComboBox" or "TextBox" or "Slider" or "UnboundedSlider"
      or "ColorPicker" or "ListBox" or "Expander" or "NumericUpDown";

  /// <summary>
  /// The callouts numbered in the order a reader meets the markers: a wide strip's (see
  /// <see cref="Tagged"/>) top margin left to right, then its bottom margin; a panel's top to bottom.
  /// </summary>
  public static List<CalloutOut> Renumber(IReadOnlyList<CalloutOut> callouts, int width, int height)
  {
    bool wide = width > 2.2 * height;
    var ordered = wide
      ? callouts.OrderBy(c => c.Y + c.H / 2 < height / 2 ? 0 : 1).ThenBy(c => c.X + Math.Min(c.W / 2, 40)).ToList()
      : callouts.OrderBy(c => c.H > 60 ? c.Y + 16 : c.Y + c.H / 2).ThenBy(c => c.X).ToList();
    return ordered.Select((c, i) => c with { N = i + 1 }).ToList();
  }

  /// <summary>White margin each side of a tagged image: the markers sit there, off the controls.</summary>
  public const int Margin = 46;

  private const int Pad = 8;

  /// <summary>
  /// The pixels on a white page with margins for the markers, off the controls. A panel gets them left
  /// and right (left half of the image → left margin); a wide strip (wider than 2.2 × its height, the
  /// Timeline) above and below (upper half → top). Each marker is joined to its control by a thin
  /// leader ending in a dot on the control's near edge (at the header of a tall container), and the
  /// markers of one margin are spread so they don't overlap.
  /// </summary>
  public static byte[] Tagged(byte[] bgra, int width, int height, IReadOnlyList<CalloutOut> callouts, out int tw, out int th)
  {
    bool wide = width > 2.2 * height;
    int ml = wide ? Pad : Margin, mt = wide ? Margin : Pad;
    tw = width + 2 * ml;
    th = height + 2 * mt;
    var page = new byte[tw * th * 4];
    Array.Fill(page, (byte)255);
    for (int y = 0; y < height; y++)
      Buffer.BlockCopy(bgra, 4 * y * width, page, 4 * ((y + mt) * tw + ml), 4 * width);
    // a hairline frame around the screenshot
    var frame = (200, 200, 200);
    for (int x = ml - 1; x <= ml + width; x++)
    {
      Blend(page, tw, x, mt - 1, frame, 1);
      Blend(page, tw, x, mt + height, frame, 1);
    }
    for (int y = mt - 1; y <= mt + height; y++)
    {
      Blend(page, tw, ml - 1, y, frame, 1);
      Blend(page, tw, ml + width, y, frame, 1);
    }
    int pageW = tw, pageH = th;
    foreach (bool first in new[] { true, false }) // left / top margin first
    {
      // along: the coordinate the markers spread on; across: the margin's fixed coordinate
      var side = callouts
        .Where(c => wide ? (c.Y + c.H / 2 < height / 2) == first : (c.X + c.W / 2 < width / 2) == first)
        .Select(c =>
        {
          // a tall container (a section): point at its header, not the middle of its body
          int ty = c.H > 60 ? c.Y + 16 : c.Y + c.H / 2;
          (int x, int y) t = wide
            ? (ml + c.X + Math.Min(c.W / 2, 40), mt + (first ? c.Y + 2 : c.Y + c.H - 3))
            : (ml + (first ? c.X + 2 : c.X + c.W - 3), mt + ty);
          return (c, t);
        })
        .OrderBy(x => wide ? x.t.x : x.t.y).ThenBy(x => x.c.N).ToList();
      int spacing = 2 * MarkerRadius + (wide ? 14 : 6), lo = MarkerRadius + 3 + (wide ? 4 : 0);
      int hi = (wide ? pageW : pageH) - lo;
      var pos = new int[side.Count];
      for (int i = 0; i < side.Count; i++)
        pos[i] = Math.Max(Math.Clamp(wide ? side[i].t.x : side[i].t.y, lo, hi), i > 0 ? pos[i - 1] + spacing : lo);
      for (int i = side.Count - 1; i >= 0; i--) // pushed past the end: pull back
        pos[i] = Math.Min(pos[i], i < side.Count - 1 ? pos[i + 1] - spacing : hi);
      int across = wide ? (first ? Margin / 2 - 2 : pageH - Margin / 2 + 2) : (first ? Margin / 2 - 2 : pageW - Margin / 2 + 2);
      int edge = wide ? (first ? mt - 1 : mt + height) : (first ? ml - 1 : ml + width);
      for (int i = 0; i < side.Count; i++)
      {
        var t = side[i].t;
        if (wide)
        {
          DrawLine(page, pageW, pageH, pos[i], across, pos[i], edge);
          DrawLine(page, pageW, pageH, pos[i], edge, t.x, t.y);
        }
        else
        {
          DrawLine(page, pageW, pageH, across, pos[i], edge, pos[i]);
          DrawLine(page, pageW, pageH, edge, pos[i], t.x, t.y);
        }
        DrawDot(page, pageW, pageH, t.x, t.y, 2.6);
      }
      for (int i = 0; i < side.Count; i++)
        DrawMarker(page, pageW, pageH, wide ? pos[i] : across, wide ? across : pos[i], side[i].c.N);
    }
    return page;
  }

  private static void DrawLine(byte[] bgra, int width, int height, int x0, int y0, int x1, int y1)
  {
    var color = (Accent.R, Accent.G, Accent.B);
    double len = Math.Max(Math.Abs(x1 - x0), Math.Abs(y1 - y0));
    if (len == 0)
      return;
    // 1.6 px wide: each pixel within 0.8 px of the line, blended once with its best coverage
    var cover = new Dictionary<int, double>();
    for (int i = 0; i <= len * 2; i++)
    {
      double fx = x0 + (x1 - x0) * i / (len * 2), fy = y0 + (y1 - y0) * i / (len * 2);
      for (int px = (int)Math.Floor(fx - 1); px <= (int)Math.Ceiling(fx + 1); px++)
        for (int py = (int)Math.Floor(fy - 1); py <= (int)Math.Ceiling(fy + 1); py++)
        {
          if (px < 0 || py < 0 || px >= width || py >= height)
            continue;
          double d = Math.Sqrt((px - fx) * (px - fx) + (py - fy) * (py - fy));
          double a = Math.Clamp(0.8 + 0.5 - d, 0, 1);
          int k = py * width + px;
          if (a > 0 && (!cover.TryGetValue(k, out var had) || had < a))
            cover[k] = a;
        }
    }
    foreach (var (k, a) in cover)
      Blend(bgra, width, k % width, k / width, color, a);
  }

  private static void DrawDot(byte[] bgra, int width, int height, int cx, int cy, double r)
  {
    for (int y = cy - 4; y <= cy + 4; y++)
      for (int x = cx - 4; x <= cx + 4; x++)
      {
        if (x < 0 || y < 0 || x >= width || y >= height)
          continue;
        double d = Math.Sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
        Blend(bgra, width, x, y, (255, 255, 255), Math.Clamp(r + 1.4 - d, 0, 1));
        Blend(bgra, width, x, y, (Accent.R, Accent.G, Accent.B), Math.Clamp(r + 0.5 - d, 0, 1));
      }
  }

  // accent circle, white ring and digits
  private static readonly (byte B, byte G, byte R) Accent = (0xE0, 0x6A, 0x1F); // BGR of #1F6AE0

  private static readonly string[][] Glyphs =
  [
    ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
  ];

  /// <summary>A filled accent disc (a capsule for two digits) with a white ring, the number in bold white.</summary>
  public static void DrawMarker(byte[] bgra, int width, int height, int cx, int cy, int number)
  {
    const int scale = 2, gap = 2, glyphW = 5 * scale, glyphH = 7 * scale;
    var text = number.ToString(System.Globalization.CultureInfo.InvariantCulture);
    int textW = text.Length * glyphW + (text.Length - 1) * gap;
    int r = MarkerRadius;
    int half = Math.Max(0, (textW + 8) / 2 - r); // capsule half-length
    for (int y = cy - r - 3; y <= cy + r + 3; y++)
    {
      for (int x = cx - half - r - 3; x <= cx + half + r + 3; x++)
      {
        if (x < 0 || y < 0 || x >= width || y >= height)
          continue;
        double dx = x < cx - half ? cx - half - x : x > cx + half ? x - (cx + half) : 0;
        double d = Math.Sqrt(dx * dx + (double)(y - cy) * (y - cy));
        Blend(bgra, width, x, y, (255, 255, 255), Math.Clamp(r + 1.5 - d + 0.5, 0, 1));
        Blend(bgra, width, x, y, (Accent.R, Accent.G, Accent.B), Math.Clamp(r - d + 0.5, 0, 1));
      }
    }
    int x0 = cx - textW / 2, y0 = cy - glyphH / 2;
    for (int i = 0; i < text.Length; i++)
    {
      var glyph = Glyphs[text[i] - '0'];
      for (int gy = 0; gy < 7; gy++)
        for (int gx = 0; gx < 5; gx++)
          if (glyph[gy][gx] == '1')
            for (int sy = 0; sy < scale; sy++)
              for (int sx = 0; sx < scale; sx++)
              {
                int px = x0 + i * (glyphW + gap) + gx * scale + sx, py = y0 + gy * scale + sy;
                if (px >= 0 && py >= 0 && px < width && py < height)
                  Blend(bgra, width, px, py, (255, 255, 255), 1);
              }
    }
  }

  private static void Blend(byte[] bgra, int width, int x, int y, (int R, int G, int B) color, double alpha)
  {
    if (alpha <= 0)
      return;
    int k = 4 * (y * width + x);
    bgra[k] = (byte)Math.Round(bgra[k] * (1 - alpha) + color.B * alpha);
    bgra[k + 1] = (byte)Math.Round(bgra[k + 1] * (1 - alpha) + color.G * alpha);
    bgra[k + 2] = (byte)Math.Round(bgra[k + 2] * (1 - alpha) + color.R * alpha);
  }

  /// <summary>An RGB PNG with the <c>pHYs</c> chunk set to <see cref="Dpi"/>.</summary>
  public static void WritePng(string path, byte[] bgra, int width, int height)
  {
    var raw = new byte[height * (1 + width * 3)];
    for (int y = 0, o = 0; y < height; y++)
    {
      raw[o++] = 0;
      for (int x = 0; x < width; x++, o += 3)
      {
        int k = 4 * (y * width + x);
        (raw[o], raw[o + 1], raw[o + 2]) = (bgra[k + 2], bgra[k + 1], bgra[k]);
      }
    }
    using var z = new MemoryStream();
    using (var zs = new ZLibStream(z, CompressionLevel.Optimal, leaveOpen: true))
      zs.Write(raw);
    var header = new byte[13];
    BinaryPrimitives.WriteInt32BigEndian(header, width);
    BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4), height);
    header[8] = 8; // bit depth
    header[9] = 2; // RGB
    var phys = new byte[9];
    int perMetre = (int)Math.Round(Dpi / 0.0254);
    BinaryPrimitives.WriteInt32BigEndian(phys, perMetre);
    BinaryPrimitives.WriteInt32BigEndian(phys.AsSpan(4), perMetre);
    phys[8] = 1; // unit: metre
    Directory.CreateDirectory(Path.GetDirectoryName(path)!);
    using var f = File.Create(path);
    f.Write([0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    Chunk(f, "IHDR", header);
    Chunk(f, "pHYs", phys);
    Chunk(f, "IDAT", z.ToArray());
    Chunk(f, "IEND", []);
  }

  /// <summary>Reads back a PNG written by <see cref="WritePng"/> (8-bit RGB, filter 0 on every row) as BGRA.</summary>
  public static byte[] ReadPng(string path, out int width, out int height)
  {
    var bytes = File.ReadAllBytes(path);
    width = height = 0;
    using var idat = new MemoryStream();
    for (int o = 8; o + 8 <= bytes.Length;)
    {
      int len = BinaryPrimitives.ReadInt32BigEndian(bytes.AsSpan(o));
      var type = System.Text.Encoding.ASCII.GetString(bytes, o + 4, 4);
      if (type == "IHDR")
      {
        width = BinaryPrimitives.ReadInt32BigEndian(bytes.AsSpan(o + 8));
        height = BinaryPrimitives.ReadInt32BigEndian(bytes.AsSpan(o + 12));
        if (bytes[o + 16] != 8 || bytes[o + 17] != 2)
          throw new InvalidDataException($"{path}: not 8-bit RGB");
      }
      else if (type == "IDAT")
        idat.Write(bytes, o + 8, len);
      o += 12 + len;
    }
    idat.Position = 0;
    using var zs = new ZLibStream(idat, CompressionMode.Decompress);
    using var raw = new MemoryStream();
    zs.CopyTo(raw);
    var r = raw.GetBuffer();
    var bgra = new byte[width * height * 4];
    for (int y = 0, o = 0; y < height; y++)
    {
      if (r[o++] != 0)
        throw new InvalidDataException($"{path}: row {y} uses a PNG filter");
      for (int x = 0; x < width; x++, o += 3)
      {
        int k = 4 * (y * width + x);
        (bgra[k], bgra[k + 1], bgra[k + 2], bgra[k + 3]) = (r[o + 2], r[o + 1], r[o], 255);
      }
    }
    return bgra;
  }

  private static void Chunk(Stream f, string type, byte[] data)
  {
    var be = new byte[4];
    BinaryPrimitives.WriteInt32BigEndian(be, data.Length);
    f.Write(be);
    var typed = new byte[4 + data.Length];
    System.Text.Encoding.ASCII.GetBytes(type).CopyTo(typed, 0);
    data.CopyTo(typed, 4);
    f.Write(typed);
    BinaryPrimitives.WriteUInt32BigEndian(be, System.IO.Hashing.Crc32.HashToUInt32(typed));
    f.Write(be);
  }
}

/// <summary><c>manifest.json</c>: what was captured, with what, and how each shot went.</summary>
internal sealed class Manifest(string appVersion, string commit, string culture, string theme, string monitor, double scaling)
{
  private readonly List<ShotRecord> _shots = [];

  public List<string> Warnings { get; } = [];

  public void Add(ShotRecord record) => _shots.Add(record);

  public IReadOnlyList<ShotRecord> Shots => _shots;

  public void Write(string path) => File.WriteAllText(path, ShotTools.ToJson(new
  {
    app_version = appVersion,
    git_commit = commit,
    date = DateTime.UtcNow.ToString("u"),
    culture,
    theme,
    monitor,
    scaling,
    dpi = ShotTools.Dpi,
    warnings = Warnings,
    shots = _shots,
  }));
}
