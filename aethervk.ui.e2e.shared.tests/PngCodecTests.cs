using System.Buffers.Binary;
using AetherVk.UI.E2E.Shared;

namespace AetherVk.UI.E2E.Shared.Tests;

public sealed class PngCodecTests
{
  private static byte[] Ihdr(int w, int h, byte depth, byte colour)
  {
    var ihdr = new byte[13];
    BinaryPrimitives.WriteInt32BigEndian(ihdr, w);
    BinaryPrimitives.WriteInt32BigEndian(ihdr.AsSpan(4), h);
    ihdr[8] = depth;
    ihdr[9] = colour;
    return ihdr;
  }

  private static int Paeth(int a, int b, int c)
  {
    int p = a + b - c;
    int pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
    return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
  }

  /// <summary>Scanlines filtered the way an encoder would: row y with filter <paramref name="filterOf"/>(y).</summary>
  private static byte[] Filter(byte[] pixels, int stride, int height, int bpp, Func<int, int> filterOf)
  {
    var outRows = new byte[height * (stride + 1)];
    for (int y = 0; y < height; y++)
    {
      int f = filterOf(y);
      outRows[y * (stride + 1)] = (byte)f;
      for (int i = 0; i < stride; i++)
      {
        int x = pixels[y * stride + i];
        int a = i >= bpp ? pixels[y * stride + i - bpp] : 0;
        int b = y > 0 ? pixels[(y - 1) * stride + i] : 0;
        int c = y > 0 && i >= bpp ? pixels[(y - 1) * stride + i - bpp] : 0;
        int pred = f switch { 0 => 0, 1 => a, 2 => b, 3 => (a + b) / 2, _ => Paeth(a, b, c) };
        outRows[y * (stride + 1) + 1 + i] = (byte)(x - pred);
      }
    }
    return outRows;
  }

  private static byte[] RandomBytes(int n, int seed)
  {
    var r = new Random(seed);
    var b = new byte[n];
    r.NextBytes(b);
    return b;
  }

  [Fact]
  public void EncodeRgb_ThenDecode_KeepsTheColours()
  {
    const int w = 7, h = 5;
    var rgba = RandomBytes(w * h * 4, 1);
    var png = PngCodec.Decode(PngCodec.EncodeRgb(rgba, w, h));
    Assert.Equal((w, h), (png.Width, png.Height));
    for (int k = 0; k < w * h; k++)
    {
      Assert.Equal(rgba[4 * k], png.Rgba[4 * k]);
      Assert.Equal(rgba[4 * k + 1], png.Rgba[4 * k + 1]);
      Assert.Equal(rgba[4 * k + 2], png.Rgba[4 * k + 2]);
      Assert.Equal(255, png.Rgba[4 * k + 3]);
    }
  }

  [Fact]
  public void Decode_Rgba_WithEveryFilter()
  {
    const int w = 6, h = 10, bpp = 4;
    var rgba = RandomBytes(w * h * bpp, 2);
    var bytes = PngCodec.Assemble(Ihdr(w, h, 8, 6), Filter(rgba, w * bpp, h, bpp, y => y % 5));
    Assert.Equal(rgba, PngCodec.Decode(bytes).Rgba);
  }

  [Fact]
  public void Decode_Rgb_WithEveryFilter()
  {
    const int w = 5, h = 5, bpp = 3;
    var rgb = RandomBytes(w * h * bpp, 3);
    var png = PngCodec.Decode(PngCodec.Assemble(Ihdr(w, h, 8, 2), Filter(rgb, w * bpp, h, bpp, y => 4 - y)));
    for (int k = 0; k < w * h; k++)
      Assert.Equal((rgb[3 * k], rgb[3 * k + 1], rgb[3 * k + 2], (byte)255),
        (png.Rgba[4 * k], png.Rgba[4 * k + 1], png.Rgba[4 * k + 2], png.Rgba[4 * k + 3]));
  }

  [Fact]
  public void Decode_SixteenBitGrey_TakesTheHighByte()
  {
    const int w = 3, h = 2;
    var samples = new byte[w * h * 2];
    for (int k = 0; k < w * h; k++)
      (samples[2 * k], samples[2 * k + 1]) = ((byte)(40 * k), (byte)0xAB);
    var png = PngCodec.Decode(PngCodec.Assemble(Ihdr(w, h, 16, 0), Filter(samples, w * 2, h, 2, _ => 1)));
    for (int k = 0; k < w * h; k++)
      Assert.Equal(((byte)(40 * k), (byte)(40 * k), (byte)(40 * k), (byte)255),
        (png.Rgba[4 * k], png.Rgba[4 * k + 1], png.Rgba[4 * k + 2], png.Rgba[4 * k + 3]));
  }

  [Fact]
  public void Decode_Palette_WithTransparency()
  {
    // a hand-made file: IHDR, PLTE (red, green), tRNS (red half transparent), IDAT
    using var f = new MemoryStream();
    var body = PngCodec.Assemble(Ihdr(2, 1, 8, 3), [0, 0, 1]);
    // splice PLTE and tRNS after IHDR (8 signature + 25 IHDR chunk)
    f.Write(body, 0, 33);
    WriteChunk(f, "PLTE", [255, 0, 0, 0, 255, 0]);
    WriteChunk(f, "tRNS", [128]);
    f.Write(body, 33, body.Length - 33);
    var png = PngCodec.Decode(f.ToArray());
    Assert.Equal(new byte[] { 255, 0, 0, 128, 0, 255, 0, 255 }, png.Rgba);
  }

  private static void WriteChunk(Stream s, string type, byte[] data)
  {
    var be = new byte[4];
    BinaryPrimitives.WriteInt32BigEndian(be, data.Length);
    s.Write(be);
    var typed = System.Text.Encoding.ASCII.GetBytes(type).Concat(data).ToArray();
    s.Write(typed);
    BinaryPrimitives.WriteUInt32BigEndian(be, PngCodec.Crc32(typed));
    s.Write(be);
  }

  [Fact]
  public void Crc32_IsThePngCrc() =>
    // the CRC every PNG ends with: IEND with no data
    Assert.Equal(0xAE426082u, PngCodec.Crc32("IEND"u8));

  [Fact]
  public void Decode_RejectsWhatIsNotAPng() =>
    Assert.Throws<InvalidDataException>(() => PngCodec.Decode([1, 2, 3, 4, 5, 6, 7, 8, 9]));

  [Fact]
  public void Luminance_IsRec709PerPixel()
  {
    var png = new PngImage(3, 1, [255, 255, 255, 255, 0, 0, 0, 255, 0, 255, 0, 255]);
    var lum = png.Luminance();
    Assert.Equal(1.0, lum[0], 12);
    Assert.Equal(0.0, lum[1], 12);
    Assert.Equal(0.7152, lum[2], 12);
  }

  [Fact]
  public void WriteRgb_WritesAFileThatReadsBack()
  {
    var dir = Path.Combine(Path.GetTempPath(), "aethervk-png-" + Guid.NewGuid().ToString("N"));
    try
    {
      var path = Path.Combine(dir, "sub", "a.png");
      var rgba = RandomBytes(4 * 4 * 4, 4);
      PngCodec.WriteRgb(path, rgba, 4, 4);
      var png = PngCodec.Read(path);
      Assert.Equal((4, 4), (png.Width, png.Height));
      Assert.Equal(rgba[0], png.Rgba[0]);
    }
    finally
    {
      if (Directory.Exists(dir))
        Directory.Delete(dir, recursive: true);
    }
  }
}
