// Source-linked into aethervk.ui.e2e (Windows), aethervk.ui.e2e.mac and aethervk.ui.e2e.shared.tests.
using System;
using System.Buffers.Binary;
using System.IO;
using System.IO.Compression;
using System.Text;

namespace AetherVk.UI.E2E.Shared;

/// <summary>An 8-bit RGBA image, rows from the top.</summary>
public sealed record PngImage(int Width, int Height, byte[] Rgba)
{
  /// <summary>Rec. 709 luminance per pixel, 0..1 (as <see cref="FrameMetricsCore.Luma"/>).</summary>
  public double[] Luminance()
  {
    var lum = new double[Width * Height];
    for (int k = 0; k < lum.Length; k++)
      lum[k] = FrameMetricsCore.Luma(Rgba[4 * k], Rgba[4 * k + 1], Rgba[4 * k + 2]);
    return lum;
  }
}

/// <summary>
/// A PNG reader and writer without an image library: the app's exports, the Mac2 element
/// screenshots and <c>screencapture</c> files are read with it (any filter, 8/16-bit grey, RGB,
/// palette, grey+alpha, RGBA; not interlaced), the probes' crops are written with it.
/// </summary>
public static class PngCodec
{
  private static readonly byte[] Signature = [0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A];

  public static PngImage Read(string path) => Decode(File.ReadAllBytes(path));

  public static PngImage Decode(byte[] bytes)
  {
    if (bytes.Length < 8 || !bytes.AsSpan(0, 8).SequenceEqual(Signature))
      throw new InvalidDataException("not a PNG");
    int width = 0, height = 0, depth = 0, colour = 0, interlace = 0;
    byte[]? palette = null, paletteAlpha = null;
    using var idat = new MemoryStream();
    for (int o = 8; o + 8 <= bytes.Length;)
    {
      int len = BinaryPrimitives.ReadInt32BigEndian(bytes.AsSpan(o));
      var type = Encoding.ASCII.GetString(bytes, o + 4, 4);
      var data = bytes.AsSpan(o + 8, len);
      switch (type)
      {
        case "IHDR":
          width = BinaryPrimitives.ReadInt32BigEndian(data);
          height = BinaryPrimitives.ReadInt32BigEndian(data[4..]);
          depth = data[8];
          colour = data[9];
          interlace = data[12];
          break;
        case "PLTE":
          palette = data.ToArray();
          break;
        case "tRNS":
          paletteAlpha = data.ToArray();
          break;
        case "IDAT":
          idat.Write(data);
          break;
      }
      if (type == "IEND")
        break;
      o += 12 + len;
    }
    if (width <= 0 || height <= 0)
      throw new InvalidDataException("PNG without IHDR");
    if (interlace != 0)
      throw new InvalidDataException("interlaced PNG");
    int channels = colour switch
    {
      0 => 1,
      2 => 3,
      3 => 1,
      4 => 2,
      6 => 4,
      _ => throw new InvalidDataException($"PNG colour type {colour}"),
    };
    if (depth != 8 && depth != 16)
      throw new InvalidDataException($"PNG bit depth {depth}");
    if (colour == 3 && depth != 8)
      throw new InvalidDataException("palette PNG below 8 bits");
    int bpp = channels * depth / 8;
    int stride = width * bpp;

    idat.Position = 0;
    using var zs = new ZLibStream(idat, CompressionMode.Decompress);
    using var rawStream = new MemoryStream();
    zs.CopyTo(rawStream);
    var raw = rawStream.GetBuffer();
    if (rawStream.Length < (long)height * (stride + 1))
      throw new InvalidDataException("PNG data too short");

    var cur = new byte[stride];
    var prev = new byte[stride];
    var rgba = new byte[width * height * 4];
    for (int y = 0; y < height; y++)
    {
      int row = y * (stride + 1);
      byte filter = raw[row];
      for (int i = 0; i < stride; i++)
      {
        int x = raw[row + 1 + i];
        int a = i >= bpp ? cur[i - bpp] : 0;
        int b = prev[i];
        int c = i >= bpp ? prev[i - bpp] : 0;
        cur[i] = filter switch
        {
          0 => (byte)x,
          1 => (byte)(x + a),
          2 => (byte)(x + b),
          3 => (byte)(x + (a + b) / 2),
          4 => (byte)(x + Paeth(a, b, c)),
          _ => throw new InvalidDataException($"PNG filter {filter} on row {y}"),
        };
      }
      for (int px = 0; px < width; px++)
      {
        int k = 4 * (y * width + px);
        // the high byte of a 16-bit sample is its 8-bit value
        byte S(int ch) => cur[px * bpp + ch * (depth / 8)];
        switch (colour)
        {
          case 0:
            rgba[k] = rgba[k + 1] = rgba[k + 2] = S(0);
            rgba[k + 3] = 255;
            break;
          case 2:
            (rgba[k], rgba[k + 1], rgba[k + 2], rgba[k + 3]) = (S(0), S(1), S(2), 255);
            break;
          case 3:
            int idx = cur[px];
            if (palette is null || 3 * idx + 2 >= palette.Length)
              throw new InvalidDataException("palette index out of range");
            (rgba[k], rgba[k + 1], rgba[k + 2]) = (palette[3 * idx], palette[3 * idx + 1], palette[3 * idx + 2]);
            rgba[k + 3] = paletteAlpha is not null && idx < paletteAlpha.Length ? paletteAlpha[idx] : (byte)255;
            break;
          case 4:
            rgba[k] = rgba[k + 1] = rgba[k + 2] = S(0);
            rgba[k + 3] = S(1);
            break;
          case 6:
            (rgba[k], rgba[k + 1], rgba[k + 2], rgba[k + 3]) = (S(0), S(1), S(2), S(3));
            break;
        }
      }
      (cur, prev) = (prev, cur);
    }
    return new PngImage(width, height, rgba);
  }

  private static int Paeth(int a, int b, int c)
  {
    int p = a + b - c;
    int pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
    return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
  }

  /// <summary>Writes RGBA bytes as an 8-bit RGB PNG (filter 0 on every row; alpha dropped).</summary>
  public static void WriteRgb(string path, byte[] rgba, int width, int height)
  {
    var dir = Path.GetDirectoryName(Path.GetFullPath(path));
    if (!string.IsNullOrEmpty(dir))
      Directory.CreateDirectory(dir);
    File.WriteAllBytes(path, EncodeRgb(rgba, width, height));
  }

  public static byte[] EncodeRgb(byte[] rgba, int width, int height)
  {
    var raw = new byte[height * (1 + width * 3)];
    for (int y = 0, o = 0; y < height; y++)
    {
      raw[o++] = 0;
      for (int x = 0; x < width; x++, o += 3)
      {
        int k = 4 * (y * width + x);
        (raw[o], raw[o + 1], raw[o + 2]) = (rgba[k], rgba[k + 1], rgba[k + 2]);
      }
    }
    var header = new byte[13];
    BinaryPrimitives.WriteInt32BigEndian(header, width);
    BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4), height);
    header[8] = 8; // bit depth
    header[9] = 2; // colour type RGB
    return Assemble(header, raw);
  }

  /// <summary>A PNG from an IHDR body and the filtered scanlines (deflated here).</summary>
  public static byte[] Assemble(byte[] ihdr, byte[] filteredRows)
  {
    using var z = new MemoryStream();
    using (var zs = new ZLibStream(z, CompressionLevel.Fastest, leaveOpen: true))
      zs.Write(filteredRows);
    using var f = new MemoryStream();
    f.Write(Signature);
    WriteChunk(f, "IHDR", ihdr);
    WriteChunk(f, "IDAT", z.ToArray());
    WriteChunk(f, "IEND", []);
    return f.ToArray();
  }

  private static void WriteChunk(Stream f, string type, byte[] data)
  {
    var be = new byte[4];
    BinaryPrimitives.WriteInt32BigEndian(be, data.Length);
    f.Write(be);
    var typed = new byte[4 + data.Length];
    Encoding.ASCII.GetBytes(type).CopyTo(typed, 0);
    data.CopyTo(typed, 4);
    f.Write(typed);
    BinaryPrimitives.WriteUInt32BigEndian(be, Crc32(typed));
    f.Write(be);
  }

  private static readonly uint[] CrcTable = BuildCrcTable();

  private static uint[] BuildCrcTable()
  {
    var t = new uint[256];
    for (uint n = 0; n < 256; n++)
    {
      uint c = n;
      for (int k = 0; k < 8; k++)
        c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      t[n] = c;
    }
    return t;
  }

  /// <summary>The PNG (ISO 3309) CRC-32.</summary>
  public static uint Crc32(ReadOnlySpan<byte> data)
  {
    uint c = 0xFFFFFFFFu;
    foreach (var b in data)
      c = CrcTable[(c ^ b) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
  }
}
