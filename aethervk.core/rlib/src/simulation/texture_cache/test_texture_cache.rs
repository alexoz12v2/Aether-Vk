use super::*;
use std::fs;

#[test]
fn test_texture_cache_lifecycle() {
  let mut cache = TextureCache::new("test_aethervk_tex_cache_1");
  let png_data = fs::read("../../test_assets/rain.png").unwrap();
  let id = cache.decode_and_insert("../../test_assets/rain.png", &png_data).unwrap();

  let (w, h, fmt, bytes) = cache.get(id).unwrap();
  assert!(w > 0);
  assert!(h > 0);
  assert_eq!(fmt.to_u32(), TexelFormat::R8G8B8A8_UNORM.to_u32());
  assert!(!bytes.is_empty());

  cache.remove(id);
  assert!(cache.get(id).is_none());

  // Clean up
  let _ = fs::remove_file(cache.path_str.clone());
}

#[test]
fn test_texture_cache_multiple_inserts_and_defragment() {
  let mut cache = TextureCache::new("test_aethervk_tex_cache_2");
  let rain_data = fs::read("../../test_assets/rain.png").unwrap();
  let cloud_data = fs::read("../../test_assets/cloud.png").unwrap();
  let sun_data = fs::read("../../test_assets/sun.png").unwrap();

  let id1 = cache.decode_and_insert("../../test_assets/rain.png", &rain_data).unwrap();
  let id2 = cache.decode_and_insert("../../test_assets/cloud.png", &cloud_data).unwrap();
  let id3 = cache.decode_and_insert("../../test_assets/sun.png", &sun_data).unwrap();

  assert!(cache.get(id1).is_some());
  assert!(cache.get(id2).is_some());
  assert!(cache.get(id3).is_some());

  // Remove one to create a hole
  cache.remove(id2);
  assert!(cache.get(id2).is_none());

  // Defragment should remove the hole
  cache.defragment();

  assert!(cache.get(id1).is_some());
  assert!(cache.get(id2).is_none());
  assert!(cache.get(id3).is_some());

  let _ = fs::remove_file(cache.path_str.clone());
}

#[test]
fn test_texture_cache_reopen() {
  let path_str;
  let id;
  {
    let mut cache = TextureCache::new("test_aethervk_tex_cache_3");
    let sun_data = fs::read("../../test_assets/sun.png").unwrap();
    id = cache.decode_and_insert("../../test_assets/sun.png", &sun_data).unwrap();
    path_str = cache.path_str.clone();
  }

  // Reopen cache
  let cache2 = TextureCache::new("test_aethervk_tex_cache_3");
  let res = cache2.get(id);
  assert!(res.is_some());

  let _ = fs::remove_file(path_str);
}

/// Grey and RGB PNGs (written by the test) are tagged RGBA8: `png_decoder` expands every colour
/// type to RGBA, so the format follows the decoded data (they were tagged `R8_UNORM`, a quarter of
/// the bytes the data holds).
#[test]
fn test_texture_cache_grey_and_rgb_png_are_rgba8() {
  let dir = std::env::temp_dir().join(format!("avk_tex_cache_png_{}", std::process::id()));
  fs::create_dir_all(&dir).unwrap();
  let grey_path = dir.join("grey.png");
  image::GrayImage::from_fn(5, 3, |x, y| image::Luma([(10 * x + 60 * y) as u8]))
    .save(&grey_path)
    .unwrap();
  let rgb_path = dir.join("rgb.png");
  image::RgbImage::from_fn(4, 2, |x, y| image::Rgb([x as u8, y as u8, 200]))
    .save(&rgb_path)
    .unwrap();

  let mut cache = TextureCache::new("test_aethervk_tex_cache_png_formats");
  for (path, w, h) in [(&grey_path, 5u32, 3u32), (&rgb_path, 4, 2)] {
    let p = path.to_str().unwrap();
    let id = cache.decode_and_insert(p, &fs::read(path).unwrap()).unwrap();
    let (gw, gh, fmt, bytes) = cache.get(id).unwrap();
    assert_eq!((gw, gh), (w, h));
    assert_eq!(fmt.to_u32(), TexelFormat::R8G8B8A8_UNORM.to_u32(), "{p}");
    assert_eq!(bytes.len() as u64, fmt.byte_size(w, h));
    assert_eq!(bytes.len(), (w * h * 4) as usize);
    cache.remove(id);
  }
  // texel contents: grey replicated with opaque alpha, RGB with opaque alpha
  let id = cache
    .decode_and_insert(grey_path.to_str().unwrap(), &fs::read(&grey_path).unwrap())
    .unwrap();
  let (_, _, _, bytes) = cache.get(id).unwrap();
  let px = |x: usize, y: usize| &bytes[(y * 5 + x) * 4..(y * 5 + x) * 4 + 4];
  assert_eq!(px(2, 1), &[80, 80, 80, 255]);
  cache.remove(id);
  let id = cache
    .decode_and_insert(rgb_path.to_str().unwrap(), &fs::read(&rgb_path).unwrap())
    .unwrap();
  let (_, _, _, bytes) = cache.get(id).unwrap();
  assert_eq!(&bytes[(4 + 3) * 4..(4 + 3) * 4 + 4], &[3, 1, 200, 255]);
  cache.remove(id);
  let _ = fs::remove_file(cache.path_str.clone());
  let _ = fs::remove_dir_all(&dir);
}
