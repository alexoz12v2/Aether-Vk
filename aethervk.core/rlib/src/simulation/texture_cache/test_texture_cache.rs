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
