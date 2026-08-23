
use super::*;

#[test]
fn test_pathbuf_push_pop() {
  let mut p = PathBuf::new();
  p = p.join("test").join("dir");
  let s = p.to_str_unified().unwrap();
  assert!(s.ends_with("test/dir") || s.ends_with("test\\dir"));

  p.pop();
  let s2 = p.to_str_unified().unwrap();
  assert!(s2.ends_with("test"));
}

#[test]
fn test_pathbuf_extension() {
  let p = PathBuf::from("file.txt");
  assert_eq!(p.extension().as_deref(), Some("txt"));

  let p2 = PathBuf::from("file_no_ext");
  assert_eq!(p2.extension(), None);
}

#[test]
fn test_pathbuf_join() {
  let p1 = PathBuf::from("test");
  let p2 = p1.join("dir");
  let s = p2.to_str_unified().unwrap();
  assert!(s.ends_with("test/dir") || s.ends_with("test\\dir"));
}
