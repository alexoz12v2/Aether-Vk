//! Resolution order, prefix matching, the mirror path, hash/size mismatches, one extraction for
//! racing threads, offline downloads and the REQUIRED panic, on a fake repository in a temp dir.

use super::*;
use std::sync::{Barrier, Mutex, MutexGuard, atomic::Ordering};

/// The extraction counter and the environment are process-global: one test at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
  SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

struct Fake {
  root: PathBuf,
  repo: PathBuf,
  mirror: PathBuf,
  cache: PathBuf,
  sha: String,
  size: u64,
}

impl Drop for Fake {
  fn drop(&mut self) {
    let _ = fs::remove_dir_all(&self.root);
  }
}

/// A repository with `test_assets/ASSETS.tsv` listing one group `saves` (paths `saves/a`,
/// `saves/b.avk`, `top.bin`) whose tarball sits in `mirror/`.
fn fake(name: &str) -> Fake {
  let root = std::env::temp_dir().join(format!("aethervk-test-assets-{}-{name}", std::process::id()));
  let _ = fs::remove_dir_all(&root);
  let (repo, mirror, cache, staging) = (root.join("repo"), root.join("mirror"), root.join("cache"), root.join("staging"));
  for d in [repo.join("test_assets"), mirror.clone(), staging.join("saves/a/sub")] {
    fs::create_dir_all(d).unwrap();
  }
  fs::write(staging.join("saves/a/x.txt"), "x").unwrap();
  fs::write(staging.join("saves/a/sub/y.txt"), "y").unwrap();
  fs::write(staging.join("saves/b.avk"), "b").unwrap();
  fs::write(staging.join("top.bin"), "top").unwrap();
  let status = Command::new("tar")
    .arg("-czf")
    .arg(mirror.join("saves.tar.gz"))
    .arg("-C")
    .arg(&staging)
    .args(["saves", "top.bin"])
    .status()
    .unwrap();
  assert!(status.success());
  let bytes = fs::read(mirror.join("saves.tar.gz")).unwrap();
  let f = Fake { root, repo, mirror, cache, sha: hex(&sha256(&bytes)), size: bytes.len() as u64 };
  f.write_manifest(&f.sha, f.size);
  f
}

impl Fake {
  fn write_manifest(&self, sha: &str, size: u64) {
    let text = format!(
      "# a comment\n# base http://127.0.0.1:9/\n# group\tpaths\tfile\tsha256\tsize\nsaves\tsaves/a,saves/b.avk,top.bin\tsaves.tar.gz\t{sha}\t{size}\n"
    );
    fs::write(self.repo.join(MANIFEST), text).unwrap();
  }

  fn cfg(&self) -> Config {
    Config { repo: self.repo.clone(), cache: self.cache.clone(), mirror: Some(self.mirror.clone()), base: None }
  }

  fn group_dir(&self) -> PathBuf {
    self.cache.join(format!("saves-{}", &self.sha[..12]))
  }
}

fn extractions() -> usize {
  EXTRACTIONS.load(Ordering::SeqCst)
}

#[test]
fn sha256_known_vectors() {
  assert_eq!(hex(&sha256(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  assert_eq!(hex(&sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  let long = vec![b'a'; 1000];
  assert_eq!(hex(&sha256(&long)), "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
}

#[test]
fn manifest_parses_base_rows_and_rejects_short_rows() {
  let m = parse_manifest("# c\n# base https://h/x/\n\ng\ta,b/c/\tg.tar.gz\tABCDEF\t12\r\n").unwrap();
  assert_eq!(m.base.as_deref(), Some("https://h/x/"));
  assert_eq!(
    m.groups,
    vec![Group { name: "g".into(), paths: vec!["a".into(), "b/c".into()], file: "g.tar.gz".into(), sha256: "abcdef".into(), size: 12 }]
  );
  assert!(parse_manifest("g\ta\tg.tar.gz\tabc\n").unwrap_err().contains("4 columns"));
  assert!(parse_manifest("g\ta\tg.tar.gz\tabc\tbig\n").unwrap_err().contains("size"));
}

#[test]
fn prefix_matching_is_whole_path_components() {
  let m = parse_manifest("s\tsaves/a,saves/b.avk\ts.tar.gz\t00\t1\nf\tfits_corpus/wcs_reference\tf.tar.gz\t00\t1\n").unwrap();
  let g = |rel: &str| m.group_of(rel).map(|g| g.name.as_str());
  assert_eq!(g("saves/a"), Some("s"));
  assert_eq!(g("saves/a/x.txt"), Some("s"));
  assert_eq!(g("saves/a/sub/y.txt"), Some("s"));
  assert_eq!(g("saves/b.avk"), Some("s"));
  assert_eq!(g("fits_corpus/wcs_reference/tan.tsv"), Some("f"));
  assert_eq!(g("saves/ab"), None, "a sibling sharing the prefix is not in the group");
  assert_eq!(g("saves/b.avk2"), None);
  assert_eq!(g("saves"), None, "a parent of an entry is not in the group");
  assert_eq!(g("fits_corpus/manifest.tsv"), None);
}

#[test]
fn local_file_wins_over_the_manifest() {
  let _s = serial();
  let f = fake("local");
  let local = f.repo.join("test_assets/saves/a/x.txt");
  fs::create_dir_all(local.parent().unwrap()).unwrap();
  fs::write(&local, "local").unwrap();
  let before = extractions();
  assert_eq!(resolve(&f.cfg(), "saves/a/x.txt", Kind::Any).unwrap(), local);
  assert_eq!(resolve(&f.cfg(), "saves/a", Kind::Dir).unwrap(), f.repo.join("test_assets/saves/a"));
  assert_eq!(extractions(), before);
  assert!(!f.cache.exists(), "nothing fetched");
  // a file the local copy lacks comes from the group
  assert_eq!(fs::read_to_string(resolve(&f.cfg(), "saves/b.avk", Kind::Any).unwrap()).unwrap(), "b");
}

#[test]
fn mirror_fills_the_group_once() {
  let _s = serial();
  let f = fake("mirror");
  let before = extractions();
  let x = resolve(&f.cfg(), "saves/a/x.txt", Kind::Any).unwrap();
  assert_eq!(x, f.group_dir().join("saves/a/x.txt"));
  assert_eq!(fs::read_to_string(&x).unwrap(), "x");
  assert!(f.group_dir().join(".complete").is_file());
  assert_eq!(extractions(), before + 1);
  assert_eq!(fs::read_to_string(resolve(&f.cfg(), "saves/b.avk", Kind::Any).unwrap()).unwrap(), "b");
  assert_eq!(fs::read_to_string(resolve(&f.cfg(), "top.bin", Kind::Any).unwrap()).unwrap(), "top");
  assert_eq!(resolve(&f.cfg(), "saves/a/", Kind::Dir).unwrap(), f.group_dir().join("saves/a"));
  assert!(resolve(&f.cfg(), "saves/b.avk", Kind::Dir).unwrap_err().contains("not a directory"));
  assert_eq!(resolve(&f.cfg(), "elsewhere/z.bin", Kind::Any).unwrap_err(), "not in ASSETS.tsv");
  assert!(resolve(&f.cfg(), "saves/a/missing.txt", Kind::Any).unwrap_err().contains("not in the saves tarball"));
  assert_eq!(extractions(), before + 1, "a complete group is never fetched again");
  // nothing left behind but the group, its lock
  let mut names: Vec<String> = fs::read_dir(&f.cache).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
  names.sort();
  assert_eq!(names, vec![format!("saves-{}", &f.sha[..12]), "saves.lock".to_string()]);
}

#[test]
fn hash_or_size_mismatch_is_an_error() {
  let _s = serial();
  let f = fake("mismatch");
  f.write_manifest(&"0".repeat(64), f.size);
  let e = resolve(&f.cfg(), "saves/a/x.txt", Kind::Any).unwrap_err();
  assert!(e.contains("sha256") && e.contains(&f.sha), "{e}");
  f.write_manifest(&f.sha, f.size + 1);
  let e = resolve(&f.cfg(), "saves/a/x.txt", Kind::Any).unwrap_err();
  assert!(e.contains("bytes"), "{e}");
  assert!(!f.group_dir().exists(), "nothing extracted");
  // a missing tarball in the mirror
  f.write_manifest(&f.sha, f.size);
  fs::remove_file(f.mirror.join("saves.tar.gz")).unwrap();
  assert!(resolve(&f.cfg(), "saves/a/x.txt", Kind::Any).unwrap_err().contains("mirror"));
}

#[test]
fn racing_threads_extract_once() {
  let _s = serial();
  let f = fake("race");
  let before = extractions();
  let barrier = Barrier::new(4);
  let cfg = f.cfg();
  std::thread::scope(|s| {
    let handles: Vec<_> = ["saves/a/x.txt", "saves/b.avk", "top.bin", "saves/a/sub/y.txt"]
      .into_iter()
      .map(|rel| {
        let (barrier, cfg) = (&barrier, &cfg);
        s.spawn(move || {
          barrier.wait();
          resolve(cfg, rel, Kind::Any)
        })
      })
      .collect();
    for h in handles {
      assert!(h.join().unwrap().unwrap().is_file());
    }
  });
  assert_eq!(extractions(), before + 1);
}

#[test]
fn unreachable_download_is_an_error() {
  let _s = serial();
  let f = fake("offline");
  let cfg = Config { mirror: None, base: Some("http://127.0.0.1:9".into()), ..f.cfg() };
  let e = resolve(&cfg, "saves/a/x.txt", Kind::Any).unwrap_err();
  assert!(e.contains("download http://127.0.0.1:9/saves.tar.gz"), "{e}");
  assert!(!f.group_dir().exists());
  // the failure is remembered: a second test of the same run does not wait for the network again
  assert_eq!(resolve(&cfg, "top.bin", Kind::Any).unwrap_err(), e);
}

#[test]
#[should_panic(expected = "AETHERVK_TEST_ASSETS_REQUIRED=1")]
fn required_turns_a_failure_into_a_panic() {
  let _ = finish(true, "saves/a/x.txt", Err("offline".into()));
}

#[test]
fn not_required_passes_the_failure_through() {
  assert_eq!(finish(false, "x", Err("offline".into())), Err("offline".to_string()));
}

#[test]
fn environment_overrides() {
  let _s = serial();
  let f = fake("env");
  let keys = ["AETHERVK_TEST_ASSETS_CACHE", "AETHERVK_TEST_ASSETS_MIRROR", "AETHERVK_TEST_ASSETS_BASE", "AETHERVK_TEST_ASSETS_REQUIRED"];
  let saved: Vec<_> = keys.iter().map(|k| std::env::var_os(k)).collect();
  // SAFETY: the tests that read the environment hold SERIAL; nothing else here reads these keys
  unsafe {
    for k in keys {
      std::env::remove_var(k);
    }
  }
  let c = Config::from_env_at(f.repo.clone());
  assert_eq!((c.cache.clone(), c.mirror.clone(), c.base.clone()), (f.repo.join("target/test-assets"), None, None));
  assert!(!required_from_env());
  unsafe {
    std::env::set_var("AETHERVK_TEST_ASSETS_CACHE", &f.cache);
    std::env::set_var("AETHERVK_TEST_ASSETS_MIRROR", &f.mirror);
    std::env::set_var("AETHERVK_TEST_ASSETS_BASE", "http://127.0.0.1:9/");
    std::env::set_var("AETHERVK_TEST_ASSETS_REQUIRED", "1");
  }
  let c = Config::from_env_at(f.repo.clone());
  assert_eq!((c.cache.clone(), c.mirror.clone(), c.base.as_deref()), (f.cache.clone(), Some(f.mirror.clone()), Some("http://127.0.0.1:9/")));
  assert!(required_from_env());
  // the mirror wins over the base
  assert!(resolve(&c, "top.bin", Kind::Any).is_ok());
  unsafe {
    for (k, v) in keys.iter().zip(saved) {
      match v {
        Some(v) => std::env::set_var(k, v),
        None => std::env::remove_var(k),
      }
    }
  }
}

#[test]
fn repo_root_holds_the_manifest() {
  let root = repo_root().unwrap();
  assert!(root.join(MANIFEST).is_file());
  assert_eq!(source_root().unwrap(), root.join("test_assets"));
}
