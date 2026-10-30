//! UTC ↔ TDB conversion with a leap-second table (the engine's one conversion point).
//!
//! Engine epochs are `(centuries, ns)` since J2000 **TDB** (`Epoch::from_tdb_duration`). A
//! calendar date typed or read from a telescope image is **UTC**, 32.184 s + ΔAT behind TDB
//! (69.184 s since 2017, ±1.7 ms periodic). The UI has no time scales: it calls
//! [`utc_to_tdb`] / [`tdb_to_utc`] through the FFI (`avkTime_utcToTdb`, `avkTime_tdbToUtc`).
//!
//! **Leap-second source** ([`LeapTable`]), in order of preference:
//! 1. the newest `naif00NN.tls` NAIF leap-second kernel in `<assets>/planets`
//!    ([`init_from_asset_dir`], mirrors the `pck00011` startup load), or a file loaded at run time
//!    ([`load_file`], the UI's NAIF download cache);
//! 2. hifitime's compile-time table ([`LeapTable::built_in`]), the fallback when the file is
//!    missing or garbled.
//!
//! A file replaces the active table only when it is **consistent** (the active table's entries
//! are its prefix, same dates and values) and **newer** (more entries); the choice is logged.
//! hifitime's own LSK reader (`LeapSecondsFile`) reads IERS lists, not NAIF kernels, and needs
//! the `std` feature, so the `DELTET/DELTA_AT` block is parsed here (bounded, numbers only).
//!
//! **Validity.** An LSK only knows the leap seconds announced so far (IERS Bulletin C gives about
//! six months' notice), so a UTC instant later than `max(last entry, now) + 6 months` converts
//! with [`UtcStatus::BeyondLskValidity`] (a warning: the value is returned as if no further leap
//! second were inserted). `now` is the wall clock the host passes with [`set_now`]; until then the
//! horizon is the last entry + 6 months.
//!
//! **Representation.** An entry's date is the UTC instant read as a naive calendar (an `Epoch`
//! in TAI with the UTC digits, the "naive" epoch). TAI = naive + ΔAT, with the ΔAT of the last
//! entry whose date ≤ the naive instant; a leap second (second = 60) is the naive instant of the
//! next entry's midnight with the previous ΔAT. TAI → TDB is hifitime's (the 32.184 s plus the
//! ~1.66 ms periodic term), the same conversion the engine already uses for ephemerides.

use alloc::{format, string::String, vec::Vec};
use hifitime::{Duration, Epoch, Unit};

/// upper bound on an LSK file read at run time (the NAIF kernel is ~5 KB)
pub const MAX_LSK_BYTES: usize = 64 * 1024;
/// IERS Bulletin C horizon: a leap second is announced at least this long before it happens
const VALIDITY_MARGIN_DAYS: f64 = 183.0;

/// Status of a UTC ↔ TDB conversion (the FFI's `status` value).
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UtcStatus {
  /// converted with a table entry
  Ok = 0,
  /// before the first table entry (1972-01-01): UTC has no leap-second definition; the value is
  /// computed with the first entry's ΔAT (10 s), a nominal extrapolation, not a UTC
  BeforeLeapSeconds = 1,
  /// after the table's validity horizon (see the module doc): converted as if no further leap
  /// second were inserted
  BeyondLskValidity = 2,
  /// invalid calendar fields (or second = 60 where the table has no leap second); no value
  Malformed = 3,
}

/// Result of [`load_file`] / [`LeapTable::adopt`] (the FFI's `avkTime_loadLeapSecondsFile` return).
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadStatus {
  /// the file is the active table now
  Adopted = 0,
  /// the file is consistent but not newer than the active table, which is kept
  KeptCurrent = 1,
  /// the file could not be read, is empty or exceeds [`MAX_LSK_BYTES`]
  Unreadable = 2,
  /// no well-formed `DELTET/DELTA_AT` block
  Malformed = 3,
  /// the file drops or changes an entry of the active table; the active table is kept
  Inconsistent = 4,
}

/// Calendar fields of a UTC instant (`second` may be 60 on a leap second).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct UtcFields {
  pub year: i32,
  pub month: u8,
  pub day: u8,
  pub hour: u8,
  pub minute: u8,
  pub second: u8,
  pub nanos: u32,
}

/// One leap-second table entry: from the UTC instant `utc` (naive calendar epoch), TAI − UTC = `delta_at`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeapEntry {
  pub utc: Epoch,
  pub delta_at: i32,
}

impl LeapEntry {
  /// `(year, month, day)` of the entry's UTC date
  pub fn ymd(&self) -> (i32, u8, u8) {
    let (y, m, d, _, _, _, _) = self.utc.to_gregorian_tai();
    (y, m, d)
  }
}

/// Where the active table came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeapSource {
  /// hifitime's compile-time table
  BuiltIn,
  /// a parsed NAIF LSK (the path or name it was loaded from)
  File(String),
}

impl core::fmt::Display for LeapSource {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    match self {
      Self::BuiltIn => write!(f, "built-in (hifitime)"),
      Self::File(p) => write!(f, "LSK {p}"),
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LeapTable {
  /// strictly increasing in `utc` and in `delta_at`; never empty
  entries: Vec<LeapEntry>,
  source: LeapSource,
}

impl LeapTable {
  /// hifitime's compile-time IERS table (the pre-1972 SOFA drift entries are dropped)
  pub fn built_in() -> Self {
    use hifitime::leap_seconds::LatestLeapSeconds;
    let entries = LatestLeapSeconds::default()
      .filter(|l| l.announced_by_iers)
      // the timestamp is the UTC instant read as TAI seconds since J1900: the naive epoch
      .map(|l| LeapEntry {
        utc: Epoch::from_tai_seconds(l.timestamp_tai_s),
        delta_at: l.delta_at as i32,
      })
      .collect::<Vec<_>>();
    debug_assert!(!entries.is_empty());
    Self {
      entries,
      source: LeapSource::BuiltIn,
    }
  }

  pub fn entries(&self) -> &[LeapEntry] {
    &self.entries
  }

  pub fn source(&self) -> &LeapSource {
    &self.source
  }

  /// the newest entry
  pub fn last(&self) -> LeapEntry {
    *self.entries.last().expect("a leap table is never empty")
  }

  /// `"<source>: N entries, last ΔAT = S s at YYYY-MM-DD"`
  pub fn describe(&self) -> String {
    let last = self.last();
    let (y, m, d) = last.ymd();
    format!(
      "{}: {} entries, last ΔAT = {} s at {y:04}-{m:02}-{d:02}",
      self.source,
      self.entries.len(),
      last.delta_at
    )
  }

  /// Parses the `DELTET/DELTA_AT` block of a NAIF leap-second kernel (`naif00NN.tls`), treating
  /// `text` as untrusted: only the `( ΔAT, @YYYY-MON-D ... )` pairs inside a `\begindata` section
  /// are read, every number must parse, dates and ΔAT must be strictly increasing. `source` names
  /// the file in [`LeapSource::File`].
  pub fn parse_lsk(text: &str, source: &str) -> Result<Self, LoadStatus> {
    if text.is_empty() || text.len() > MAX_LSK_BYTES {
      return Err(LoadStatus::Unreadable);
    }
    // the assignment lives in a data section; the explanation text above it also spells
    // "DELTA_AT", so the search starts at the first `\begindata`
    let data_start = text.find("\\begindata").ok_or(LoadStatus::Malformed)?;
    let data = &text[data_start..];
    let key = data.find("DELTET/DELTA_AT").ok_or(LoadStatus::Malformed)?;
    let after_key = &data[key + "DELTET/DELTA_AT".len()..];
    let eq = after_key.find('=').ok_or(LoadStatus::Malformed)?;
    let after_eq = after_key[eq + 1..].trim_start();
    let body = after_eq.strip_prefix('(').ok_or(LoadStatus::Malformed)?;
    let close = body.find(')').ok_or(LoadStatus::Malformed)?;
    let body = &body[..close];
    if body.contains("\\begintext") {
      return Err(LoadStatus::Malformed);
    }

    let mut entries = Vec::new();
    let mut tokens = body.split(|c: char| c.is_whitespace() || c == ',').filter(|t| !t.is_empty());
    loop {
      let Some(delta) = tokens.next() else { break };
      let date = tokens.next().ok_or(LoadStatus::Malformed)?;
      let delta_at: i32 = delta.parse().map_err(|_| LoadStatus::Malformed)?;
      if !(1..=10_000).contains(&delta_at) {
        return Err(LoadStatus::Malformed);
      }
      let utc = parse_lsk_date(date).ok_or(LoadStatus::Malformed)?;
      entries.push(LeapEntry { utc, delta_at });
    }
    Self::from_entries(entries, LeapSource::File(String::from(source)))
  }

  /// A table from `entries` (strictly increasing dates and ΔAT, at least one entry).
  pub fn from_entries(entries: Vec<LeapEntry>, source: LeapSource) -> Result<Self, LoadStatus> {
    if entries.is_empty() {
      return Err(LoadStatus::Malformed);
    }
    for w in entries.windows(2) {
      if w[1].utc <= w[0].utc || w[1].delta_at <= w[0].delta_at {
        return Err(LoadStatus::Malformed);
      }
    }
    Ok(Self { entries, source })
  }

  /// Whether one table is a prefix of the other with the same values: a newer file keeps every
  /// entry of `self`, an older one (e.g. the shipped file after a download) matches `self`'s first
  /// entries. A dropped or changed entry in between is inconsistent.
  pub fn is_consistent_with(&self, candidate: &LeapTable) -> bool {
    self.entries.iter().zip(&candidate.entries).all(|(a, b)| a == b)
  }

  /// Decides whether `candidate` replaces `self`: adopted when consistent and newer (more
  /// entries, or a file equal to the built-in table); kept when consistent but not newer;
  /// rejected otherwise. Returns the status; on
  /// `Adopted` `self` is `candidate`.
  pub fn adopt(&mut self, candidate: LeapTable) -> LoadStatus {
    if !self.is_consistent_with(&candidate) {
      return LoadStatus::Inconsistent;
    }
    // a file is preferred over the compile-time table when they are the same (a user can
    // update the file without a rebuild)
    let newer = candidate.entries.len() > self.entries.len();
    let file_over_built_in = self.source == LeapSource::BuiltIn && candidate.source != LeapSource::BuiltIn;
    if newer || (file_over_built_in && candidate.entries.len() == self.entries.len()) {
      *self = candidate;
      LoadStatus::Adopted
    } else {
      LoadStatus::KeptCurrent
    }
  }

  /// Index of the last entry whose UTC date ≤ `naive` (`None` before the first entry).
  fn index_for_naive(&self, naive: Epoch) -> Option<usize> {
    self.entries.iter().rposition(|e| e.utc <= naive)
  }

  /// Index of the last entry already in force at the TAI instant `tai` (an entry applies from
  /// `utc + delta_at` in TAI); `None` before the first entry.
  fn index_for_tai(&self, tai: Epoch) -> Option<usize> {
    self
      .entries
      .iter()
      .rposition(|e| e.utc + Unit::Second * (e.delta_at as i64) <= tai)
  }

  /// Validity horizon: `max(last entry, now) + 6 months` (naive UTC epoch).
  pub fn validity_horizon(&self, now: Option<Epoch>) -> Epoch {
    let last = self.last().utc;
    let base = match now {
      Some(n) if n > last => n,
      _ => last,
    };
    base + Unit::Day * VALIDITY_MARGIN_DAYS
  }
}

/// `@YYYY-MON-D` (NAIF LSK date, month as a three-letter name) → naive midnight epoch
fn parse_lsk_date(token: &str) -> Option<Epoch> {
  let date = token.strip_prefix('@')?;
  let mut parts = date.split('-');
  let year: i32 = parts.next()?.parse().ok()?;
  let month = match parts.next()?.to_ascii_uppercase().as_str() {
    "JAN" => 1,
    "FEB" => 2,
    "MAR" => 3,
    "APR" => 4,
    "MAY" => 5,
    "JUN" => 6,
    "JUL" => 7,
    "AUG" => 8,
    "SEP" => 9,
    "OCT" => 10,
    "NOV" => 11,
    "DEC" => 12,
    _ => return None,
  };
  let day: u8 = parts.next()?.parse().ok()?;
  if parts.next().is_some() || !(1900..=9999).contains(&year) {
    return None;
  }
  Epoch::maybe_from_gregorian_tai(year, month, day, 0, 0, 0, 0).ok()
}

fn fields_valid(f: &UtcFields) -> bool {
  (1..=12).contains(&f.month)
    && (1..=31).contains(&f.day)
    && f.hour < 24
    && f.minute < 60
    && f.second <= 60
    && f.nanos < 1_000_000_000
}

/// The naive epoch (UTC digits read as TAI) of `f`, with second 60 as the next midnight.
fn naive_epoch(f: &UtcFields) -> Result<Epoch, UtcStatus> {
  if !fields_valid(f) {
    return Err(UtcStatus::Malformed);
  }
  let second = f.second.min(59);
  let naive = Epoch::maybe_from_gregorian_tai(
    f.year, f.month, f.day, f.hour, f.minute, second, f.nanos,
  )
  .map_err(|_| UtcStatus::Malformed)?;
  Ok(if f.second == 60 {
    naive + Unit::Second * 1
  } else {
    naive
  })
}

/// UTC calendar fields → TDB duration since J2000 (the engine epoch) with `table`. `now` (naive
/// UTC epoch of the wall clock) sets the validity horizon. `Err` only on malformed fields.
pub fn utc_to_tdb(
  table: &LeapTable,
  f: &UtcFields,
  now: Option<Epoch>,
) -> Result<(Duration, UtcStatus), UtcStatus> {
  let naive = naive_epoch(f)?;
  let (delta_at, mut status) = if f.second == 60 {
    // a leap second exists only where an entry starts at the next midnight (23:59:60 is the
    // last second before it, still with the previous ΔAT)
    let (y, m, d, h, mi, _, _) = naive.to_gregorian_tai();
    let midnight = Epoch::maybe_from_gregorian_tai(y, m, d, 0, 0, 0, 0).map_err(|_| UtcStatus::Malformed)?;
    if (h, mi) != (0, 0) || naive - midnight >= Unit::Second * 1 {
      return Err(UtcStatus::Malformed);
    }
    let k = table
      .entries()
      .iter()
      .position(|e| e.utc == midnight)
      .ok_or(UtcStatus::Malformed)?;
    if k == 0 {
      return Err(UtcStatus::Malformed);
    }
    (table.entries()[k - 1].delta_at, UtcStatus::Ok)
  } else {
    match table.index_for_naive(naive) {
      Some(i) => (table.entries()[i].delta_at, UtcStatus::Ok),
      None => (table.entries()[0].delta_at, UtcStatus::BeforeLeapSeconds),
    }
  };
  if status == UtcStatus::Ok && naive > table.validity_horizon(now) {
    status = UtcStatus::BeyondLskValidity;
  }
  let tai = naive + Unit::Second * (delta_at as i64);
  Ok((tai.to_tdb_duration(), status))
}

/// Fields of the naive epoch `naive` after subtracting ΔAT of entry `i`, folding the leap second
/// (`naive` on or after the next entry's date while that entry is not in force yet → 23:59:60).
fn fields_from_naive(table: &LeapTable, naive: Epoch, i: usize) -> UtcFields {
  let next = table.entries().get(i + 1);
  let in_leap_second = next.is_some_and(|n| naive >= n.utc);
  let shown = if in_leap_second {
    naive - Unit::Second * 1
  } else {
    naive
  };
  let (y, m, d, h, mi, s, ns) = shown.to_gregorian_tai();
  UtcFields {
    year: y,
    month: m,
    day: d,
    hour: h,
    minute: mi,
    second: if in_leap_second { 60 } else { s },
    nanos: ns,
  }
}

/// TDB duration since J2000 (the engine epoch) → UTC calendar fields with `table`. The result
/// round-trips through [`utc_to_tdb`] to the same nanosecond (one refinement step absorbs the
/// rounding of hifitime's periodic TDB term). Never fails: before 1972 the fields are a nominal
/// extrapolation with [`UtcStatus::BeforeLeapSeconds`].
pub fn tdb_to_utc(table: &LeapTable, tdb: Duration, now: Option<Epoch>) -> (UtcFields, UtcStatus) {
  let tai = Epoch::from_tdb_duration(tdb).to_time_scale(hifitime::TimeScale::TAI);
  let (i, mut status) = match table.index_for_tai(tai) {
    Some(i) => (i, UtcStatus::Ok),
    None => (0, UtcStatus::BeforeLeapSeconds),
  };
  let delta_at = table.entries()[i].delta_at;
  let mut naive = tai - Unit::Second * (delta_at as i64);
  let mut fields = fields_from_naive(table, naive, i);
  // refine: the forward conversion is the reference; shift the naive instant by the residual
  if let Ok((back, _)) = utc_to_tdb(table, &fields, None) {
    let residual = tdb.total_nanoseconds() - back.total_nanoseconds();
    if residual != 0 && residual.abs() < 1_000_000 {
      naive = naive + Duration::from_total_nanoseconds(residual);
      fields = fields_from_naive(table, naive, i);
    }
  }
  if status == UtcStatus::Ok && naive > table.validity_horizon(now) {
    status = UtcStatus::BeyondLskValidity;
  }
  (fields, status)
}

// ── the active table ───────────────────────────────────────────────────────────────────────────

struct Active {
  table: Option<LeapTable>,
  /// naive UTC epoch of the host's wall clock ([`set_now`])
  now: Option<Epoch>,
}

static ACTIVE: parking_lot::RwLock<Active> = parking_lot::RwLock::new(Active {
  table: None,
  now: None,
});

/// Runs `f` on the active table (built-in until a file is adopted).
pub fn with_active<R>(f: impl FnOnce(&LeapTable, Option<Epoch>) -> R) -> R {
  let guard = ACTIVE.upgradable_read();
  if let Some(t) = guard.table.as_ref() {
    return f(t, guard.now);
  }
  let mut w = parking_lot::RwLockUpgradableReadGuard::upgrade(guard);
  let now = w.now;
  f(w.table.insert(LeapTable::built_in()), now)
}

/// The host's wall clock as UTC calendar fields (sets the validity horizon, see the module doc).
pub fn set_now(f: &UtcFields) -> Result<(), UtcStatus> {
  let naive = naive_epoch(f)?;
  ACTIVE.write().now = Some(naive);
  Ok(())
}

/// Loads a NAIF LSK from `path` (bounded read, untrusted text) and offers it to the active table
/// ([`LeapTable::adopt`]); logs the choice. Returns the status and the file's entry count
/// (0 when it did not parse).
pub fn load_file(path: &str) -> (LoadStatus, usize) {
  let Ok(bytes) = aethervk_oshal_rlib::os::fs::read(path) else {
    aethervk_oshal_rlib::log!("[leap seconds] {path}: unreadable, keeping {}", with_active(|t, _| t.describe()));
    return (LoadStatus::Unreadable, 0);
  };
  if bytes.is_empty() || bytes.len() > MAX_LSK_BYTES {
    aethervk_oshal_rlib::log!(
      "[leap seconds] {path}: {} bytes (limit {MAX_LSK_BYTES}), keeping {}",
      bytes.len(),
      with_active(|t, _| t.describe())
    );
    return (LoadStatus::Unreadable, 0);
  }
  let text = String::from_utf8_lossy(&bytes);
  load_text(&text, path)
}

/// [`load_file`] on already-read text (`source` names it in the log and in [`LeapSource`]).
pub fn load_text(text: &str, source: &str) -> (LoadStatus, usize) {
  let candidate = match LeapTable::parse_lsk(text, source) {
    Ok(t) => t,
    Err(status) => {
      aethervk_oshal_rlib::log!(
        "[leap seconds] {source}: {status:?}, keeping {}",
        with_active(|t, _| t.describe())
      );
      return (status, 0);
    }
  };
  let count = candidate.entries().len();
  let candidate_desc = candidate.describe();
  with_active(|_, _| ());
  let mut w = ACTIVE.write();
  let active = w.table.as_mut().expect("initialised by with_active");
  let status = active.adopt(candidate);
  match status {
    LoadStatus::Adopted => aethervk_oshal_rlib::log!("[leap seconds] using {candidate_desc}"),
    LoadStatus::KeptCurrent => aethervk_oshal_rlib::log!(
      "[leap seconds] {candidate_desc} is not newer; keeping {}",
      active.describe()
    ),
    _ => aethervk_oshal_rlib::log!(
      "[leap seconds] {candidate_desc} rejected ({status:?}); keeping {}",
      active.describe()
    ),
  }
  (status, count)
}

/// Replaces the active table (tests).
#[cfg(any(test, feature = "std"))]
pub fn set_active_for_tests(table: LeapTable) {
  ACTIVE.write().table = Some(table);
}

/// Loads the newest `naif00NN.tls` of `<asset_dir>/planets` (highest NN), the shipped LSK;
/// logs the table in use either way.
pub fn init_from_asset_dir(asset_dir: &str) {
  let dir = format!("{asset_dir}/planets");
  let mut best: Option<(u32, String)> = None;
  if let Ok(entries) = aethervk_oshal_rlib::os::fs::read_dir(dir.as_str()) {
    for entry in entries.flatten() {
      let path = entry.path();
      let Some(path_str) = path.to_str_unified().map(|c| String::from(c.as_ref())) else {
        continue;
      };
      let name = path_str.rsplit(['/', '\\']).next().unwrap_or("");
      if let Some(nn) = name
        .strip_prefix("naif")
        .and_then(|r| r.strip_suffix(".tls"))
        .and_then(|n| n.parse::<u32>().ok())
        && best.as_ref().is_none_or(|(b, _)| nn > *b)
      {
        best = Some((nn, path_str));
      }
    }
  }
  match best {
    Some((_, path)) => {
      let _ = load_file(&path);
    }
    None => aethervk_oshal_rlib::log!(
      "[leap seconds] no naif00NN.tls in {dir}; using {}",
      with_active(|t, _| t.describe())
    ),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// the tests that replace the process-wide active table run one at a time
  static ACTIVE_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

  /// the shipped kernel, beside `pck00011.pca` (paths as `astrometry.rs` tests)
  const SHIPPED_LSK: &str = "../../assets/planets/naif0012.tls";

  fn shipped() -> LeapTable {
    let text = std::fs::read_to_string(SHIPPED_LSK).expect(SHIPPED_LSK);
    LeapTable::parse_lsk(&text, "naif0012.tls").expect("shipped LSK parses")
  }

  fn fields(y: i32, mo: u8, d: u8, h: u8, mi: u8, s: u8, ns: u32) -> UtcFields {
    UtcFields {
      year: y,
      month: mo,
      day: d,
      hour: h,
      minute: mi,
      second: s,
      nanos: ns,
    }
  }

  /// a far-future wall clock so no test epoch is beyond the validity horizon
  fn now() -> Option<Epoch> {
    Some(Epoch::from_gregorian_tai_at_midnight(2100, 1, 1))
  }

  #[test]
  fn lsk_shipped_file_has_28_entries_last_37_at_2017() {
    let t = shipped();
    assert_eq!(t.entries().len(), 28);
    let last = t.last();
    assert_eq!(last.delta_at, 37);
    assert_eq!(last.ymd(), (2017, 1, 1));
    assert_eq!(t.entries()[0].delta_at, 10);
    assert_eq!(t.entries()[0].ymd(), (1972, 1, 1));
    assert_eq!(t.source(), &LeapSource::File(String::from("naif0012.tls")));
  }

  #[test]
  fn lsk_and_built_in_table_agree() {
    let (lsk, built_in) = (shipped(), LeapTable::built_in());
    assert_eq!(built_in.last().delta_at, 37);
    assert_eq!(lsk.entries(), built_in.entries());
    assert!(lsk.is_consistent_with(&built_in) && built_in.is_consistent_with(&lsk));
  }

  #[test]
  fn lsk_truncated_or_garbled_falls_back_to_built_in() {
    let _serial = ACTIVE_LOCK.lock();
    let text = std::fs::read_to_string(SHIPPED_LSK).unwrap();
    // cut inside the DELTA_AT block: no closing parenthesis
    let cut = text.find("@2006-JAN-1").unwrap();
    assert_eq!(LeapTable::parse_lsk(&text[..cut], "cut"), Err(LoadStatus::Malformed));
    // a non-numeric ΔAT
    let garbled = text.replace("33,   @2006-JAN-1", "3x,   @2006-JAN-1");
    assert_eq!(LeapTable::parse_lsk(&garbled, "garbled"), Err(LoadStatus::Malformed));
    // ΔAT going backwards
    let backwards = text.replace("33,   @2006-JAN-1", "31,   @2006-JAN-1");
    assert_eq!(LeapTable::parse_lsk(&backwards, "backwards"), Err(LoadStatus::Malformed));
    // no data section at all / empty / oversized
    assert_eq!(LeapTable::parse_lsk("DELTET/DELTA_AT = ( 10, @1972-JAN-1 )", "nodata"), Err(LoadStatus::Malformed));
    assert_eq!(LeapTable::parse_lsk("", "empty"), Err(LoadStatus::Unreadable));
    let big = alloc::vec![b' '; MAX_LSK_BYTES + 1];
    assert_eq!(LeapTable::parse_lsk(core::str::from_utf8(&big).unwrap(), "big"), Err(LoadStatus::Unreadable));

    // the loader keeps the active (built-in) table and reports the rejection
    set_active_for_tests(LeapTable::built_in());
    assert_eq!(load_text(&text[..cut], "cut"), (LoadStatus::Malformed, 0));
    assert_eq!(load_file("../../assets/planets/does_not_exist.tls"), (LoadStatus::Unreadable, 0));
    with_active(|t, _| {
      assert_eq!(t.source(), &LeapSource::BuiltIn);
      assert_eq!(t.last().delta_at, 37);
    });
  }

  #[test]
  fn utc_to_tdb_2021_12_27_offset_is_69_184_s() {
    let t = shipped();
    let f = fields(2021, 12, 27, 0, 14, 47, 0);
    let (tdb, status) = utc_to_tdb(&t, &f, now()).unwrap();
    assert_eq!(status, UtcStatus::Ok);
    // TDB − UTC: the same digits read as TDB are this long past J2000 (2000-01-01T12:00:00 TDB)
    let naive_as_tdb = Epoch::from_gregorian_tai(2021, 12, 27, 0, 14, 47, 0)
      - Epoch::from_gregorian_tai_at_noon(2000, 1, 1);
    let offset_s = (tdb - naive_as_tdb).to_seconds();
    assert!((offset_s - 69.184).abs() < 2e-3, "TDB − UTC = {offset_s}");
    // and hifitime's own built-in conversion agrees to the nanosecond
    let hifi = Epoch::from_gregorian_utc(2021, 12, 27, 0, 14, 47, 0).to_tdb_duration();
    assert_eq!(tdb, hifi);
  }

  #[test]
  fn tdb_utc_round_trip_is_exact_at_several_epochs() {
    let t = shipped();
    let samples = [
      Epoch::from_gregorian_utc(1972, 1, 1, 0, 0, 1, 0),
      Epoch::from_gregorian_utc(1999, 12, 31, 23, 59, 59, 999_999_999),
      Epoch::from_gregorian_utc(2000, 1, 1, 11, 58, 55, 816_000_000),
      Epoch::from_gregorian_utc(2016, 12, 31, 23, 59, 59, 500_000_000),
      Epoch::from_gregorian_utc(2017, 1, 1, 0, 0, 0, 0),
      Epoch::from_gregorian_utc(2021, 12, 27, 0, 14, 47, 123_456_789),
      Epoch::from_gregorian_utc(2026, 10, 30, 12, 0, 0, 1),
    ];
    for e in samples {
      for jitter in [0i128, 1, -1, 333, 999_999_999] {
        let tdb = e.to_tdb_duration() + Duration::from_total_nanoseconds(jitter);
        let (f, status) = tdb_to_utc(&t, tdb, now());
        assert_eq!(status, UtcStatus::Ok, "{e}");
        let (back, _) = utc_to_tdb(&t, &f, now()).unwrap();
        assert_eq!(back.total_nanoseconds(), tdb.total_nanoseconds(), "{e} {f:?}");
        // and the (centuries, ns) parts the engine stores are the same
        assert_eq!(back.to_parts(), tdb.to_parts());
      }
    }
  }

  #[test]
  fn leap_second_2016_12_31_23_59_60_parses_and_is_one_second_before_2017() {
    let t = shipped();
    let (before, _) = utc_to_tdb(&t, &fields(2016, 12, 31, 23, 59, 59, 0), now()).unwrap();
    let (leap, s_leap) = utc_to_tdb(&t, &fields(2016, 12, 31, 23, 59, 60, 0), now()).unwrap();
    let (after, _) = utc_to_tdb(&t, &fields(2017, 1, 1, 0, 0, 0, 0), now()).unwrap();
    assert_eq!(s_leap, UtcStatus::Ok);
    assert_eq!((leap - before).total_nanoseconds(), 1_000_000_000);
    assert_eq!((after - leap).total_nanoseconds(), 1_000_000_000);
    // the leap second shows as 23:59:60 on the way back, and the day after as 00:00:00
    let (f, _) = tdb_to_utc(&t, leap + Duration::from_total_nanoseconds(250_000_000), now());
    assert_eq!(f, fields(2016, 12, 31, 23, 59, 60, 250_000_000));
    let (f, _) = tdb_to_utc(&t, after, now());
    assert_eq!(f, fields(2017, 1, 1, 0, 0, 0, 0));
    // ΔAT steps from 36 to 37 across it
    let tai = |d: Duration| Epoch::from_tdb_duration(d).to_tai_duration();
    let naive = |f: UtcFields| Epoch::from_gregorian_tai(f.year, f.month, f.day, f.hour, f.minute, f.second.min(59), f.nanos).to_tai_duration();
    assert_eq!((tai(before) - naive(fields(2016, 12, 31, 23, 59, 59, 0))).to_seconds().round() as i32, 36);
    assert_eq!((tai(after) - naive(fields(2017, 1, 1, 0, 0, 0, 0))).to_seconds().round() as i32, 37);
    // 60 s where there is no leap second is malformed, as are impossible fields
    assert_eq!(utc_to_tdb(&t, &fields(2016, 6, 30, 23, 59, 60, 0), now()), Err(UtcStatus::Malformed));
    assert_eq!(utc_to_tdb(&t, &fields(2016, 12, 31, 23, 58, 60, 0), now()), Err(UtcStatus::Malformed));
    assert_eq!(utc_to_tdb(&t, &fields(2021, 13, 1, 0, 0, 0, 0), now()), Err(UtcStatus::Malformed));
    assert_eq!(utc_to_tdb(&t, &fields(2021, 2, 30, 0, 0, 0, 0), now()), Err(UtcStatus::Malformed));
    assert_eq!(utc_to_tdb(&t, &fields(2021, 1, 1, 0, 0, 0, 1_000_000_000), now()), Err(UtcStatus::Malformed));
  }

  #[test]
  fn before_1972_returns_status_1_both_ways() {
    let t = shipped();
    let (tdb, status) = utc_to_tdb(&t, &fields(1971, 12, 31, 23, 59, 59, 0), now()).unwrap();
    assert_eq!(status, UtcStatus::BeforeLeapSeconds);
    let (f, status) = tdb_to_utc(&t, tdb, now());
    assert_eq!(status, UtcStatus::BeforeLeapSeconds);
    assert_eq!(f, fields(1971, 12, 31, 23, 59, 59, 0));
    // J2000 TDB itself is fine
    assert_eq!(tdb_to_utc(&t, Duration::ZERO, now()).1, UtcStatus::Ok);
  }

  #[test]
  fn beyond_validity_is_a_warning_with_a_value() {
    let t = shipped();
    // no wall clock: horizon = 2017-01-01 + 6 months
    let (a, s) = utc_to_tdb(&t, &fields(2017, 6, 1, 0, 0, 0, 0), None).unwrap();
    assert_eq!(s, UtcStatus::Ok);
    let (b, s) = utc_to_tdb(&t, &fields(2018, 1, 1, 0, 0, 0, 0), None).unwrap();
    assert_eq!(s, UtcStatus::BeyondLskValidity);
    assert_eq!(tdb_to_utc(&t, b, None).1, UtcStatus::BeyondLskValidity);
    assert_eq!(tdb_to_utc(&t, a, None).1, UtcStatus::Ok);
    // the wall clock moves the horizon: 2018 is fine when now = 2021, 2022 is not
    let now = Some(Epoch::from_gregorian_tai_at_midnight(2021, 6, 1));
    assert_eq!(utc_to_tdb(&t, &fields(2018, 1, 1, 0, 0, 0, 0), now).unwrap().1, UtcStatus::Ok);
    assert_eq!(utc_to_tdb(&t, &fields(2022, 1, 1, 0, 0, 0, 0), now).unwrap().1, UtcStatus::BeyondLskValidity);
    // the value is still the no-further-leap-second one
    let hifi = Epoch::from_gregorian_utc(2018, 1, 1, 0, 0, 0, 0).to_tdb_duration();
    assert_eq!(b, hifi);
  }

  /// the shipped text with one extra entry appended
  fn newer_lsk_text() -> String {
    let text = std::fs::read_to_string(SHIPPED_LSK).unwrap();
    text.replace("37,   @2017-JAN-1 )", "37,   @2017-JAN-1\n                           38,   @2030-JUL-1 )")
  }

  #[test]
  fn load_newer_lsk_with_extra_entry_is_adopted() {
    let _serial = ACTIVE_LOCK.lock();
    set_active_for_tests(shipped());
    let (status, count) = load_text(&newer_lsk_text(), "naif0013.tls");
    assert_eq!((status, count), (LoadStatus::Adopted, 29));
    with_active(|t, _| {
      assert_eq!(t.source(), &LeapSource::File(String::from("naif0013.tls")));
      assert_eq!(t.last().delta_at, 38);
      assert_eq!(t.last().ymd(), (2030, 7, 1));
      // the new leap second converts
      assert_eq!(utc_to_tdb(t, &fields(2030, 6, 30, 23, 59, 60, 0), now()).unwrap().1, UtcStatus::Ok);
    });
    // loading the shipped file again keeps the newer one
    let shipped_text = std::fs::read_to_string(SHIPPED_LSK).unwrap();
    assert_eq!(load_text(&shipped_text, "naif0012.tls"), (LoadStatus::KeptCurrent, 28));
    with_active(|t, _| assert_eq!(t.last().delta_at, 38));
  }

  #[test]
  fn load_lsk_that_drops_or_changes_an_entry_is_rejected() {
    let _serial = ACTIVE_LOCK.lock();
    set_active_for_tests(shipped());
    // LF: a Windows checkout (core.autocrlf) has the kernel in CRLF, the patterns below are LF
    let text = std::fs::read_to_string(SHIPPED_LSK).unwrap().replace("\r\n", "\n");
    // changed value of an old entry (and an extra entry, so it is "newer")
    let changed = newer_lsk_text().replace("33,   @2006-JAN-1", "33,   @2006-JUL-1");
    assert_eq!(load_text(&changed, "changed"), (LoadStatus::Inconsistent, 29));
    // dropped entry
    let dropped = text.replace("                           33,   @2006-JAN-1\n", "");
    assert_eq!(load_text(&dropped, "dropped"), (LoadStatus::Inconsistent, 27));
    // the shipped file with its last entry cut is older, not inconsistent
    let older = text.replace("                           37,   @2017-JAN-1 )", " )");
    assert_eq!(load_text(&older, "older"), (LoadStatus::KeptCurrent, 27));
    with_active(|t, _| {
      assert_eq!(t.source(), &LeapSource::File(String::from("naif0012.tls")));
      assert_eq!(t.entries().len(), 28);
    });
  }

  #[test]
  fn load_oversized_or_garbled_lsk_keeps_current_table() {
    let _serial = ACTIVE_LOCK.lock();
    set_active_for_tests(shipped());
    let mut big = newer_lsk_text();
    big.push_str(&" ".repeat(MAX_LSK_BYTES));
    assert_eq!(load_text(&big, "big"), (LoadStatus::Unreadable, 0));
    assert_eq!(load_text("\\begindata\nDELTET/DELTA_AT = ( 10, @1972-JAN-1, 11, @1972-FOO-1 )", "garbled"), (LoadStatus::Malformed, 0));
    assert_eq!(load_text("\\begindata\nDELTET/DELTA_AT = ( 10, @1972-JAN-1, 1e3, @1973-JAN-1 )", "garbled2"), (LoadStatus::Malformed, 0));
    with_active(|t, _| {
      assert_eq!(t.source(), &LeapSource::File(String::from("naif0012.tls")));
      assert_eq!(t.entries().len(), 28);
    });
  }

  #[test]
  fn init_from_asset_dir_loads_the_shipped_lsk() {
    let _serial = ACTIVE_LOCK.lock();
    set_active_for_tests(LeapTable::built_in());
    init_from_asset_dir("../../assets");
    with_active(|t, _| {
      assert!(matches!(t.source(), LeapSource::File(p) if p.ends_with("naif0012.tls")), "{}", t.describe());
      assert_eq!(t.entries().len(), 28);
    });
  }
}
