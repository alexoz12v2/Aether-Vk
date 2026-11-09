//! The IRAF `WATi_nnn` attribute strings of TNX / ZPX headers (FITS registry "TNX" and "ZPX WCS"
//! entries; IRAF `mwcs`): the cards `WATi_001`, `WATi_002`, … of axis `i` hold one string cut
//! into 68-character pieces, `attribute=value` pairs separated by blanks, a value either a bare
//! token (`wtype=tnx`, `projp3=337.74`) or quoted (`lngcor = "3. 4. 4. 2. -0.3 0.3 …"`).
//!
//! IRAF cuts the string at exactly 68 characters whatever falls there, so a piece may end in the
//! blank between two numbers; FITS readers (this crate's header parser among them) strip the
//! trailing blanks of a string value, which would glue the two numbers together. Every piece but
//! the last is therefore padded back to 68 characters before concatenation.

use super::super::FitsHeader;
use alloc::{format, string::String, vec::Vec};

/// Length of the string value of an IRAF `WAT` card.
pub const WAT_CHUNK: usize = 68;

/// The concatenated `WATi_nnn` string of axis `i` (empty without any card).
pub fn wat_string(h: &FitsHeader, axis: u8) -> String {
  let prefix = format!("WAT{axis}_");
  let mut pieces: Vec<(u32, &str)> = Vec::new();
  for (k, v) in h.cards.iter() {
    if let Some(n) = k.strip_prefix(&prefix)
      && let Ok(n) = n.parse::<u32>()
    {
      pieces.push((n, v.as_str()));
    }
  }
  pieces.sort_by_key(|p| p.0);
  let mut s = String::new();
  let last = pieces.len();
  for (i, (_, v)) in pieces.iter().enumerate() {
    s.push_str(v);
    if i + 1 < last {
      for _ in v.chars().count()..WAT_CHUNK {
        s.push(' ');
      }
    }
  }
  s
}

/// The value of `key` in an attribute string: the token after `key=` (blanks allowed around the
/// `=`), the quoted text when it starts with `"`.
pub fn wat_attr<'a>(s: &'a str, key: &str) -> Option<&'a str> {
  let bytes = s.as_bytes();
  let mut start = 0;
  while let Some(pos) = s[start..].find(key) {
    let at = start + pos;
    start = at + 1;
    // a token boundary before the key
    if at > 0 && !bytes[at - 1].is_ascii_whitespace() && bytes[at - 1] != b'"' {
      continue;
    }
    let mut i = at + key.len();
    while i < bytes.len() && bytes[i] == b' ' {
      i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'=' {
      continue;
    }
    i += 1;
    while i < bytes.len() && bytes[i] == b' ' {
      i += 1;
    }
    if i >= bytes.len() {
      return Some("");
    }
    if bytes[i] == b'"' {
      let rest = &s[i + 1..];
      return Some(match rest.find('"') {
        Some(e) => &rest[..e],
        None => rest,
      });
    }
    let rest = &s[i..];
    let e = rest.find(|c: char| c.is_ascii_whitespace()).unwrap_or(rest.len());
    return Some(&rest[..e]);
  }
  None
}

/// A numeric attribute (`projp3=337.74`, `ro=57.29…`).
pub fn wat_attr_f64(s: &str, key: &str) -> Option<f64> {
  wat_attr(s, key).and_then(parse_number)
}

/// A number as IRAF / wcstools write them: `3.`, `-0.3`, `1.5E-05`, `1.5D-05`.
pub fn parse_number(t: &str) -> Option<f64> {
  let t = t.trim();
  if t.is_empty() {
    return None;
  }
  let t = t.replace(['D', 'd'], "E");
  let t = t.strip_suffix('.').unwrap_or(&t);
  t.parse::<f64>().ok().or_else(|| format!("{t}0").parse::<f64>().ok())
}

/// The blank-separated numbers of a quoted surface value.
pub fn parse_numbers(s: &str) -> Result<Vec<f64>, String> {
  let mut out = Vec::new();
  for t in s.split_ascii_whitespace() {
    out.push(parse_number(t).ok_or_else(|| format!("WAT surface: '{t}' is not a number"))?);
  }
  Ok(out)
}
