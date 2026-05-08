use base64::Engine;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize, Serialize)]
pub struct SbdbQueryResponse {
  pub fields: Vec<String>,
  pub data: Vec<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct HorizonsJsonResponse {
  pub result: Option<String>,
  pub spk: Option<String>,
  pub error: Option<String>,
}

pub struct HorizonJplService {
  client: Client,
}

impl HorizonJplService {
  pub fn new() -> Result<Self, reqwest::Error> {
    let client = Client::builder()
      .user_agent("AetherVk/1.0")
      .timeout(std::time::Duration::from_secs(30))
      .build()?;
    Ok(Self { client })
  }

  /// List all comets
  pub fn list_comets(&self) -> Result<SbdbQueryResponse, Box<dyn std::error::Error>> {
    let url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api?fields=full_name,pdes&sb-kind=c";
    let resp = self.client.get(url).send()?;
    if !resp.status().is_success() {
      return Err(format!("API request failed with status: {}", resp.status()).into());
    }
    let json = resp.json::<SbdbQueryResponse>()?;
    Ok(json)
  }

  /// List comets with first observation between start_time and stop_time (e.g. "2023-01-01")
  pub fn list_comets_with_time(
    &self,
    start_time: &str,
    stop_time: &str,
  ) -> Result<SbdbQueryResponse, Box<dyn std::error::Error>> {
    let cdata = format!(
      "{{\"AND\":[\"first_obs|GE|{}\",\"first_obs|LE|{}\"]}}",
      start_time, stop_time
    );
    let url = format!(
      "https://ssd-api.jpl.nasa.gov/sbdb_query.api?sb-kind=c&fields=full_name,pdes&sb-cdata={}",
      urlencoding::encode(&cdata)
    );
    let resp = self.client.get(&url).send()?;
    if !resp.status().is_success() {
      return Err(format!("API request failed with status: {}", resp.status()).into());
    }
    let json = resp.json::<SbdbQueryResponse>()?;
    Ok(json)
  }

  /// Enumerate SPK records for a designator
  pub fn enumerate_spk(
    &self,
    designator: &str,
    start_time: &str,
    stop_time: &str,
  ) -> Result<Vec<Vec<String>>, Box<dyn std::error::Error>> {
    let command = format!("'DES={};'", designator);
    let url = format!(
      "https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND={}&EPHEM_TYPE='SPK'&START_TIME='{}'&STOP_TIME='{}'&MAKE_EPHEM='YES'",
      urlencoding::encode(&command),
      urlencoding::encode(start_time),
      urlencoding::encode(stop_time)
    );
    let resp = self.client.get(&url).send()?;
    if !resp.status().is_success() {
      return Err(format!("API request failed with status: {}", resp.status()).into());
    }
    let json = resp.json::<HorizonsJsonResponse>()?;

    let mut records = Vec::new();
    if let Some(result_str) = json.result {
      let lines: Vec<&str> = result_str.lines().collect();
      let mut in_data = false;
      for line in lines {
        if line.trim().starts_with("--------") {
          in_data = true;
          continue;
        }
        if in_data {
          if line.trim().starts_with("*") || line.trim().is_empty() {
            break;
          }
          let parts: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();
          if parts.len() >= 5 {
            records.push(parts);
          }
        }
      }
    }
    Ok(records)
  }

  /// Download an SPK file
  pub fn download_spk<P: AsRef<Path>>(
    &self,
    spk_id: &str,
    start_time: &str,
    stop_time: &str,
    save_dir: P,
    filename: &str,
  ) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let command = format!("'{};'", spk_id);
    let url = format!(
      "https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND={}&OBJ_DATA='NO'&MAKE_EPHEM='YES'&EPHEM_TYPE='SPK'&START_TIME='{}'&STOP_TIME='{}'",
      urlencoding::encode(&command),
      urlencoding::encode(start_time),
      urlencoding::encode(stop_time)
    );

    let resp = self.client.get(&url).send()?;
    if !resp.status().is_success() {
      return Err(format!("API request failed with status: {}", resp.status()).into());
    }
    let json = resp.json::<HorizonsJsonResponse>()?;

    if let Some(error) = json.error {
      return Err(format!("API Error: {}", error).into());
    }

    if let Some(spk_b64) = json.spk {
      let binary_spk = base64::engine::general_purpose::STANDARD.decode(spk_b64)?;
      std::fs::create_dir_all(&save_dir)?;
      let file_path = save_dir.as_ref().join(filename);
      let mut file = File::create(&file_path)?;
      file.write_all(&binary_spk)?;
      Ok(file_path)
    } else {
      Err("No SPK data found in response".into())
    }
  }

  /// Download object constants
  pub fn download_object_constants(
      &self,
      command_id: &str,
  ) -> Result<String, Box<dyn std::error::Error>> {
      let command = format!("'{}'", command_id); // Removed the forced ';' so we can pass '399' or 'DES=1000012;' as needed
      let url = format!(
          "https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND={}&MAKE_EPHEM='NO'&OBJ_DATA='YES'",
          urlencoding::encode(&command)
      );
      let resp = self.client.get(&url).send()?;
      if !resp.status().is_success() {
          return Err(format!("API request failed with status: {}", resp.status()).into());
      }
      let json = resp.json::<HorizonsJsonResponse>()?;

      if let Some(result) = json.result {
          Ok(result)
      } else if let Some(error) = json.error {
          Err(format!("API Error: {}", error).into())
      } else {
          Err("No result or error in response".into())
      }
  }

  pub fn parse_object_constants(&self, text: &str) -> (Option<f32>, Option<f32>, Option<f32>) {
      let mut radius_km = None;
      let mut rot_period_hours = None;
      let mut mass_kg = None;

      for line in text.lines() {
          let line_lower = line.to_lowercase();
          // Parse Radius
          if radius_km.is_none() {
              if let Some(idx) = line_lower.find("rad=") {
                  let rem = &line[idx + 4..];
                  if let Some(val) = rem.split_whitespace().next() {
                      radius_km = val.parse::<f32>().ok();
                  }
              } else if let Some(idx) = line_lower.find("equ. radius, km") {
                  if let Some(eq_idx) = line_lower[idx..].find('=') {
                      let rem = &line[idx + eq_idx + 1..];
                      if let Some(val) = rem.split_whitespace().next() {
                          radius_km = val.parse::<f32>().ok();
                      }
                  }
              }
          }

          // Parse Rotational Period
          if rot_period_hours.is_none() {
              if let Some(idx) = line_lower.find("rotper=") {
                  let rem = &line[idx + 7..];
                  if let Some(val) = rem.split_whitespace().next() {
                      rot_period_hours = val.parse::<f32>().ok();
                  }
              } else if let Some(idx) = line_lower.find("rot. rate (rad/s)") {
                  if let Some(eq_idx) = line_lower[idx..].find('=') {
                      let rem = &line[idx + eq_idx + 1..];
                      if let Some(val) = rem.split_whitespace().next() {
                          if let Ok(rad_per_s) = val.parse::<f32>() {
                              rot_period_hours = Some(2.0 * std::f32::consts::PI / rad_per_s / 3600.0);                          }
                      }
                  }
              } else if let Some(idx) = line_lower.find("mean sidereal day, hr") {
                  if let Some(eq_idx) = line_lower[idx..].find('=') {
                      let rem = &line[idx + eq_idx + 1..];
                      if let Some(val) = rem.split_whitespace().next() {
                          rot_period_hours = val.parse::<f32>().ok();
                      }
                  }
              }
          }

          // Parse Mass
          if mass_kg.is_none() {
              if let Some(idx) = line_lower.find("mass x10^24 (kg)") {
                  if let Some(eq_idx) = line_lower[idx..].find('=') {
                      let rem = &line[idx + eq_idx + 1..];
                      if let Some(val) = rem.split_whitespace().next() {
                          // Some values are like 5.97219+-0.0006, take the part before +-
                          let val = val.split("+-").next().unwrap_or(val);
                          if let Ok(mass) = val.parse::<f32>() {
                              mass_kg = Some(mass * 1e24);
                          }
                      }
                  }
              } else if let Some(idx) = line_lower.find("gm=") {
                  // Try to deduce mass from GM
                  let rem = &line[idx + 3..];
                  if let Some(val) = rem.split_whitespace().next() {
                      if let Ok(gm) = val.parse::<f32>() {
                          let g = 6.67430e-20; // km^3 / (kg * s^2)
                          mass_kg = Some(gm / g);
                      }
                  }
              } else if let Some(idx) = line_lower.find("gm, km^3/s^2") {
                  if let Some(eq_idx) = line_lower[idx..].find('=') {
                      let rem = &line[idx + eq_idx + 1..];
                      if let Some(val) = rem.split_whitespace().next() {
                          if let Ok(gm) = val.parse::<f32>() {
                              let g = 6.67430e-20; // km^3 / (kg * s^2)
                              mass_kg = Some(gm / g);
                          }
                      }
                  }
              }
          }
      }
      (radius_km, rot_period_hours, mass_kg)
  }
  }
